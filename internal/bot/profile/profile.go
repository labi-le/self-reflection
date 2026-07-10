// Package profile maintains a compact, always-injected memory of the person the
// persona talks to. It periodically re-derives STABLE facts about the
// interlocutor from the conversation's intro + recent turns via an LLM extractor,
// so identity and key facts survive after they slide off the recent-window
// horizon — which is exactly what per-message vector recall failed to deliver.
package profile

import (
	"context"
	"fmt"
	"strings"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/pkg/ctxlog"
)

const (
	defaultEvery   = 15
	defaultIntroN  = 50
	defaultRecentN = 50
)

// Store is the subset of the SQLite state the builder reads and writes.
type Store interface {
	CountMessages(chatID int64) (int, error)
	GetProfile(chatID int64) (string, int, error)
	SaveProfile(chatID int64, profile string, count int) error
	LoadFirst(chatID int64, n int) ([]bot.Message, error)
	LoadRecent(chatID int64, n int) ([]bot.Message, error)
}

// Builder re-derives and stores the interlocutor profile when enough new
// messages have accumulated since the last build.
type Builder struct {
	store     Store
	extractor Extractor
	every     int
	introN    int
	recentN   int
	logger    zerolog.Logger
}

// Config tunes the builder. Every is the message-count gap between rebuilds.
type Config struct {
	Every  int
	Logger zerolog.Logger
}

// NewBuilder wires a builder over a store and an extractor.
func NewBuilder(store Store, extractor Extractor, cfg Config) *Builder {
	every := cfg.Every
	if every <= 0 {
		every = defaultEvery
	}
	return &Builder{
		store:     store,
		extractor: extractor,
		every:     every,
		introN:    defaultIntroN,
		recentN:   defaultRecentN,
		logger:    cfg.Logger.With().Str("component", "bot-profile").Logger(),
	}
}

// Maybe rebuilds chatID's profile when the message count has grown by at least
// `every` since the last build, or when there is history but no profile yet.
// Best-effort: failures are logged and swallowed. Meant to run out-of-band (a
// goroutine) so it never delays a reply.
func (b *Builder) Maybe(ctx context.Context, chatID int64) {
	if b == nil || b.store == nil || b.extractor == nil {
		return
	}
	log := ctxlog.Op(b.logger, "Builder.Maybe").With().Int64("chat_id", chatID).Logger()
	count, err := b.store.CountMessages(chatID)
	if err != nil || count == 0 {
		return
	}
	current, builtAt, err := b.store.GetProfile(chatID)
	if err != nil {
		return
	}
	if current != "" && count-builtAt < b.every {
		return // not enough new messages since the last build
	}
	if err := b.build(ctx, chatID, current, count); err != nil {
		log.Warn().Err(err).Msg("profile build failed")
	}
}

// build derives an updated profile from the intro + recent turns and stores it.
// It never wipes a good profile on an empty extraction result.
func (b *Builder) build(ctx context.Context, chatID int64, current string, count int) error {
	transcript, err := b.transcript(chatID, count)
	if err != nil {
		return err
	}
	if transcript == "" {
		return nil
	}
	updated, err := b.extractor.Extract(ctx, buildPrompt(current, transcript))
	if err != nil {
		return fmt.Errorf("extract: %w", err)
	}
	if updated = strings.TrimSpace(updated); updated == "" {
		return nil
	}
	log := ctxlog.Op(b.logger, "Builder.build").With().Int64("chat_id", chatID).Int("count", count).Logger()
	log.Info().Msg("interlocutor profile updated")
	return b.store.SaveProfile(chatID, updated, count)
}

// transcript renders the messages the extractor sees: the whole chat when short,
// otherwise the intro plus the recent window (with a gap marker) so both the
// identity-establishing start and the current context are covered.
func (b *Builder) transcript(chatID int64, count int) (string, error) {
	if count <= b.introN+b.recentN {
		all, err := b.store.LoadRecent(chatID, count)
		if err != nil {
			return "", fmt.Errorf("load history: %w", err)
		}
		return renderTranscript(all), nil
	}
	intro, err := b.store.LoadFirst(chatID, b.introN)
	if err != nil {
		return "", fmt.Errorf("load intro: %w", err)
	}
	recent, err := b.store.LoadRecent(chatID, b.recentN)
	if err != nil {
		return "", fmt.Errorf("load recent: %w", err)
	}
	return renderTranscript(intro) + "\n[...]\n" + renderTranscript(recent), nil
}

func renderTranscript(msgs []bot.Message) string {
	var sb strings.Builder
	for _, m := range msgs {
		who := "собеседник"
		if m.Role == roleAssistant {
			who = "Денис"
		}
		sb.WriteString(who + ": " + m.Content + "\n")
	}
	return strings.TrimSpace(sb.String())
}

const roleAssistant = "assistant"

func buildPrompt(current, transcript string) string {
	known := "пока ничего"
	if strings.TrimSpace(current) != "" {
		known = current
	}
	return "Ты ведёшь короткое досье на собеседника Дениса. В переписке ниже строки «собеседник:» — " +
		"это сообщения собеседника, строки «Денис:» — сам Денис (его описывать НЕ нужно).\n\n" +
		"Что уже известно о собеседнике:\n" + known + "\n\n" +
		"Переписка:\n" + transcript + "\n\n" +
		"Обнови досье короткими пунктами о собеседнике. Строгие правила:\n" +
		"1. Записывай ТОЛЬКО факты, прямо подтверждённые перепиской: имя и как обращаться, чем занимается, " +
		"интересы, конкретные детали жизни. Если названо имя — укажи его.\n" +
		"2. НЕ выдумывай и НЕ гадай. Чего не знаешь — просто пропусти пункт. Никаких заглушек вроде " +
		"«неизвестно», «не указано», «нет данных».\n" +
		"3. ЗАПРЕЩЕНО оценивать характер и стиль общения собеседника (нельзя писать «странный», " +
		"«нерегулярный», «непоследовательный», «путается» и т.п.).\n" +
		"4. Сохраняй ранее известные факты, если переписка им не противоречит.\n" +
		"5. Пиши кратко. Если фактов почти нет — верни одну-две строки, ничего не досочиняя."
}
