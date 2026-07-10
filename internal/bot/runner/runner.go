// Package runner is the persona bot's live conversation loop: it consumes
// inbound updates from a ChatTransport, gates each sender against the allowlist,
// loads recent context, injects the interlocutor profile, asks the Brain for the
// persona's reply, persists both turns and delivers the reply — unless DRAFT
// mode or the send kill-switch suppresses delivery. After each reply it fires an
// out-of-band profile refresh. It wires the ports to the SQLite state and owns
// no I/O of its own, so tests drive it with a fake transport and a stub brain.
package runner

import (
	"context"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/internal/bot/store"
	"tg2llm/pkg/ctxlog"
)

const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"

	defaultContextTurns = 10
	defaultReplyTimeout = 120 * time.Second
)

// Builder refreshes a chat's interlocutor profile out-of-band; the runner fires
// it after each reply. The profile package implements it.
type Builder interface {
	Maybe(ctx context.Context, chatID int64)
}

// nopBuilder is the "no profile builder configured" default so the loop never
// nil-checks.
type nopBuilder struct{}

func (nopBuilder) Maybe(context.Context, int64) {}

// Deps carries the runner's collaborators and tuning. Transport and Brain are
// ports (fakes in tests); Store is the nil-safe SQLite state; Builder is the
// nil-safe profile refresher. A zero-valued ContextTurns or ReplyTimeout falls
// back to the package default; a nil Clock uses the wall clock.
type Deps struct {
	Transport bot.ChatTransport
	Brain     bot.Brain
	Store     *store.Store
	Builder   Builder
	Logger    zerolog.Logger
	Clock     func() int64

	ReplyTimeout time.Duration
	ContextTurns int

	Draft        bool
	SendDisabled bool
}

func (d Deps) contextTurns() int {
	if d.ContextTurns <= 0 {
		return defaultContextTurns
	}
	return d.ContextTurns
}

func (d Deps) builder() Builder {
	if d.Builder == nil {
		return nopBuilder{}
	}
	return d.Builder
}

func (d Deps) replyTimeout() time.Duration {
	if d.ReplyTimeout <= 0 {
		return defaultReplyTimeout
	}
	return d.ReplyTimeout
}

func (d Deps) now() int64 {
	if d.Clock == nil {
		return time.Now().Unix()
	}
	return d.Clock()
}

// Run consumes updates until ctx is cancelled or the transport's channel closes,
// handling each inbound update in turn. It returns the transport's start-up
// error, if any; a per-update failure is logged and swallowed so a single bad
// turn never stops the loop.
func Run(ctx context.Context, deps Deps) error {
	log := ctxlog.Op(deps.Logger, "runner.Run")
	updates, err := deps.Transport.Updates(ctx)
	if err != nil {
		return err
	}
	log.Info().
		Int("context_turns", deps.contextTurns()).
		Dur("reply_timeout", deps.replyTimeout()).
		Bool("profile", deps.Builder != nil).
		Bool("draft", deps.Draft).
		Bool("send_disabled", deps.SendDisabled).
		Msg("bot loop started")
	for {
		select {
		case <-ctx.Done():
			log.Info().Msg("context cancelled; bot loop stopping")
			return nil
		case up, ok := <-updates:
			if !ok {
				log.Info().Msg("transport channel closed; bot loop stopping")
				return nil
			}
			handleUpdate(ctx, deps, up)
		}
	}
}

// handleUpdate processes one inbound update: gate the sender, load recent
// context, inject the interlocutor profile, persist the inbound turn, ask the
// Brain (bounded by ReplyTimeout), persist the reply, fire an out-of-band
// profile refresh, then deliver — unless draft mode or the kill-switch
// suppresses delivery. Every failure is logged and swallowed so the loop keeps
// running.
func handleUpdate(ctx context.Context, deps Deps, up bot.Update) {
	log := ctxlog.Op(deps.Logger, "runner.handleUpdate").With().
		Int64("chat_id", up.ChatID).Int64("user_id", up.UserID).Logger()

	if !deps.Store.IsAllowed(up.UserID) {
		log.Debug().Msg("sender not allowed; dropping update")
		return
	}

	if isCommand(up.Text) && handleCommand(ctx, deps, up) {
		return
	}

	log.Info().Str("text", up.Text).Msg("dialogue in")

	convo, err := deps.Store.LoadRecent(up.ChatID, deps.contextTurns())
	if err != nil {
		log.Warn().Err(err).Msg("load recent context failed; using the inbound turn only")
	}
	if profile, _, pErr := deps.Store.GetProfile(up.ChatID); pErr != nil {
		log.Warn().Err(pErr).Msg("load profile failed; replying without it")
	} else if block := renderProfileBlock(profile); block != "" {
		convo = append([]bot.Message{{Role: roleSystem, Content: block}}, convo...)
		log.Debug().Msg("injected interlocutor profile")
	}
	convo = append(convo, bot.Message{Role: roleUser, Content: up.Text})
	if err := deps.Store.AppendMessage(up.ChatID, roleUser, up.Text, deps.now()); err != nil {
		log.Warn().Err(err).Msg("persist inbound turn failed")
	}

	replyCtx, cancel := context.WithTimeout(ctx, deps.replyTimeout())
	defer cancel()
	reply, err := deps.Brain.Reply(replyCtx, convo)
	if err != nil {
		log.Warn().Err(err).Msg("brain reply failed; skipping this turn")
		return
	}
	log.Info().Str("reply", reply).Msg("dialogue out")
	if err := deps.Store.AppendMessage(up.ChatID, roleAssistant, reply, deps.now()); err != nil {
		log.Warn().Err(err).Msg("persist assistant reply failed")
	}
	// Refresh the interlocutor profile out-of-band so it never delays the reply;
	// the builder self-throttles and uses its own timeout.
	go deps.builder().Maybe(context.Background(), up.ChatID)

	if deps.Draft || deps.SendDisabled {
		log.Info().
			Bool("draft", deps.Draft).
			Bool("send_disabled", deps.SendDisabled).
			Str("would_send", reply).
			Msg("send suppressed; reply generated but not delivered")
		return
	}
	if err := deps.Transport.Send(ctx, up.ChatID, reply); err != nil {
		log.Warn().Err(err).Msg("send reply failed")
	}
}

// renderProfileBlock frames the stored interlocutor profile as the persona's own
// knowledge of the person it is talking to (never as the persona itself). Empty
// when there is no profile.
func renderProfileBlock(profile string) string {
	profile = strings.TrimSpace(profile)
	if profile == "" {
		return ""
	}
	return "Что ты помнишь о собеседнике (это не ты, а тот, с кем ты общаешься):\n" + profile
}
