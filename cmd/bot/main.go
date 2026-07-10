package main

import (
	"context"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/internal/bot/botmedia"
	"tg2llm/internal/bot/brain/llamacpp"
	"tg2llm/internal/bot/profile"
	"tg2llm/internal/bot/runner"
	"tg2llm/internal/bot/store"
	"tg2llm/internal/bot/transport/botapi"
	"tg2llm/pkg/ctxlog"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		os.Exit(1)
	}
}

// run resolves config, opens the state store, seeds the allowlist, wires the Bot
// API transport and the llama.cpp persona Brain to the store, then drives the
// live conversation loop until a shutdown signal. Without a token it refuses to
// poll (there is nothing to authenticate with) and simply waits for the signal.
func run(args []string) error {
	cfg, err := parseConfig(args, os.Getenv)
	if err != nil {
		return err
	}

	logger := initLogger(cfg.verbose)
	ctxLog := ctxlog.Op(logger, "bot.run")

	var opts []store.Option
	if cfg.openWhenEmpty {
		opts = append(opts, store.WithOpenWhenEmpty())
	}
	st := store.New(cfg.dbPath, cfg.ownerID, logger, opts...)
	defer func() { _ = st.Close() }()

	for _, id := range cfg.allowSeed {
		if err := st.AddAllowed(id, "seed"); err != nil {
			ctxLog.Warn().Err(err).Int64("user_id", id).Msg("seeding allowlist id failed")
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if cfg.token == "" {
		ctxLog.Warn().Msgf("$%s is not set; refusing to poll — set it and restart", envBotToken)
		<-ctx.Done()
		return nil
	}

	// store.IsAllowed gates inbound updates at the transport edge (and again in
	// the loop); the token stays inside the transport and is never logged.
	// Media understanding: a decoder is built when a whisper model and/or a
	// vision server is configured; without either the bot stays text-only.
	var decoder bot.MediaDecoder
	if cfg.whisperModel != "" || cfg.visionHost != "" {
		decoder = botmedia.New(botmedia.Config{
			WhisperModel: cfg.whisperModel,
			WhisperBin:   cfg.whisperBin,
			WhisperLang:  cfg.whisperLang,
			VisionHost:   cfg.visionHost,
			VisionModel:  cfg.visionModel,
			VisionPrompt: cfg.visionPrompt,
			Logger:       logger,
		})
	}
	transport := botapi.New(cfg.token, st.IsAllowed, logger,
		botapi.WithBaseURL(cfg.baseURL),
		botapi.WithWordSplit(cfg.splitWordProb, cfg.splitMaxMsgs),
		botapi.WithMediaDecoder(decoder))
	// An empty model name lets the Brain use its default; the persona server
	// answers with whatever model it loaded, so the name is immaterial here.
	brain := llamacpp.New(cfg.llamaHost, cfg.personaSystem, "",
		llamacpp.WithLogger(logger), llamacpp.WithTimeout(cfg.replyTimeout),
		llamacpp.WithTemperature(cfg.temperature),
		llamacpp.WithRepeatPenalty(cfg.repeatPenalty),
		llamacpp.WithPresencePenalty(cfg.presencePenalty),
		llamacpp.WithFrequencyPenalty(cfg.frequencyPenalty),
		llamacpp.WithRepeatLastN(cfg.repeatLastN))

	// The interlocutor profile is refreshed out-of-band by a general instruct
	// model (profile-host); without it the bot relies on its recent window alone.
	var builder runner.Builder
	if cfg.profileHost != "" {
		builder = profile.NewBuilder(st,
			profile.NewExtractor(cfg.profileHost, cfg.profileModel, 0),
			profile.Config{Every: cfg.profileEvery, Logger: logger})
	}

	// Only token presence is logged, never the token itself.
	ctxLog.Info().
		Bool("token_set", true).
		Str("db", cfg.dbPath).
		Str("llama_host", cfg.llamaHost).
		Int64("owner_id", cfg.ownerID).
		Int("allow_seeded", len(cfg.allowSeed)).
		Int("context_turns", cfg.contextTurns).
		Bool("profile", cfg.profileHost != "").
		Str("profile_host", cfg.profileHost).
		Int("profile_every", cfg.profileEvery).
		Bool("media", cfg.whisperModel != "" || cfg.visionHost != "").
		Str("whisper_model", cfg.whisperModel).
		Str("vision_host", cfg.visionHost).
		Dur("reply_timeout", cfg.replyTimeout).
		Float64("temperature", cfg.temperature).
		Float64("repeat_penalty", cfg.repeatPenalty).
		Float64("presence_penalty", cfg.presencePenalty).
		Float64("frequency_penalty", cfg.frequencyPenalty).
		Int("repeat_last_n", cfg.repeatLastN).
		Bool("draft", cfg.draft).
		Bool("send_disabled", cfg.sendDisabled).
		Bool("open_when_empty", cfg.openWhenEmpty).
		Float64("split_word_prob", cfg.splitWordProb).
		Int("split_max_msgs", cfg.splitMaxMsgs).
		Msg("bot starting")

	return runner.Run(ctx, runner.Deps{
		Transport:    transport,
		Brain:        brain,
		Store:        st,
		Builder:      builder,
		Logger:       logger,
		ContextTurns: cfg.contextTurns,
		ReplyTimeout: cfg.replyTimeout,
		Draft:        cfg.draft,
		SendDisabled: cfg.sendDisabled,
	})
}

func initLogger(verbose bool) zerolog.Logger {
	output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
	if verbose {
		zerolog.CallerMarshalFunc = func(_ uintptr, file string, line int) string {
			return filepath.Base(file) + ":" + strconv.Itoa(line)
		}
		return zerolog.New(output).Level(zerolog.TraceLevel).With().Timestamp().Caller().Logger()
	}
	return zerolog.New(output).Level(zerolog.InfoLevel).With().Timestamp().Logger()
}
