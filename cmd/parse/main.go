package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"tg2llm/internal/app"
	"tg2llm/internal/bot"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/infra/media"
	"tg2llm/internal/infra/tgexport"
	"tg2llm/pkg/ctxlog"

	"github.com/rs/zerolog"
)

const outputTxt = "parsed_dialogue.txt"

func main() {
	if err := run(os.Args[1:]); err != nil {
		os.Exit(1)
	}
}

// parsedFlags holds the raw values needed by run() that are NOT part of
// ParseOptions (media-decode wiring, verbosity). ParseOptions itself carries
// the pipeline-level knobs.
type parsedFlags struct {
	decodeMedia  bool
	whisperModel string
	whisperBin   string
	whisperLang  string
	visionModel  string
	visionHost   string
	visionPrompt string
	mediaCache   string
	verbose      bool
}

// parseFlags is the CLI-parsing seam: it registers every flag on a fresh
// FlagSet, applies value normalisation (owner id prefixing, ~ expansion for
// path-shaped strings), and returns the populated ParseOptions plus the raw
// positional args. Splitting this out of run() lets tests exercise the flag
// wiring without touching disk or the pipeline.
func parseFlags(args []string) (app.ParseOptions, []string, error) {
	var raw parsedFlags
	opts, positional, err := parseFlagsInto(args, &raw)
	return opts, positional, err
}

func parseFlagsInto(args []string, raw *parsedFlags) (app.ParseOptions, []string, error) {
	fs := flag.NewFlagSet("parse", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: parse [flags] <input.json> [output.txt]\n\nFlags:\n")
		fs.PrintDefaults()
	}

	decodeMedia := fs.Bool("decode-media", false, "transcribe voice messages and describe photos using local models")
	whisperModel := fs.String("whisper-model", "", "path to a whisper.cpp GGUF model (required for voice transcription)")
	whisperBin := fs.String("whisper-bin", "whisper-cli", "whisper.cpp binary")
	whisperLang := fs.String("whisper-lang", "ru", "spoken language for transcription")
	visionModel := fs.String("vision-model", "qwen2.5vl:7b", "vision model name sent to the server")
	visionHost := fs.String("vision-host", "http://localhost:8080", "vision server base URL (llama.cpp/OpenAI-compatible /v1/chat/completions)")
	visionPrompt := fs.String("vision-prompt", "", "prompt sent to the vision model")
	mediaCache := fs.String("media-cache", "", "decoded-media cache file")
	jobs := fs.Int("jobs", 3, "parallel media-decode workers (1 = sequential)")
	dateFrom := fs.String("date-from", "", "skip messages before this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
	dateTo := fs.String("date-to", "", "skip messages after this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
	verbose := fs.Bool("verbose", false, "verbose (trace-level) logging with caller info")

	// Dataset / persona-bot flags (Task 4). Wired into ParseOptions here;
	// downstream sink/scope logic lands in later tasks.
	format := fs.String("format", "text", "output format: text|chatml")
	ownerUserID := fs.String("owner-user-id", "", "owner sender id (bare number N → user<N>; already-prefixed passes through)")
	chatScope := fs.String("chat-scope", "personal", "chat scope: personal|all")
	sessionGap := fs.Duration("session-gap", 30*time.Minute, "gap between messages that starts a new session")
	ctxTurns := fs.Int("ctx-turns", 6, "context turns to prepend when materialising a training sample")
	cutoffLen := fs.Int("cutoff-len", 1024, "per-example length cap used to bound context turns")
	maxPerChat := fs.Int("max-per-chat", 4000, "cap of samples per chat (0 = unlimited)")
	personaSystem := fs.String("persona-system", bot.DefaultPersonaSystem, "literal system prompt OR path to a file (leading ~ expanded); defaults to the locked persona prompt")
	burstDelim := fs.String("burst-delim", "\u2063", "delimiter joining consecutive same-sender messages inside a burst")

	if err := fs.Parse(args); err != nil {
		return app.ParseOptions{}, nil, err
	}

	raw.decodeMedia = *decodeMedia
	raw.whisperModel = *whisperModel
	raw.whisperBin = *whisperBin
	raw.whisperLang = *whisperLang
	raw.visionModel = *visionModel
	raw.visionHost = *visionHost
	raw.visionPrompt = *visionPrompt
	raw.mediaCache = *mediaCache
	raw.verbose = *verbose

	var dateRange dialogue.DateRange
	if *dateFrom != "" {
		from, err := dialogue.ParseBound(*dateFrom, false)
		if err != nil {
			return app.ParseOptions{}, nil, fmt.Errorf("invalid --date-from: %w", err)
		}
		dateRange.From = &from
	}
	if *dateTo != "" {
		to, err := dialogue.ParseBound(*dateTo, true)
		if err != nil {
			return app.ParseOptions{}, nil, fmt.Errorf("invalid --date-to: %w", err)
		}
		dateRange.To = &to
	}

	opts := app.ParseOptions{
		ShowDateHeaders: true,
		DateRange:       dateRange,
		Jobs:            *jobs,

		Format:        *format,
		OwnerFromID:   dialogue.NormalizeOwnerID(*ownerUserID),
		ChatScope:     *chatScope,
		SessionGap:    *sessionGap,
		CtxTurns:      *ctxTurns,
		CutoffLen:     *cutoffLen,
		MaxPerChat:    *maxPerChat,
		PersonaSystem: resolvePersonaSystem(*personaSystem),
		BurstDelim:    *burstDelim,
	}

	return opts, fs.Args(), nil
}


// resolvePersonaSystem accepts either a literal system prompt or a path to a
// file whose contents are the prompt. A leading "~" is expanded to $HOME
// exactly like --whisper-model. Anything that does not resolve to an existing
// file is treated as a literal and returned verbatim.
func resolvePersonaSystem(v string) string {
	if v == "" {
		return ""
	}
	candidate := v
	if candidate[0] == '~' {
		if home, err := os.UserHomeDir(); err == nil {
			candidate = filepath.Join(home, candidate[1:])
		}
	}
	info, err := os.Stat(candidate)
	if err != nil || info.IsDir() {
		return v
	}
	body, err := os.ReadFile(candidate)
	if err != nil {
		return v
	}
	return string(body)
}

func run(args []string) error {
	var raw parsedFlags
	opts, positional, err := parseFlagsInto(args, &raw)
	if err != nil {
		return err
	}

	logger := initLogger(raw.verbose)
	ctxLog := ctxlog.Op(logger, "run")

	if len(positional) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: parse [flags] <input.json> [output.txt]\n")
		ctxLog.Error().Msg("input file is required")
		return fmt.Errorf("input file is required")
	}
	inputPath := positional[0]
	outputPath := outputTxt
	if len(positional) >= 2 {
		outputPath = positional[1]
	}

	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		ctxLog.Error().Str("input", inputPath).Msg("input file not found")
		return fmt.Errorf("input file not found: %s", inputPath)
	}

	reader := tgexport.NewReader(inputPath, logger)

	ctxLog.Debug().Object("opts", opts).Msg("parse options")

	source, sink, sinkFile, err := buildPipeline(opts, reader, outputPath, logger)
	if err != nil {
		ctxLog.Error().Err(err).Msg("opening output file")
		return err
	}
	if sinkFile != nil {
		defer func() { _ = sinkFile.Close() }()
	}

	var decoder app.MediaDecoder

	if raw.decodeMedia {
		exportRoot := filepath.Dir(abspath(inputPath))
		cachePath := raw.mediaCache
		if cachePath == "" {
			cachePath = filepath.Join(exportRoot, ".tg2llm_media_cache.db")
		}

		if raw.whisperModel == "" {
			ctxLog.Warn().Msg("whisper model not set; voice messages stay as placeholders")
		}

		expandedModel := raw.whisperModel
		if expandedModel != "" && expandedModel[0] == '~' {
			home, err := os.UserHomeDir()
			if err == nil {
				expandedModel = filepath.Join(home, expandedModel[1:])
			}
		}

		decoder = media.BuildDefaultDecoder(
			exportRoot,
			cachePath,
			&media.WhisperConfig{Model: expandedModel, Bin: raw.whisperBin, Lang: raw.whisperLang},
			&media.VisionConfig{Model: raw.visionModel, Host: raw.visionHost, Prompt: raw.visionPrompt},
			logger,
		)
	}

	svc := app.NewParseService(source, sink, decoder, opts)
	if err := svc.Run(); err != nil {
		ctxLog.Error().Err(err).Msg("parse failed")
		return err
	}

	fmt.Printf("Wrote dialogue to %s\n", outputPath)
	return nil
}

func initLogger(verbose bool) zerolog.Logger {
	output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}

	if verbose {
		zerolog.CallerMarshalFunc = func(_ uintptr, file string, line int) string {
			return filepath.Base(file) + ":" + strconv.Itoa(line)
		}
		return zerolog.New(output).
			Level(zerolog.TraceLevel).
			With().
			Timestamp().
			Caller().
			Logger()
	}

	return zerolog.New(output).
		Level(zerolog.InfoLevel).
		With().
		Timestamp().
		Logger()
}

func abspath(p string) string {
	if p == "" {
		return p
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return abs
}
