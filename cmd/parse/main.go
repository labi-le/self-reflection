package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"tg2llm/internal/app"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/infra/media"
	"tg2llm/internal/infra/textsink"
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

func run(args []string) error {
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

	if err := fs.Parse(args); err != nil {
		return err
	}

	logger := initLogger(*verbose)
	ctxLog := ctxlog.Op(logger, "run")

	positional := fs.Args()
	if len(positional) == 0 {
		fs.Usage()
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

	var dateRange dialogue.DateRange
	if *dateFrom != "" {
		from, err := dialogue.ParseBound(*dateFrom, false)
		if err != nil {
			ctxLog.Error().Err(err).Str("date-from", *dateFrom).Msg("invalid --date-from")
			return err
		}
		dateRange.From = &from
	}
	if *dateTo != "" {
		to, err := dialogue.ParseBound(*dateTo, true)
		if err != nil {
			ctxLog.Error().Err(err).Str("date-to", *dateTo).Msg("invalid --date-to")
			return err
		}
		dateRange.To = &to
	}

	source := tgexport.NewReader(inputPath)
	sink, err := textsink.NewWriter(outputPath)
	if err != nil {
		ctxLog.Error().Err(err).Msg("opening output file")
		return err
	}

	opts := app.ParseOptions{
		ShowDateHeaders: true,
		DateRange:       dateRange,
		Jobs:            *jobs,
	}
	ctxLog.Debug().Object("opts", opts).Msg("parse options")

	var decoder app.MediaDecoder

	if *decodeMedia {
		exportRoot := filepath.Dir(abspath(inputPath))
		cachePath := *mediaCache
		if cachePath == "" {
			cachePath = filepath.Join(exportRoot, ".tg2llm_media_cache.db")
		}

		if *whisperModel == "" {
			ctxLog.Warn().Msg("whisper model not set; voice messages stay as placeholders")
		}

		expandedModel := *whisperModel
		if expandedModel != "" && expandedModel[0] == '~' {
			home, err := os.UserHomeDir()
			if err == nil {
				expandedModel = filepath.Join(home, expandedModel[1:])
			}
		}

		decoder = media.BuildDefaultDecoder(
			exportRoot,
			cachePath,
			expandedModel,
			*whisperBin,
			*whisperLang,
			*visionModel,
			*visionHost,
			*visionPrompt,
			true, // enableVoice
			true, // enablePhoto
			logger,
		)
	}

	svc := app.NewParseService(source, sink, decoder, opts)
	if err := svc.Run(); err != nil {
		ctxLog.Error().Err(err).Msg("parse failed")
		return err
	}

	if decoder != nil {
		if err := decoder.Flush(); err != nil {
			ctxLog.Error().Err(err).Msg("flushing decoded media")
			return fmt.Errorf("flushing decoded media: %w", err)
		}
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
