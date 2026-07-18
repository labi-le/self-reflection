package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/analyzellm"
	"tg2llm/internal/infra/kvcache"
	"tg2llm/internal/infra/llm"
	"tg2llm/internal/infra/vecindex"
	"tg2llm/internal/infra/vector"
)

// runPortrait builds portrait.json (and an optional Markdown report) for one
// subject: read -> optional media decode -> deterministic stats + LLM map/reduce
// -> evidence-indexed portrait. Building the Q&A vector index is opt-in (both
// --embed-host and --chroma-host).
func runPortrait(args []string) error {
	fs := flag.NewFlagSet("portrait", flag.ExitOnError)
	target := fs.String("target", "", "subject from_id (bare N -> userN); default = inferred interlocutor")
	with := fs.String("with", "", "second party for --relationship (defaults to the other participant in a 1:1 chat)")
	relationship := fs.Bool("relationship", false, "produce 1:1 dynamics: Karpman roles, power balance, relationship arc")
	llmHost := fs.String("llm-host", "http://localhost:8080", "OpenAI-compatible chat base URL")
	llmModel := fs.String("llm-model", "default", "chat model name")
	llmKey := fs.String("llm-key", "", "bearer key for the chat/embed endpoints")
	embedHost := fs.String("embed-host", "", "embeddings base URL (with --chroma-host builds the Q&A index)")
	embedModel := fs.String("embed-model", "default", "embedding model name")
	chromaHost := fs.String("chroma-host", "", "Chroma vector server base URL (with --embed-host builds the Q&A index)")
	jobs := fs.Int("jobs", 3, "parallel map workers")
	cachePath := fs.String("cache", "", "fragment cache db (default <export-dir>/.tg2llm_analysis_cache.db)")
	sessionGap := fs.Duration("session-gap", 30*time.Minute, "silence gap that starts a new session")
	chunkChars := fs.Int("chunk-chars", 12000, "max characters per chunk fed to the model")
	repairRetries := fs.Int("repair-retries", 1, "structured-output repair attempts")
	dateFrom := fs.String("date-from", "", "skip messages before this date")
	dateTo := fs.String("date-to", "", "skip messages after this date")
	chatID := fs.String("chat", "", "restrict analysis to one chat_id (see: analyze chats)")
	windowBy := fs.String("window-by", "", "emit a per-window timeline: month | week")
	out := fs.String("out", "portrait.json", "portrait JSON output path")
	markdown := fs.String("markdown", "", "also write a Markdown report to this path")
	decodeMedia := fs.Bool("decode-media", false, "transcribe voice + describe photos first")
	whisperModel := fs.String("whisper-model", "", "whisper GGUF model path (voice, with --decode-media)")
	whisperBin := fs.String("whisper-bin", "whisper-cli", "whisper binary")
	whisperLang := fs.String("whisper-lang", "ru", "spoken language")
	visionModel := fs.String("vision-model", "qwen2.5vl:7b", "vision model name")
	visionHost := fs.String("vision-host", "http://localhost:8080", "vision server base URL")
	visionPrompt := fs.String("vision-prompt", "", "vision prompt")
	verbose := fs.Bool("verbose", false, "trace logging")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: analyze portrait [flags] <export.json>")
	}
	if *windowBy != "" && *windowBy != "month" && *windowBy != "week" {
		return fmt.Errorf("--window-by must be month or week")
	}
	input := fs.Arg(0)
	logger := initLogger(*verbose)
	warnRemote(logger, *llmHost, *embedHost, *chromaHost)
	if *decodeMedia {
		warnRemote(logger, *visionHost)
	}

	dr, err := dateRangeFrom(*dateFrom, *dateTo)
	if err != nil {
		return err
	}
	msgs, owner, err := readMessages(input, logger)
	if err != nil {
		return err
	}
	msgs = filterByDate(msgs, dr)
	msgs = filterByChat(msgs, *chatID)
	if len(msgs) == 0 {
		return fmt.Errorf("no messages after date filtering")
	}
	if *decodeMedia {
		dec := buildDecoder(input, *cachePath, *whisperModel, *whisperBin, *whisperLang, *visionModel, *visionHost, *visionPrompt, logger)
		msgs = decodeMessages(msgs, dec, logger)
	}

	tgt := dialogue.NormalizeOwnerID(*target)
	if tgt == "" {
		tgt = inferTarget(msgs, owner)
		if tgt == "" {
			return fmt.Errorf("no sender to analyze; pass --target (see: analyze participants)")
		}
		logger.Info().Str("target", tgt).Msg("inferred target")
	}

	cache := openCache(input, *cachePath, logger)
	defer func() {
		if cache != nil {
			_ = cache.Close()
		}
	}()

	chat := llm.New(*llmHost, *llmModel, *llmKey, 0)
	analyzer := analyzellm.New(chat, *llmModel, *repairRetries)

	var emb analyze.Embedder
	var idx analyze.Index
	if *embedHost != "" && *chromaHost != "" {
		emb = vecindex.NewEmbedder(llm.New(*embedHost, *embedModel, *llmKey, 0), *embedModel)
		idx = vecindex.NewIndex(vector.New(*chromaHost))
	}

	opts := analyze.AnalyzeOptions{
		Model:         *llmModel,
		EmbedModel:    *embedModel,
		Target:        tgt,
		With:          dialogue.NormalizeOwnerID(*with),
		Relationship:  *relationship,
		DateRange:     dr,
		SessionGap:    *sessionGap,
		ChunkChars:    *chunkChars,
		Jobs:          *jobs,
		RepairRetries: *repairRetries,
		WindowBy:      *windowBy,
	}
	svc := analyze.NewAnalyzeService(analyzer, emb, idx, cache, opts, logger)
	p, err := svc.Run(context.Background(), msgs)
	if err != nil {
		return err
	}

	if err := writeJSON(*out, p); err != nil {
		return err
	}
	fmt.Printf("Wrote portrait to %s (coverage %.0f%%, %d/%d chunks)\n", *out, p.Meta.Coverage*100, p.Meta.ChunksAnalyzed, p.Meta.ChunksTotal)

	if *markdown != "" {
		if err := os.WriteFile(*markdown, []byte(portrait.RenderMarkdown(p)), 0o644); err != nil {
			return fmt.Errorf("writing markdown: %w", err)
		}
		fmt.Printf("Wrote report to %s\n", *markdown)
	}
	return nil
}

func openCache(input, cachePath string, logger zerolog.Logger) *kvcache.Cache {
	if cachePath == "" {
		cachePath = filepath.Join(filepath.Dir(abspath(input)), ".tg2llm_analysis_cache.db")
	}
	c, err := kvcache.New(cachePath, "analysis_cache")
	if err != nil {
		logger.Warn().Err(err).Msg("opening analysis cache; running uncached")
		return nil
	}
	return c
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
