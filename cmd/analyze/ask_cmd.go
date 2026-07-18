package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
	"tg2llm/internal/infra/vecindex"
	"tg2llm/internal/infra/vector"
)

// runAsk answers questions grounded in the conversation: it segments and embeds
// the export into a per-analysis Chroma collection, then loops on QAService,
// which cites only retrieved message ids and refuses when unsupported.
func runAsk(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	target := fs.String("target", "", "subject from_id (bare N -> userN); default = inferred")
	llmHost := fs.String("llm-host", "http://localhost:8080", "OpenAI-compatible chat base URL")
	llmModel := fs.String("llm-model", "default", "chat model name")
	llmKey := fs.String("llm-key", "", "bearer key for the chat/embed endpoints")
	embedHost := fs.String("embed-host", "", "embeddings base URL (required)")
	embedModel := fs.String("embed-model", "default", "embedding model name")
	chromaHost := fs.String("chroma-host", "", "Chroma vector server base URL (required)")
	sessionGap := fs.Duration("session-gap", 30*time.Minute, "silence gap that starts a new session")
	chunkChars := fs.Int("chunk-chars", 12000, "max characters per chunk fed to the model")
	k := fs.Int("k", 6, "retrieved chunks per question")
	question := fs.String("question", "", "one-shot question (omit for interactive REPL)")
	keepIndex := fs.Bool("keep-index", false, "keep the vector collection after exit")
	dateFrom := fs.String("date-from", "", "skip messages before this date")
	dateTo := fs.String("date-to", "", "skip messages after this date")
	chatID := fs.String("chat", "", "restrict to one chat_id (see: analyze chats)")
	verbose := fs.Bool("verbose", false, "trace logging")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: analyze ask [flags] <export.json>")
	}
	if *embedHost == "" || *chromaHost == "" {
		return fmt.Errorf("ask requires --embed-host and --chroma-host for grounded Q&A (keyword fallback is not implemented in M1)")
	}
	logger := initLogger(*verbose)
	warnRemote(logger, *llmHost, *embedHost, *chromaHost)

	dr, err := dateRangeFrom(*dateFrom, *dateTo)
	if err != nil {
		return err
	}
	msgs, owner, err := readMessages(fs.Arg(0), logger)
	if err != nil {
		return err
	}
	msgs = filterByDate(msgs, dr)
	msgs = filterByChat(msgs, *chatID)
	if len(msgs) == 0 {
		return fmt.Errorf("no messages after date filtering")
	}
	tgt := dialogue.NormalizeOwnerID(*target)
	if tgt == "" {
		tgt = inferTarget(msgs, owner)
	}

	chunks := portrait.Segment(msgs, *sessionGap, *chunkChars)
	if len(chunks) == 0 {
		return fmt.Errorf("nothing to index")
	}

	ctx := context.Background()
	chat := llm.New(*llmHost, *llmModel, *llmKey, 0)
	emb := vecindex.NewEmbedder(llm.New(*embedHost, *embedModel, *llmKey, 0), *embedModel)
	idx := vecindex.NewIndex(vector.New(*chromaHost))
	collection := analyze.CollectionName(tgt, *llmModel, *embedModel)

	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embedding chunks: %w", err)
	}
	if err := idx.Add(ctx, collection, chunks, vecs); err != nil {
		return fmt.Errorf("indexing chunks: %w", err)
	}
	if !*keepIndex {
		defer func() {
			if err := idx.Delete(ctx, collection); err != nil {
				logger.Warn().Err(err).Msg("purging Q&A index collection")
			}
		}()
	}

	qa := analyze.NewQAService(chat, emb, idx, collection)
	askOne := func(q string) {
		ans, cited, err := qa.Ask(ctx, q, *k)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			return
		}
		fmt.Println(ans)
		if len(cited) > 0 {
			fmt.Printf("  [cited messages: %v]\n", cited)
		}
	}

	if *question != "" {
		askOne(*question)
		return nil
	}

	fmt.Fprintln(os.Stderr, "Ask questions about the conversation (Ctrl-D or 'exit' to quit):")
	sc := bufio.NewScanner(os.Stdin)
	fmt.Print("> ")
	for sc.Scan() {
		q := strings.TrimSpace(sc.Text())
		switch q {
		case "":
		case "exit", "quit":
			return nil
		default:
			askOne(q)
		}
		fmt.Print("> ")
	}
	return sc.Err()
}
