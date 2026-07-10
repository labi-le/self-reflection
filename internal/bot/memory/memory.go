package memory

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"tg2llm/internal/bot"
)

// DefaultCollection is the Chroma collection all persona memory lands in; rows
// are separated per conversation by the chat_id metadata, not by collection.
const DefaultCollection = "persona-memory"

// embedder is the text→vector seam (the concrete *Embedder, or a fake in tests).
type embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// vecStore is the vector-store seam (the concrete *chromaStore, or a fake).
type vecStore interface {
	getOrCreate(ctx context.Context, name string) (string, error)
	add(ctx context.Context, collectionID string, recs []record) error
	query(ctx context.Context, collectionID string, vec []float32, k int, where map[string]any) ([]queryHit, error)
}

var (
	_ embedder   = (*Embedder)(nil)
	_ vecStore   = (*chromaStore)(nil)
	_ bot.Memory = (*RAG)(nil)
)

// RAG is the ChromaDB-backed bot.Memory adapter: it embeds a turn with a bge-m3
// server and stores/queries it in one Chroma collection, scoped per chat via the
// chat_id metadata. The collection uuid is resolved once and cached.
type RAG struct {
	embed          embedder
	store          vecStore
	collectionName string

	mu           sync.Mutex
	collectionID string
}

// Config is the memory adapter's construction input. Collection defaults to
// DefaultCollection when empty.
type Config struct {
	EmbedHost  string
	EmbedModel string
	ChromaHost string
	Collection string
}

// New builds a RAG adapter that embeds via EmbedHost and stores in the Chroma at
// ChromaHost. No I/O happens here; the collection is resolved lazily on first use.
func New(cfg Config) *RAG {
	collection := cfg.Collection
	if collection == "" {
		collection = DefaultCollection
	}
	return &RAG{
		embed:          NewEmbedder(cfg.EmbedHost, cfg.EmbedModel),
		store:          newChromaStore(cfg.ChromaHost),
		collectionName: collection,
	}
}

// Remember embeds text and stores it as one memory row scoped to chatID. Empty
// text is a no-op. Embed/store errors are returned so the caller can log and
// continue — a failed write never blocks a reply.
func (r *RAG) Remember(ctx context.Context, chatID int64, role, text string, ts int64) error {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	collectionID, err := r.ensureCollection(ctx)
	if err != nil {
		return err
	}
	vec, err := r.embed.Embed(ctx, text)
	if err != nil {
		return fmt.Errorf("remember embed: %w", err)
	}
	rec := record{
		ID:        chunkID(chatID, ts, role, text),
		Embedding: vec,
		Document:  text,
		Metadata: map[string]any{
			"chat_id": strconv.FormatInt(chatID, 10),
			"role":    role,
			"ts":      ts,
		},
	}
	return r.store.add(ctx, collectionID, []record{rec})
}

// Retrieve returns up to k past fragments of chatID ranked by relevance to
// query. An empty query or k <= 0 yields no memory. Every hit is re-checked
// against chatID so one conversation's memory never surfaces in another.
func (r *RAG) Retrieve(ctx context.Context, chatID int64, query string, k int) ([]bot.Chunk, error) {
	query = strings.TrimSpace(query)
	if k <= 0 || query == "" {
		return nil, nil
	}
	collectionID, err := r.ensureCollection(ctx)
	if err != nil {
		return nil, err
	}
	vec, err := r.embed.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("retrieve embed: %w", err)
	}
	where := map[string]any{"chat_id": strconv.FormatInt(chatID, 10)}
	hits, err := r.store.query(ctx, collectionID, vec, k, where)
	if err != nil {
		return nil, err
	}
	chunks := make([]bot.Chunk, 0, len(hits))
	for _, hit := range hits {
		if got, ok := metaChatID(hit.Metadata); !ok || got != chatID {
			continue // privacy re-check: never surface another chat's memory
		}
		chunks = append(chunks, bot.Chunk{
			ChatID: chatID,
			Text:   hit.Document,
			Meta:   metaString(hit.Metadata, "role"),
			Score:  1 - float64(hit.Distance),
		})
	}
	return chunks, nil
}

// ensureCollection resolves (once) and caches the collection uuid.
func (r *RAG) ensureCollection(ctx context.Context) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.collectionID != "" {
		return r.collectionID, nil
	}
	id, err := r.store.getOrCreate(ctx, r.collectionName)
	if err != nil {
		return "", fmt.Errorf("resolve collection %q: %w", r.collectionName, err)
	}
	r.collectionID = id
	return id, nil
}

// chunkID is a stable id for a stored turn: identical (chat, ts, role, text)
// dedups to one row instead of piling duplicates. The four parts ARE the row's
// identity, so they are hashed together rather than grouped into a throwaway type.
func chunkID(chatID, ts int64, role, text string) string {
	sum := sha1.Sum([]byte(fmt.Sprintf("%d|%d|%s|%s", chatID, ts, role, text)))
	return hex.EncodeToString(sum[:])
}

// metaChatID parses the string chat_id back to int64 for the privacy re-check.
func metaChatID(meta map[string]any) (int64, bool) {
	raw, ok := meta["chat_id"].(string)
	if !ok {
		return 0, false
	}
	id, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, false
	}
	return id, true
}

// metaString reads a string metadata value; empty if absent or non-string.
func metaString(meta map[string]any, key string) string {
	if v, ok := meta[key].(string); ok {
		return v
	}
	return ""
}
