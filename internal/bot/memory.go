package bot

import "context"

// Memory is the persona bot's long-term-memory seam (retrieval-augmented
// generation). It has two sides, both scoped to a single chat:
//
//   - Remember persists one conversation turn so it can be recalled later.
//   - Retrieve returns the k most relevant past fragments of the SAME chat for a
//     query (typically the latest inbound turn), to inject as a scoped "memory"
//     block ahead of the Brain call.
//
// This package declares the port ONLY — no embedding or vector math lives here.
// The concrete adapter (a bge-m3 embeddings server + ChromaDB) lives in
// internal/bot/memory and is wired in cmd/bot. Per-chat scoping is the privacy
// contract every implementation must uphold: memory from one conversation is
// never surfaced in another.
type Memory interface {
	// Retrieve returns up to k Chunks for chatID, ranked most-relevant-first
	// for query. Results MUST be scoped to chatID. Returning (nil, nil) means
	// "no memory" and is always a valid answer.
	Retrieve(ctx context.Context, chatID int64, query string, k int) ([]Chunk, error)

	// Remember persists one turn (role + text at ts) for chatID. An empty text
	// is a no-op. A returned error lets the caller log and continue — a failed
	// memory write never blocks a reply.
	Remember(ctx context.Context, chatID int64, role, text string, ts int64) error
}

// Chunk is one retrieved memory fragment. ChatID is carried so callers can
// re-assert the privacy scope after retrieval; Score is the cosine similarity
// to the query (higher is closer); Meta is opaque provenance (e.g. the source
// role) the injector may render but the Brain need not parse. ID is left 0 by
// the ChromaDB adapter (its document ids are content hashes, not integers) and
// is not consumed downstream.
type Chunk struct {
	ID     int64
	ChatID int64
	Text   string
	Meta   string
	Score  float64
}

// nopMemory is the trivial Memory implementation: it remembers nothing and
// retrieves nothing. It is the "RAG disabled" default so the runner can hold a
// non-nil Memory without special-casing nil.
type nopMemory struct{}

// Compile-time proof that nopMemory satisfies Memory.
var _ Memory = nopMemory{}

// Retrieve always returns no memory.
func (nopMemory) Retrieve(context.Context, int64, string, int) ([]Chunk, error) {
	return nil, nil
}

// Remember discards the turn.
func (nopMemory) Remember(context.Context, int64, string, string, int64) error {
	return nil
}

// NopMemory returns the no-op Memory used when RAG is disabled, so callers wire
// a real, non-nil port either way.
func NopMemory() Memory { return nopMemory{} }
