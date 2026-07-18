// Package analyze orchestrates the psychological-analysis pipeline: it computes
// deterministic statistics, segments the dialogue, maps each chunk through an
// LLM Analyzer, reduces the fragments in pure Go, assembles the portrait, and
// indexes the chunks for grounded Q&A. The heavy pure logic lives in
// internal/domain/portrait; this package owns the ports (adapters plug in) and
// the run/ask orchestration.
package analyze

import (
	"context"
	"time"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
)

// Analyzer maps one chunk to a fragment via an LLM. Implementations MUST restrict
// every evidence id to chunk.MemberIDs before returning (the member-id guard);
// a claim left with no evidence is emitted assessed:false / dropped, never a
// fabricated score.
type Analyzer interface {
	Analyze(ctx context.Context, chunk portrait.Chunk, target string) (portrait.Fragment, error)
}

// Embedder turns texts into vectors via an OpenAI-compatible /v1/embeddings
// endpoint. One vector per input, in order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Index is a per-analysis vector collection used for Q&A retrieval.
type Index interface {
	Add(ctx context.Context, collection string, chunks []portrait.Chunk, vectors [][]float32) error
	Query(ctx context.Context, collection string, vector []float32, k int, where map[string]any) ([]portrait.Chunk, error)
	Delete(ctx context.Context, collection string) error
}

// AnalyzeOptions configures a portrait run.
type AnalyzeOptions struct {
	Model         string
	EmbedModel    string
	Target        string // prefixed from_id (the subject)
	With          string // second party for dynamics ("" = infer)
	Relationship  bool
	DateRange     dialogue.DateRange
	SessionGap    time.Duration
	ChunkChars    int
	Jobs          int
	RepairRetries int
	WindowBy      string // "" | "month" | "week" — emit per-window Timeline
}

// DynamicsAnalyzer is an optional capability: an Analyzer that also infers 1:1
// relationship dynamics (Karpman roles, power balance) for a chunk between two
// participants a and b. It returns a Fragment carrying the guarded Dynamics and
// the count of evidence ids dropped by the member-id guard. Implementations MUST
// apply the same evidence member-id guard as Analyze. Callers type-assert for
// it, so adding it never changes the base Analyzer contract.
type DynamicsAnalyzer interface {
	AnalyzeDynamics(ctx context.Context, chunk portrait.Chunk, a, b string) (portrait.Fragment, error)
}
