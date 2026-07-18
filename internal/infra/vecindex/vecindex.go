// Package vecindex adapts the Wave-1 llm and vector clients to the analyze
// ports: an Embedder backed by an OpenAI-compatible /v1/embeddings endpoint and
// an Index backed by one Chroma collection per analysis. The Index persists the
// sorted member-id set of each chunk as metadata so Q&A retrieval can rebuild
// Chunk.MemberIDs and validate that every cited id came from a retrieved chunk.
package vecindex

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
	"tg2llm/internal/infra/vector"
)

// Embedder turns texts into vectors via the shared llm client, pinning the
// embedding model chosen at construction.
type Embedder struct {
	c     *llm.Client
	model string
}

// NewEmbedder binds an llm client to the embedding model it should use.
func NewEmbedder(c *llm.Client, model string) *Embedder {
	return &Embedder{c: c, model: model}
}

// Embed delegates to the llm client, preserving input order.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return e.c.Embed(ctx, e.model, texts)
}

// Index is a per-analysis Chroma collection used for grounded Q&A retrieval.
type Index struct {
	c *vector.Client
}

// NewIndex wraps a Chroma client as an analyze.Index.
func NewIndex(c *vector.Client) *Index {
	return &Index{c: c}
}

// Add persists one record per chunk, carrying the chunk text as the document and
// the chunk index plus a sorted member-id CSV as metadata. Vectors are the
// parallel embeddings; a length mismatch is a programming error and is rejected
// before any request is issued.
func (ix *Index) Add(ctx context.Context, collection string, chunks []portrait.Chunk, vectors [][]float32) error {
	if len(chunks) != len(vectors) {
		return fmt.Errorf("vecindex add %q: %d chunks vs %d vectors", collection, len(chunks), len(vectors))
	}
	if len(chunks) == 0 {
		return nil
	}
	collID, err := ix.c.GetOrCreate(ctx, collection)
	if err != nil {
		return err
	}
	recs := make([]vector.Record, len(chunks))
	for i, ch := range chunks {
		recs[i] = vector.Record{
			ID:        collection + ":" + strconv.Itoa(ch.Index),
			Embedding: vectors[i],
			Document:  ch.Text,
			Metadata: map[string]any{
				"chunk_index": ch.Index,
				"member_ids":  memberCSV(ch.MemberIDs),
			},
		}
	}
	return ix.c.Add(ctx, collID, recs)
}

// Query returns the k nearest chunks, rebuilding each Chunk's Index and
// MemberIDs from the stored metadata so callers can validate cited ids.
func (ix *Index) Query(ctx context.Context, collection string, vec []float32, k int, where map[string]any) ([]portrait.Chunk, error) {
	collID, err := ix.c.GetOrCreate(ctx, collection)
	if err != nil {
		return nil, err
	}
	hits, err := ix.c.Query(ctx, collID, vec, k, where)
	if err != nil {
		return nil, err
	}
	chunks := make([]portrait.Chunk, 0, len(hits))
	for _, h := range hits {
		chunks = append(chunks, portrait.Chunk{
			Index:     metaInt(h.Metadata, "chunk_index"),
			Text:      h.Document,
			MemberIDs: parseMemberCSV(h.Metadata["member_ids"]),
		})
	}
	return chunks, nil
}

// Delete drops the collection.
func (ix *Index) Delete(ctx context.Context, collection string) error {
	return ix.c.DeleteCollection(ctx, collection)
}

// memberCSV renders the member-id set as a sorted, comma-separated string so the
// stored metadata is deterministic and Query can rebuild the exact set.
func memberCSV(ids map[int]bool) string {
	sorted := make([]int, 0, len(ids))
	for id := range ids {
		sorted = append(sorted, id)
	}
	sort.Ints(sorted)
	parts := make([]string, len(sorted))
	for i, id := range sorted {
		parts[i] = strconv.Itoa(id)
	}
	return strings.Join(parts, ",")
}

// parseMemberCSV inverts memberCSV; a missing or empty value yields a nil set.
func parseMemberCSV(v any) map[int]bool {
	s, _ := v.(string)
	if s == "" {
		return nil
	}
	ids := make(map[int]bool)
	for _, part := range strings.Split(s, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(part)); err == nil {
			ids[n] = true
		}
	}
	return ids
}

// metaInt reads an int from metadata that survived a JSON round-trip (where a
// number decodes as float64).
func metaInt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case string:
		n, _ := strconv.Atoi(v)
		return n
	}
	return 0
}

var (
	_ analyze.Embedder = (*Embedder)(nil)
	_ analyze.Index    = (*Index)(nil)
)
