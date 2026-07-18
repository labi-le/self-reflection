package memory

import (
	"context"
	"time"

	"tg2llm/internal/infra/vector"
)

// chromaStore wraps the neutral vector.Client, adapting the memory adapter's
// internal record/queryHit rows to the client's Record/Hit wire types.
type chromaStore struct {
	client *vector.Client
}

// ChromaOption configures a chromaStore at construction time.
type ChromaOption func(*[]vector.Option)

// WithChromaTimeout overrides the per-request timeout (values <= 0 are ignored).
func WithChromaTimeout(d time.Duration) ChromaOption {
	return func(opts *[]vector.Option) {
		*opts = append(*opts, vector.WithTimeout(d))
	}
}

// newChromaStore builds a client for host (the Chroma base URL, e.g.
// http://192.168.1.2:8000).
func newChromaStore(host string, opts ...ChromaOption) *chromaStore {
	var vecOpts []vector.Option
	for _, opt := range opts {
		opt(&vecOpts)
	}
	return &chromaStore{client: vector.New(host, vecOpts...)}
}

// record is one row to persist: a stable id, its embedding, the source text
// (document) and opaque metadata (chat_id/role/ts) used for filtering + provenance.
type record struct {
	ID        string
	Embedding []float32
	Document  string
	Metadata  map[string]any
}

// queryHit is one retrieved row: the stored document, its metadata and the
// cosine distance to the query (smaller is closer).
type queryHit struct {
	Document string
	Metadata map[string]any
	Distance float32
}

func (c *chromaStore) getOrCreate(ctx context.Context, name string) (string, error) {
	return c.client.GetOrCreate(ctx, name)
}

func (c *chromaStore) add(ctx context.Context, collectionID string, recs []record) error {
	vecRecs := make([]vector.Record, len(recs))
	for i, r := range recs {
		vecRecs[i] = vector.Record{
			ID:        r.ID,
			Embedding: r.Embedding,
			Document:  r.Document,
			Metadata:  r.Metadata,
		}
	}
	return c.client.Add(ctx, collectionID, vecRecs)
}

func (c *chromaStore) query(ctx context.Context, collectionID string, vec []float32, k int, where map[string]any) ([]queryHit, error) {
	hits, err := c.client.Query(ctx, collectionID, vec, k, where)
	if err != nil {
		return nil, err
	}
	out := make([]queryHit, len(hits))
	for i, h := range hits {
		out[i] = queryHit{Document: h.Document, Metadata: h.Metadata, Distance: h.Distance}
	}
	return out, nil
}
