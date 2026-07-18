// Package vector is a minimal Chroma v2 REST client scoped to the single
// default_tenant/default_database. It owns the four verbs the callers need —
// get-or-create, add, query, delete — and carries no domain types.
package vector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// defaultTimeout bounds a single Chroma REST call. Add/query on a small
// collection are sub-second; this is headroom for a cold HNSW.
const defaultTimeout = 15 * time.Second

// Client talks to one Chroma server over its v2 REST API.
type Client struct {
	base string
	http *http.Client
}

// Option configures a Client at construction time.
type Option func(*Client)

// WithTimeout overrides the per-request timeout (values <= 0 are ignored).
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		if d > 0 {
			c.http.Timeout = d
		}
	}
}

// New builds a client for host (the Chroma base URL, e.g. http://192.168.1.2:8000).
func New(host string, opts ...Option) *Client {
	c := &Client{
		base: strings.TrimRight(host, "/"),
		http: &http.Client{Timeout: defaultTimeout},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// collectionsURL is the collections endpoint under the fixed default tenant/db.
func (c *Client) collectionsURL() string {
	return c.base + "/api/v2/tenants/default_tenant/databases/default_database/collections"
}

// Record is one row to persist: a stable id, its embedding, the source text
// (document) and opaque metadata used for filtering + provenance.
type Record struct {
	ID        string
	Embedding []float32
	Document  string
	Metadata  map[string]any
}

// Hit is one retrieved row: the stored document, its metadata and the cosine
// distance to the query (smaller is closer).
type Hit struct {
	Document string
	Metadata map[string]any
	Distance float32
}

// GetOrCreate resolves (creating if absent) the named collection with cosine
// space and returns its uuid. Cosine keeps one distance metric across every
// collection on this server.
func (c *Client) GetOrCreate(ctx context.Context, name string) (string, error) {
	body, err := json.Marshal(map[string]any{
		"name":          name,
		"get_or_create": true,
		"configuration": map[string]any{"hnsw": map[string]any{"space": "cosine"}},
	})
	if err != nil {
		return "", fmt.Errorf("marshal get_or_create %q: %w", name, err)
	}
	var out struct {
		ID string `json:"id"`
	}
	if err := c.do(ctx, http.MethodPost, c.collectionsURL(), body, &out); err != nil {
		return "", fmt.Errorf("chroma get_or_create %q: %w", name, err)
	}
	if out.ID == "" {
		return "", fmt.Errorf("chroma get_or_create %q: empty collection id", name)
	}
	return out.ID, nil
}

// Add upserts records into the collection. Chroma requires the four columns
// (ids, embeddings, documents, metadatas) as equal-length parallel arrays.
func (c *Client) Add(ctx context.Context, collID string, recs []Record) error {
	if len(recs) == 0 {
		return nil
	}
	ids := make([]string, len(recs))
	embeddings := make([][]float32, len(recs))
	documents := make([]string, len(recs))
	metadatas := make([]map[string]any, len(recs))
	for i, r := range recs {
		ids[i] = r.ID
		embeddings[i] = r.Embedding
		documents[i] = r.Document
		metadatas[i] = r.Metadata
	}
	body, err := json.Marshal(map[string]any{
		"ids":        ids,
		"embeddings": embeddings,
		"documents":  documents,
		"metadatas":  metadatas,
	})
	if err != nil {
		return fmt.Errorf("marshal add: %w", err)
	}
	if err := c.do(ctx, http.MethodPost, c.collectionsURL()+"/"+collID+"/add", body, nil); err != nil {
		return fmt.Errorf("chroma add: %w", err)
	}
	return nil
}

// Query returns up to k rows nearest to vec, restricted by the where filter.
// The Chroma response is per-query nested arrays; with one query vector the
// answer is the first (and only) row of each column.
func (c *Client) Query(ctx context.Context, collID string, vec []float32, k int, where map[string]any) ([]Hit, error) {
	reqBody := map[string]any{
		"query_embeddings": [][]float32{vec},
		"n_results":        k,
		"include":          []string{"documents", "metadatas", "distances"},
	}
	if len(where) > 0 {
		reqBody["where"] = where
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal query: %w", err)
	}
	var out struct {
		Ids       [][]string         `json:"ids"`
		Documents [][]string         `json:"documents"`
		Metadatas [][]map[string]any `json:"metadatas"`
		Distances [][]float32        `json:"distances"`
	}
	if err := c.do(ctx, http.MethodPost, c.collectionsURL()+"/"+collID+"/query", body, &out); err != nil {
		return nil, fmt.Errorf("chroma query: %w", err)
	}
	if len(out.Ids) == 0 {
		return nil, nil
	}
	hits := make([]Hit, 0, len(out.Ids[0]))
	for i := range out.Ids[0] {
		var hit Hit
		if len(out.Documents) > 0 && i < len(out.Documents[0]) {
			hit.Document = out.Documents[0][i]
		}
		if len(out.Metadatas) > 0 && i < len(out.Metadatas[0]) {
			hit.Metadata = out.Metadatas[0][i]
		}
		if len(out.Distances) > 0 && i < len(out.Distances[0]) {
			hit.Distance = out.Distances[0][i]
		}
		hits = append(hits, hit)
	}
	return hits, nil
}

// DeleteCollection removes the named collection under the fixed tenant/db.
func (c *Client) DeleteCollection(ctx context.Context, name string) error {
	if err := c.do(ctx, http.MethodDelete, c.collectionsURL()+"/"+name, nil, nil); err != nil {
		return fmt.Errorf("chroma delete_collection %q: %w", name, err)
	}
	return nil
}

// do performs one JSON request, decoding a 2xx body into out (when non-nil) and
// turning any non-2xx into an error carrying the status and body text.
func (c *Client) do(ctx context.Context, method, url string, body []byte, out any) error {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if out != nil {
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}
	return nil
}
