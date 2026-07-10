package memory

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

// defaultChromaTimeout bounds a single Chroma REST call. Add/query on a small
// per-chat collection are sub-second; this is headroom for a cold HNSW.
const defaultChromaTimeout = 15 * time.Second

// chromaStore is a minimal Chroma v2 REST client scoped to the single
// default_tenant/default_database, mirroring the wire contract the index-repo
// indexer already uses against this exact server. It owns only the three verbs
// the memory adapter needs: get-or-create, add, query.
type chromaStore struct {
	base string
	http *http.Client
}

// ChromaOption configures a chromaStore at construction time.
type ChromaOption func(*chromaStore)

// WithChromaTimeout overrides the per-request timeout (values <= 0 are ignored).
func WithChromaTimeout(d time.Duration) ChromaOption {
	return func(c *chromaStore) {
		if d > 0 {
			c.http.Timeout = d
		}
	}
}

// newChromaStore builds a client for host (the Chroma base URL, e.g.
// http://192.168.1.2:8000).
func newChromaStore(host string, opts ...ChromaOption) *chromaStore {
	c := &chromaStore{
		base: strings.TrimRight(host, "/"),
		http: &http.Client{Timeout: defaultChromaTimeout},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// collectionsURL is the collections endpoint under the fixed default tenant/db.
func (c *chromaStore) collectionsURL() string {
	return c.base + "/api/v2/tenants/default_tenant/databases/default_database/collections"
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
	ID       string
	Document string
	Metadata map[string]any
	Distance float32
}

// getOrCreate resolves (creating if absent) the named collection with cosine
// space and returns its uuid. Cosine matches the index-repo indexer so this
// server keeps one distance metric across every collection.
func (c *chromaStore) getOrCreate(ctx context.Context, name string) (string, error) {
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
	if err := c.post(ctx, c.collectionsURL(), body, &out); err != nil {
		return "", fmt.Errorf("chroma get_or_create %q: %w", name, err)
	}
	if out.ID == "" {
		return "", fmt.Errorf("chroma get_or_create %q: empty collection id", name)
	}
	return out.ID, nil
}

// add upserts records into the collection. Chroma requires the four columns
// (ids, embeddings, documents, metadatas) as equal-length parallel arrays.
func (c *chromaStore) add(ctx context.Context, collectionID string, recs []record) error {
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
	if err := c.post(ctx, c.collectionsURL()+"/"+collectionID+"/add", body, nil); err != nil {
		return fmt.Errorf("chroma add: %w", err)
	}
	return nil
}

// query returns up to k rows nearest to vec, restricted by the where filter
// (typically {"chat_id": "<id>"} for the per-conversation scope). The Chroma
// response is per-query nested arrays; with one query vector the answer is the
// first (and only) row of each column.
func (c *chromaStore) query(ctx context.Context, collectionID string, vec []float32, k int, where map[string]any) ([]queryHit, error) {
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
	if err := c.post(ctx, c.collectionsURL()+"/"+collectionID+"/query", body, &out); err != nil {
		return nil, fmt.Errorf("chroma query: %w", err)
	}
	if len(out.Ids) == 0 {
		return nil, nil
	}
	hits := make([]queryHit, 0, len(out.Ids[0]))
	for i := range out.Ids[0] {
		hit := queryHit{ID: out.Ids[0][i]}
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

// post performs one JSON POST, decoding a 2xx body into out (when non-nil) and
// turning any non-2xx into an error carrying the status and body text.
func (c *chromaStore) post(ctx context.Context, url string, body []byte, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

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
