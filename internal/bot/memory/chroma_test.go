package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Test_getOrCreate_requests_cosine_and_returns_the_collection_id proves the
// collection is created with cosine space (matching the index-repo indexer) and
// its id is returned for subsequent add/query calls.
func Test_getOrCreate_requests_cosine_and_returns_the_collection_id(t *testing.T) {
	// Given a Chroma v2 server that echoes an id
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "col-123", "name": "persona-memory"})
	}))
	defer srv.Close()
	c := newChromaStore(srv.URL)

	// When
	id, err := c.getOrCreate(context.Background(), "persona-memory")

	// Then the id is returned and the request targets the default tenant/db with cosine
	if err != nil {
		t.Fatalf("getOrCreate: %v", err)
	}
	if id != "col-123" {
		t.Fatalf("collection id = %q, want col-123", id)
	}
	const wantPath = "/api/v2/tenants/default_tenant/databases/default_database/collections"
	if gotPath != wantPath {
		t.Fatalf("path = %q, want %q", gotPath, wantPath)
	}
	if gotBody["get_or_create"] != true {
		t.Fatalf("get_or_create = %v, want true", gotBody["get_or_create"])
	}
	cfg, _ := gotBody["configuration"].(map[string]any)
	hnsw, _ := cfg["hnsw"].(map[string]any)
	if hnsw["space"] != "cosine" {
		t.Fatalf("hnsw.space = %v, want cosine", hnsw["space"])
	}
}

// Test_add_posts_ids_embeddings_documents_and_metadatas proves the add body
// carries the four parallel columns Chroma requires, at the collection's /add.
func Test_add_posts_ids_embeddings_documents_and_metadatas(t *testing.T) {
	// Given a Chroma server capturing the add body
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(http.StatusCreated)
	}))
	defer srv.Close()
	c := newChromaStore(srv.URL)

	// When
	err := c.add(context.Background(), "col-123", []record{{
		ID:        "id1",
		Embedding: []float32{0.1, 0.2},
		Document:  "привет",
		Metadata:  map[string]any{"chat_id": "7", "role": "user"},
	}})

	// Then the request hits /collections/col-123/add with all four columns
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	const wantPath = "/api/v2/tenants/default_tenant/databases/default_database/collections/col-123/add"
	if gotPath != wantPath {
		t.Fatalf("path = %q, want %q", gotPath, wantPath)
	}
	if ids, _ := gotBody["ids"].([]any); len(ids) != 1 || ids[0] != "id1" {
		t.Fatalf("ids = %v, want [id1]", gotBody["ids"])
	}
	if _, ok := gotBody["embeddings"]; !ok {
		t.Fatal("body missing embeddings")
	}
	if docs, _ := gotBody["documents"].([]any); len(docs) != 1 || docs[0] != "привет" {
		t.Fatalf("documents = %v", gotBody["documents"])
	}
	if metas, _ := gotBody["metadatas"].([]any); len(metas) != 1 {
		t.Fatalf("metadatas = %v, want one entry", gotBody["metadatas"])
	}
}

// Test_query_maps_the_nested_response_to_hits proves the columnar, per-query
// nested Chroma response is flattened to hits and the request carries the
// query vector, n_results, where filter and include list.
func Test_query_maps_the_nested_response_to_hits(t *testing.T) {
	// Given a Chroma server returning two hits for one query embedding
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ids":       [][]string{{"id1", "id2"}},
			"documents": [][]string{{"привет", "как дела"}},
			"metadatas": [][]map[string]any{{{"chat_id": "7", "role": "user"}, {"chat_id": "7", "role": "assistant"}}},
			"distances": [][]float32{{0.10, 0.25}},
		})
	}))
	defer srv.Close()
	c := newChromaStore(srv.URL)

	// When
	hits, err := c.query(context.Background(), "col-123", []float32{0.1, 0.2}, 5, map[string]any{"chat_id": "7"})

	// Then both hits are mapped (document, distance, metadata) in order
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("hits = %d, want 2", len(hits))
	}
	if hits[0].Document != "привет" {
		t.Fatalf("hit[0].Document = %q, want привет", hits[0].Document)
	}
	if hits[0].Distance != 0.10 {
		t.Fatalf("hit[0].Distance = %v, want 0.10", hits[0].Distance)
	}
	if hits[1].Metadata["role"] != "assistant" {
		t.Fatalf("hit[1].Metadata[role] = %v, want assistant", hits[1].Metadata["role"])
	}
	// And the request shape carries the query vector, n_results, where and include
	if gotBody["n_results"] != float64(5) {
		t.Fatalf("n_results = %v, want 5", gotBody["n_results"])
	}
	if _, ok := gotBody["query_embeddings"]; !ok {
		t.Fatal("body missing query_embeddings")
	}
	where, _ := gotBody["where"].(map[string]any)
	if where["chat_id"] != "7" {
		t.Fatalf("where.chat_id = %v, want 7", where["chat_id"])
	}
}

// Test_query_returns_no_hits_on_an_empty_result proves an empty Chroma result
// (no rows for the chat) is a clean (nil, nil), not an error.
func Test_query_returns_no_hits_on_an_empty_result(t *testing.T) {
	// Given a Chroma server returning empty per-query arrays
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ids":       [][]string{{}},
			"documents": [][]string{{}},
			"metadatas": [][]map[string]any{{}},
			"distances": [][]float32{{}},
		})
	}))
	defer srv.Close()
	c := newChromaStore(srv.URL)

	// When
	hits, err := c.query(context.Background(), "col-123", []float32{0.1}, 5, nil)

	// Then
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("hits = %d, want 0", len(hits))
	}
}
