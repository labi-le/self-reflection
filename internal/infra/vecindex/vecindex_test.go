package vecindex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
	"tg2llm/internal/infra/vector"
)

// Test_Embed_delegates_to_the_pinned_model proves Embedder forwards texts to the
// llm client under the model chosen at construction and preserves vector order.
func Test_Embed_delegates_to_the_pinned_model(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 0, "embedding": []float32{0.1, 0.2}},
				{"index": 1, "embedding": []float32{0.3, 0.4}},
			},
		})
	}))
	defer srv.Close()
	e := NewEmbedder(llm.New(srv.URL, "chat-model", "", time.Second), "embed-model")

	vecs, err := e.Embed(context.Background(), []string{"a", "b"})

	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 || vecs[0][0] != 0.1 || vecs[1][1] != 0.4 {
		t.Fatalf("vecs = %v, want two vectors in order", vecs)
	}
	if body["model"] != "embed-model" {
		t.Fatalf("model = %v, want embed-model", body["model"])
	}
	input, _ := body["input"].([]any)
	if len(input) != 2 || input[0] != "a" || input[1] != "b" {
		t.Fatalf("input = %v, want [a b]", body["input"])
	}
}

// Test_Add_sends_member_ids_metadata_and_parallel_arrays proves Add gets/creates
// the collection then upserts one record per chunk carrying the document, the
// chunk index and a sorted member-id CSV as provenance.
func Test_Add_sends_member_ids_metadata_and_parallel_arrays(t *testing.T) {
	var addBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/add") {
			_ = json.NewDecoder(r.Body).Decode(&addBody)
			w.WriteHeader(http.StatusOK)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "col-123", "name": "persona"})
	}))
	defer srv.Close()
	ix := NewIndex(vector.New(srv.URL))

	chunks := []portrait.Chunk{
		{Index: 0, Text: "[10] hi\n[20] yo", MemberIDs: map[int]bool{20: true, 10: true}},
		{Index: 1, Text: "[10] bye", MemberIDs: map[int]bool{10: true}},
	}
	vectors := [][]float32{{0.1, 0.2}, {0.3, 0.4}}

	if err := ix.Add(context.Background(), "persona", chunks, vectors); err != nil {
		t.Fatalf("Add: %v", err)
	}

	ids, _ := addBody["ids"].([]any)
	if len(ids) != 2 || ids[0] != "persona:0" || ids[1] != "persona:1" {
		t.Fatalf("ids = %v, want [persona:0 persona:1]", addBody["ids"])
	}
	docs, _ := addBody["documents"].([]any)
	if len(docs) != 2 || docs[0] != "[10] hi\n[20] yo" {
		t.Fatalf("documents = %v", addBody["documents"])
	}
	metas, _ := addBody["metadatas"].([]any)
	if len(metas) != 2 {
		t.Fatalf("metadatas = %v, want 2 entries", addBody["metadatas"])
	}
	m0, _ := metas[0].(map[string]any)
	if m0["member_ids"] != "10,20" {
		t.Fatalf("member_ids = %v, want sorted CSV 10,20", m0["member_ids"])
	}
	if m0["chunk_index"] != float64(0) {
		t.Fatalf("chunk_index = %v, want 0", m0["chunk_index"])
	}
}

// Test_Add_errors_on_length_mismatch proves parallel-array lengths are checked
// before any request is issued.
func Test_Add_errors_on_length_mismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("Add issued a request despite a length mismatch")
	}))
	defer srv.Close()
	ix := NewIndex(vector.New(srv.URL))

	err := ix.Add(context.Background(), "persona", []portrait.Chunk{{Index: 0}}, nil)

	if err == nil {
		t.Fatal("Add: want length-mismatch error, got nil")
	}
}

// Test_Query_round_trips_a_hit_into_a_chunk proves a retrieved Hit is mapped back
// to a portrait.Chunk with Index and MemberIDs rebuilt from stored metadata.
func Test_Query_round_trips_a_hit_into_a_chunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/query") {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ids":       [][]string{{"persona:2"}},
				"documents": [][]string{{"[10] hi\n[20] yo"}},
				"metadatas": [][]map[string]any{{{"chunk_index": 2, "member_ids": "10,20"}}},
				"distances": [][]float32{{0.15}},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "col-123"})
	}))
	defer srv.Close()
	ix := NewIndex(vector.New(srv.URL))

	chunks, err := ix.Query(context.Background(), "persona", []float32{0.1}, 3, nil)

	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Index != 2 {
		t.Fatalf("Index = %d, want 2", chunks[0].Index)
	}
	if chunks[0].Text != "[10] hi\n[20] yo" {
		t.Fatalf("Text = %q", chunks[0].Text)
	}
	if len(chunks[0].MemberIDs) != 2 || !chunks[0].MemberIDs[10] || !chunks[0].MemberIDs[20] {
		t.Fatalf("MemberIDs = %v, want {10,20}", chunks[0].MemberIDs)
	}
}

// Test_Delete_drops_the_collection proves Delete issues a DELETE on the named
// collection.
func Test_Delete_drops_the_collection(t *testing.T) {
	var method, path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.Path
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	ix := NewIndex(vector.New(srv.URL))

	if err := ix.Delete(context.Background(), "persona"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if method != http.MethodDelete {
		t.Fatalf("method = %q, want DELETE", method)
	}
	if !strings.HasSuffix(path, "/collections/persona") {
		t.Fatalf("path = %q, want .../collections/persona", path)
	}
}
