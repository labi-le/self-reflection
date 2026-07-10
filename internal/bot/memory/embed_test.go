package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Test_Embed_returns_the_vector_from_an_openai_embeddings_response proves the
// happy path: Embed POSTs {model,input} to /v1/embeddings and returns the vector
// carried by data[0].embedding.
func Test_Embed_returns_the_vector_from_an_openai_embeddings_response(t *testing.T) {
	// Given an OpenAI-compatible embeddings server returning a 4-dim vector
	want := []float32{0.10, -0.20, 0.30, 0.40}
	var gotPath, gotModel, gotInput string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		var req struct {
			Model string `json:"model"`
			Input string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotModel, gotInput = req.Model, req.Input
		_ = json.NewEncoder(w).Encode(embeddingsResponse{Data: []embeddingDatum{{Embedding: want}}})
	}))
	defer srv.Close()
	e := NewEmbedder(srv.URL, "bge-m3", WithEmbedTimeout(2*time.Second))

	// When
	got, err := e.Embed(context.Background(), "привет")

	// Then the vector, request path, model and input all match
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotPath != "/v1/embeddings" {
		t.Fatalf("request path = %q, want /v1/embeddings", gotPath)
	}
	if gotModel != "bge-m3" {
		t.Fatalf("request model = %q, want bge-m3", gotModel)
	}
	if gotInput != "привет" {
		t.Fatalf("request input = %q, want %q", gotInput, "привет")
	}
	if len(got) != len(want) {
		t.Fatalf("vector length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("vector[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// Test_Embed_errors_on_a_non_200_response proves a non-OK status is a returned
// error, never a silent empty vector.
func Test_Embed_errors_on_a_non_200_response(t *testing.T) {
	// Given a server that always fails
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "model not loaded", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	e := NewEmbedder(srv.URL, "bge-m3", WithEmbedTimeout(2*time.Second))

	// When
	_, err := e.Embed(context.Background(), "привет")

	// Then
	if err == nil {
		t.Fatal("Embed returned nil error on a 503 response, want an error")
	}
}

// Test_Embed_errors_on_empty_input proves empty text is rejected before any
// request is sent (the caller must never embed an empty string).
func Test_Embed_errors_on_empty_input(t *testing.T) {
	// Given a server that would fail the test if hit
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Fatal("Embed sent a request for empty input; it must short-circuit")
	}))
	defer srv.Close()
	e := NewEmbedder(srv.URL, "bge-m3")

	// When
	_, err := e.Embed(context.Background(), "")

	// Then
	if err == nil {
		t.Fatal("Embed returned nil error for empty input, want an error")
	}
}
