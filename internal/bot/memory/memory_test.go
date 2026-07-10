package memory

import (
	"context"
	"errors"
	"testing"
)

// fakeEmbedder records every embedded text and returns a fixed vector (or error).
type fakeEmbedder struct {
	vec   []float32
	err   error
	calls []string
}

func (f *fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	f.calls = append(f.calls, text)
	return f.vec, f.err
}

// fakeStore is an in-memory vecStore: it captures adds and the last query args
// and returns canned hits.
type fakeStore struct {
	id        string
	added     []record
	hits      []queryHit
	lastVec   []float32
	lastK     int
	lastWhere map[string]any
}

func (f *fakeStore) getOrCreate(_ context.Context, _ string) (string, error) {
	return f.id, nil
}

func (f *fakeStore) add(_ context.Context, _ string, recs []record) error {
	f.added = append(f.added, recs...)
	return nil
}

func (f *fakeStore) query(_ context.Context, _ string, vec []float32, k int, where map[string]any) ([]queryHit, error) {
	f.lastVec, f.lastK, f.lastWhere = vec, k, where
	return f.hits, nil
}

func newRAG(e embedder, s vecStore) *RAG {
	return &RAG{embed: e, store: s, collectionName: DefaultCollection}
}

// Test_Remember_embeds_and_adds_one_record_scoped_to_the_chat proves a turn is
// embedded and stored with chat_id/role metadata and its document text.
func Test_Remember_embeds_and_adds_one_record_scoped_to_the_chat(t *testing.T) {
	// Given a RAG over fakes
	fe := &fakeEmbedder{vec: []float32{0.1, 0.2, 0.3}}
	fs := &fakeStore{id: "col-1"}
	r := newRAG(fe, fs)

	// When one user turn is remembered
	if err := r.Remember(context.Background(), 7, "user", "привет как дела", 1000); err != nil {
		t.Fatalf("Remember: %v", err)
	}

	// Then the text was embedded and stored as one record with the right metadata
	if len(fe.calls) != 1 || fe.calls[0] != "привет как дела" {
		t.Fatalf("embed calls = %v, want one call with the turn text", fe.calls)
	}
	if len(fs.added) != 1 {
		t.Fatalf("stored records = %d, want 1", len(fs.added))
	}
	rec := fs.added[0]
	if rec.Document != "привет как дела" {
		t.Fatalf("record document = %q", rec.Document)
	}
	if rec.Metadata["chat_id"] != "7" {
		t.Fatalf("record chat_id = %v, want \"7\"", rec.Metadata["chat_id"])
	}
	if rec.Metadata["role"] != "user" {
		t.Fatalf("record role = %v, want user", rec.Metadata["role"])
	}
	if rec.ID == "" {
		t.Fatal("record id is empty; want a stable content hash")
	}
}

// Test_Remember_skips_blank_text proves whitespace-only turns are neither
// embedded nor stored.
func Test_Remember_skips_blank_text(t *testing.T) {
	// Given a RAG over fakes
	fe := &fakeEmbedder{vec: []float32{0.1}}
	fs := &fakeStore{id: "col-1"}
	r := newRAG(fe, fs)

	// When a blank turn is remembered
	if err := r.Remember(context.Background(), 7, "user", "   ", 1000); err != nil {
		t.Fatalf("Remember: %v", err)
	}

	// Then nothing was embedded or stored
	if len(fe.calls) != 0 {
		t.Fatalf("embed calls = %v, want none", fe.calls)
	}
	if len(fs.added) != 0 {
		t.Fatalf("stored records = %d, want 0", len(fs.added))
	}
}

// Test_Retrieve_embeds_the_query_and_maps_hits_scoped_to_the_chat proves the
// query is embedded, the where filter scopes to the chat, and hits map to
// Chunks with cosine similarity (1 - distance).
func Test_Retrieve_embeds_the_query_and_maps_hits_scoped_to_the_chat(t *testing.T) {
	// Given a store returning one same-chat hit
	fe := &fakeEmbedder{vec: []float32{0.4, 0.5}}
	fs := &fakeStore{
		id:   "col-1",
		hits: []queryHit{{Document: "вчера ели пиццу", Metadata: map[string]any{"chat_id": "7", "role": "assistant"}, Distance: 0.2}},
	}
	r := newRAG(fe, fs)

	// When retrieving for chat 7
	chunks, err := r.Retrieve(context.Background(), 7, "что мы ели?", 5)

	// Then the query was embedded, scoped to chat 7, and the hit mapped
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(fe.calls) != 1 || fe.calls[0] != "что мы ели?" {
		t.Fatalf("embed calls = %v", fe.calls)
	}
	if fs.lastK != 5 {
		t.Fatalf("query k = %d, want 5", fs.lastK)
	}
	if fs.lastWhere["chat_id"] != "7" {
		t.Fatalf("query where chat_id = %v, want \"7\"", fs.lastWhere["chat_id"])
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Text != "вчера ели пиццу" {
		t.Fatalf("chunk text = %q", chunks[0].Text)
	}
	if chunks[0].ChatID != 7 {
		t.Fatalf("chunk chat id = %d, want 7", chunks[0].ChatID)
	}
	if got := chunks[0].Score; got < 0.79 || got > 0.81 {
		t.Fatalf("chunk score = %v, want ~0.8 (1 - distance 0.2)", got)
	}
}

// Test_Retrieve_drops_a_hit_from_another_chat proves the privacy re-check filters
// out any row whose metadata chat_id differs from the requested chat.
func Test_Retrieve_drops_a_hit_from_another_chat(t *testing.T) {
	// Given a store that (wrongly) returns a hit tagged for chat 99
	fe := &fakeEmbedder{vec: []float32{0.4}}
	fs := &fakeStore{
		id:   "col-1",
		hits: []queryHit{{Document: "чужой секрет", Metadata: map[string]any{"chat_id": "99", "role": "user"}, Distance: 0.1}},
	}
	r := newRAG(fe, fs)

	// When retrieving for chat 7
	chunks, err := r.Retrieve(context.Background(), 7, "вопрос", 5)

	// Then the cross-chat hit is dropped
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(chunks) != 0 {
		t.Fatalf("chunks = %d, want 0 (cross-chat hit must be filtered)", len(chunks))
	}
}

// Test_Retrieve_returns_nothing_for_empty_query_or_zero_k proves the guards
// short-circuit before any embed/query call.
func Test_Retrieve_returns_nothing_for_empty_query_or_zero_k(t *testing.T) {
	// Given a RAG whose fakes would fail the test if hit
	fe := &fakeEmbedder{err: errors.New("must not embed")}
	fs := &fakeStore{id: "col-1"}
	r := newRAG(fe, fs)

	// When k <= 0
	if chunks, err := r.Retrieve(context.Background(), 7, "вопрос", 0); err != nil || chunks != nil {
		t.Fatalf("Retrieve(k=0) = (%v, %v), want (nil, nil)", chunks, err)
	}
	// And when the query is blank
	if chunks, err := r.Retrieve(context.Background(), 7, "   ", 5); err != nil || chunks != nil {
		t.Fatalf("Retrieve(blank) = (%v, %v), want (nil, nil)", chunks, err)
	}
	if len(fe.calls) != 0 {
		t.Fatalf("embed calls = %v, want none (guards short-circuit)", fe.calls)
	}
}
