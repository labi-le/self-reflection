package analyze_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
)

// chatStub is an OpenAI-compatible chat endpoint that returns a canned content
// string and records how many times it was called.
type chatStub struct {
	mu      sync.Mutex
	calls   int
	content string
	srv     *httptest.Server
}

func newChatStub(content string) *chatStub {
	c := &chatStub{content: content}
	c.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		c.mu.Lock()
		c.calls++
		c.mu.Unlock()
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": c.content}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	return c
}

func (c *chatStub) client() *llm.Client { return llm.New(c.srv.URL, "qa-model", "", time.Second) }
func (c *chatStub) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}
func (c *chatStub) close() { c.srv.Close() }

// stubEmbedder returns a fixed 1-D vector for any input.
type stubEmbedder struct{}

func (stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1}
	}
	return out, nil
}

// stubQueryIndex returns a fixed set of chunks from Query.
type stubQueryIndex struct {
	chunks []portrait.Chunk
	calls  int
}

func (x *stubQueryIndex) Add(context.Context, string, []portrait.Chunk, [][]float32) error {
	return nil
}
func (x *stubQueryIndex) Query(context.Context, string, []float32, int, map[string]any) ([]portrait.Chunk, error) {
	x.calls++
	return x.chunks, nil
}
func (x *stubQueryIndex) Delete(context.Context, string) error { return nil }

func chunk(idx int, memberIDs ...int) portrait.Chunk {
	m := map[int]bool{}
	for _, id := range memberIDs {
		m[id] = true
	}
	return portrait.Chunk{Index: idx, MemberIDs: m, Text: "excerpt"}
}

func TestAsk_CitesRetrievedID(t *testing.T) {
	chat := newChatStub("They enjoy hiking [1].")
	defer chat.close()
	idx := &stubQueryIndex{chunks: []portrait.Chunk{chunk(0, 1, 2)}}

	svc := analyze.NewQAService(chat.client(), stubEmbedder{}, idx, "coll")
	ans, cited, err := svc.Ask(context.Background(), "hobbies?", 5)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans == "" {
		t.Fatalf("empty answer")
	}
	if len(cited) != 1 || cited[0] != 1 {
		t.Fatalf("cited = %v, want [1]", cited)
	}
}

func TestAsk_StripsOutsideIDs(t *testing.T) {
	chat := newChatStub("Mixed cite [1] and [999].")
	defer chat.close()
	idx := &stubQueryIndex{chunks: []portrait.Chunk{chunk(0, 1, 2)}}

	svc := analyze.NewQAService(chat.client(), stubEmbedder{}, idx, "coll")
	_, cited, err := svc.Ask(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if len(cited) != 1 || cited[0] != 1 {
		t.Fatalf("cited = %v, want [1] (999 not in retrieved member ids)", cited)
	}
}

func TestAsk_NoValidCitationRefuses(t *testing.T) {
	chat := newChatStub("I think so [42].")
	defer chat.close()
	idx := &stubQueryIndex{chunks: []portrait.Chunk{chunk(0, 1, 2)}}

	svc := analyze.NewQAService(chat.client(), stubEmbedder{}, idx, "coll")
	ans, cited, err := svc.Ask(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if cited != nil {
		t.Fatalf("cited = %v, want nil", cited)
	}
	if ans != analyze.QARefusal {
		t.Fatalf("answer = %q, want refusal", ans)
	}
}

func TestAsk_EmptyRetrievalRefusesWithoutCallingChat(t *testing.T) {
	chat := newChatStub("should not be used")
	defer chat.close()
	idx := &stubQueryIndex{chunks: nil}

	svc := analyze.NewQAService(chat.client(), stubEmbedder{}, idx, "coll")
	ans, cited, err := svc.Ask(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans != analyze.QARefusal || cited != nil {
		t.Fatalf("expected refusal with nil cited, got ans=%q cited=%v", ans, cited)
	}
	if chat.count() != 0 {
		t.Fatalf("chat was called %d times on empty retrieval; must be 0", chat.count())
	}
}
