package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type capturedRequest struct {
	path   string
	auth   string
	body   map[string]any
	hasKey func(string) bool
}

func newCapture() (*capturedRequest, http.HandlerFunc) {
	cap := &capturedRequest{}
	h := func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.auth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		cap.body = map[string]any{}
		_ = json.Unmarshal(raw, &cap.body)
		switch r.URL.Path {
		case "/v1/chat/completions":
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"  hello world  "}}]}`))
		case "/v1/embeddings":
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2]},{"index":1,"embedding":[0.3,0.4,0.5]}]}`))
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}
	return cap, h
}

func TestChatSendsBearerWhenKeySet(t *testing.T) {
	cap, h := newCapture()
	srv := httptest.NewServer(h)
	defer srv.Close()

	c := New(srv.URL, "m1", "secret", time.Second)
	if _, err := c.Chat(context.Background(), "hi", ChatOptions{Temperature: 0}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if cap.auth != "Bearer secret" {
		t.Fatalf("auth = %q, want %q", cap.auth, "Bearer secret")
	}
	if cap.path != "/v1/chat/completions" {
		t.Fatalf("path = %q", cap.path)
	}
	if cap.body["model"] != "m1" {
		t.Fatalf("model = %v, want m1", cap.body["model"])
	}
}

func TestChatOmitsBearerWhenKeyEmpty(t *testing.T) {
	cap, h := newCapture()
	srv := httptest.NewServer(h)
	defer srv.Close()

	c := New(srv.URL, "m1", "", time.Second)
	if _, err := c.Chat(context.Background(), "hi", ChatOptions{}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if cap.auth != "" {
		t.Fatalf("auth = %q, want empty", cap.auth)
	}
}

func TestChatBodyFieldsRespectOptions(t *testing.T) {
	cap, h := newCapture()
	srv := httptest.NewServer(h)
	defer srv.Close()

	c := New(srv.URL, "m1", "", time.Second)

	// Zero options: temperature always present; max_tokens and response_format absent.
	if _, err := c.Chat(context.Background(), "hi", ChatOptions{Temperature: 0.7}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, ok := cap.body["temperature"]; !ok {
		t.Fatalf("temperature missing")
	}
	if cap.body["temperature"] != 0.7 {
		t.Fatalf("temperature = %v, want 0.7", cap.body["temperature"])
	}
	if _, ok := cap.body["max_tokens"]; ok {
		t.Fatalf("max_tokens present when MaxTokens==0")
	}
	if _, ok := cap.body["response_format"]; ok {
		t.Fatalf("response_format present when ResponseFormat==nil")
	}
	if cap.body["stream"] != false {
		t.Fatalf("stream = %v, want false", cap.body["stream"])
	}

	// Set options: both present.
	rf := json.RawMessage(`{"type":"json_object"}`)
	if _, err := c.Chat(context.Background(), "hi", ChatOptions{Temperature: 0, MaxTokens: 128, ResponseFormat: rf}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if cap.body["max_tokens"] != float64(128) {
		t.Fatalf("max_tokens = %v, want 128", cap.body["max_tokens"])
	}
	rfOut, ok := cap.body["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format missing/wrong type: %v", cap.body["response_format"])
	}
	if rfOut["type"] != "json_object" {
		t.Fatalf("response_format.type = %v, want json_object", rfOut["type"])
	}
}

func TestChatParsesContent(t *testing.T) {
	_, h := newCapture()
	srv := httptest.NewServer(h)
	defer srv.Close()

	c := New(srv.URL, "m1", "", time.Second)
	got, err := c.Chat(context.Background(), "hi", ChatOptions{})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "hello world" {
		t.Fatalf("content = %q, want %q", got, "hello world")
	}
}

func TestChatNon200IsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := New(srv.URL, "m1", "", time.Second)
	if _, err := c.Chat(context.Background(), "hi", ChatOptions{}); err == nil {
		t.Fatalf("expected error on non-200")
	}
}

func TestChatEmptyChoicesIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "m1", "", time.Second)
	if _, err := c.Chat(context.Background(), "hi", ChatOptions{}); err == nil {
		t.Fatalf("expected error on empty choices")
	}
}

func TestEmbedParsesVectorsInOrder(t *testing.T) {
	cap, h := newCapture()
	srv := httptest.NewServer(h)
	defer srv.Close()

	c := New(srv.URL, "m1", "key", time.Second)
	got, err := c.Embed(context.Background(), "embed-model", []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if cap.path != "/v1/embeddings" {
		t.Fatalf("path = %q", cap.path)
	}
	if cap.auth != "Bearer key" {
		t.Fatalf("auth = %q", cap.auth)
	}
	if cap.body["model"] != "embed-model" {
		t.Fatalf("model = %v, want embed-model", cap.body["model"])
	}
	input, ok := cap.body["input"].([]any)
	if !ok || len(input) != 2 || input[0] != "a" || input[1] != "b" {
		t.Fatalf("input = %v", cap.body["input"])
	}
	want := [][]float32{{0.1, 0.2}, {0.3, 0.4, 0.5}}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("vec %d len = %d, want %d", i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("vec[%d][%d] = %v, want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestEmbedNon200IsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	defer srv.Close()

	c := New(srv.URL, "m1", "", time.Second)
	if _, err := c.Embed(context.Background(), "e", []string{"a"}); err == nil {
		t.Fatalf("expected error on non-200")
	}
}

func TestEmbedPlacesVectorsByReturnedIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Deliberately out of order: index 1 first, then index 0.
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "m", "", time.Second)
	got, err := c.Embed(context.Background(), "e", []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	want := [][]float32{{0.1, 0.2}, {0.3, 0.4}}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("vec %d len = %d, want %d", i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("vec[%d][%d] = %v, want %v (index pairing ignored)", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestEmbedCountMismatchIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]}]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "m", "", time.Second)
	if _, err := c.Embed(context.Background(), "e", []string{"a", "b"}); err == nil {
		t.Fatalf("expected error when response count != input count")
	}
}
