package media

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenAIDescribeSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
		}
		var req oaiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Stream {
			t.Error("request set stream=true, want false")
		}
		if req.Temperature != 0 {
			t.Errorf("temperature = %v, want 0", req.Temperature)
		}
		if len(req.Messages) != 1 {
			t.Errorf("messages = %d, want 1", len(req.Messages))
		} else if parts := req.Messages[0].Content; len(parts) != 2 {
			t.Errorf("content parts = %d, want 2", len(parts))
		} else {
			if parts[0].Type != "text" || parts[0].Text != "describe" {
				t.Errorf("part0 = %+v, want text/describe", parts[0])
			}
			if parts[1].Type != "image_url" || parts[1].ImageURL == nil ||
				!strings.HasPrefix(parts[1].ImageURL.URL, "data:image/jpeg;base64,") {
				t.Errorf("part1 = %+v, want image_url data URI", parts[1])
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"a cat\non a sofa"}}]}`))
	}))
	defer srv.Close()

	img := filepath.Join(t.TempDir(), "p.jpg")
	if err := os.WriteFile(img, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write image: %v", err)
	}

	d := NewOpenAIDescriber("qwen", srv.URL, "describe", testLogger)
	if got := d.Describe(img); got != "a cat on a sofa" {
		t.Fatalf("Describe() = %q, want %q", got, "a cat on a sofa")
	}
}

func TestOpenAIDescribeNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	img := filepath.Join(t.TempDir(), "p.jpg")
	if err := os.WriteFile(img, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write image: %v", err)
	}
	d := NewOpenAIDescriber("m", srv.URL, "p", testLogger)
	if got := d.Describe(img); got != "" {
		t.Fatalf("non-200 Describe = %q, want empty", got)
	}
}

func TestOpenAIDescribeMissingFile(t *testing.T) {
	d := NewOpenAIDescriber("m", "http://127.0.0.1:1", "p", testLogger)
	if got := d.Describe(filepath.Join(t.TempDir(), "nope.jpg")); got != "" {
		t.Fatalf("missing-file Describe = %q, want empty", got)
	}
}

func TestOpenAIDescribeMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{not valid json`))
	}))
	defer srv.Close()

	img := filepath.Join(t.TempDir(), "p.jpg")
	if err := os.WriteFile(img, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write image: %v", err)
	}
	d := NewOpenAIDescriber("qwen", srv.URL, "describe", testLogger)
	if got := d.Describe(img); got != "" {
		t.Fatalf("malformed-JSON Describe = %q, want empty", got)
	}
}
