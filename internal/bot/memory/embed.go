// Package memory is the persona bot's long-term-memory adapter: it embeds chat
// messages with a bge-m3 llama.cpp server and stores/retrieves them in ChromaDB,
// implementing the bot.Memory retrieval port plus a Remember write path. It is
// the concrete RAG adapter the core bot package (internal/bot) only declares as
// a seam; all embedding and vector I/O lives here, wired in cmd/bot.
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

// defaultEmbedTimeout bounds a single embeddings call. bge-m3 on CPU embeds a
// chat-length input in well under a second, so this is generous headroom, not a
// tuned budget.
const defaultEmbedTimeout = 30 * time.Second

// Embedder turns text into a dense vector via an OpenAI-compatible
// /v1/embeddings endpoint (llama.cpp in --embedding mode serving bge-m3). The
// model name is sent for schema compliance; llama.cpp answers with whatever it
// loaded, so it only documents intent.
type Embedder struct {
	host  string
	model string
	http  *http.Client
}

// EmbedOption configures an Embedder at construction time.
type EmbedOption func(*Embedder)

// WithEmbedTimeout overrides the per-request timeout (values <= 0 are ignored).
func WithEmbedTimeout(d time.Duration) EmbedOption {
	return func(e *Embedder) {
		if d > 0 {
			e.http.Timeout = d
		}
	}
}

// NewEmbedder builds an Embedder for host (its base URL, e.g.
// http://persona-embed:8092) sending model as the request model name.
func NewEmbedder(host, model string, opts ...EmbedOption) *Embedder {
	e := &Embedder{
		host:  strings.TrimRight(host, "/"),
		model: model,
		http:  &http.Client{Timeout: defaultEmbedTimeout},
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// embeddingsRequest is the OpenAI /v1/embeddings request body for a single input.
type embeddingsRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// embeddingDatum is one entry of the embeddings response data array.
type embeddingDatum struct {
	Embedding []float32 `json:"embedding"`
}

// embeddingsResponse is the OpenAI /v1/embeddings response envelope.
type embeddingsResponse struct {
	Data []embeddingDatum `json:"data"`
}

// Embed returns the embedding vector for text. An empty input is rejected before
// any request; a transport, status, decode or empty-vector failure is a returned
// error so callers can degrade to "no memory" rather than store a bad vector.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("embed: empty input")
	}
	body, err := json.Marshal(embeddingsRequest{Model: e.model, Input: text})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.host+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embed response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed server returned status %d", resp.StatusCode)
	}

	var out embeddingsResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("embed response has no vector")
	}
	return out.Data[0].Embedding, nil
}
