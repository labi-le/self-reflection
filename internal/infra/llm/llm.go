package llm

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

const defaultTimeout = 600 * time.Second

// Client is an OpenAI-compatible chat + embeddings client. It carries an
// optional bearer key and always POSTs non-streaming requests.
type Client struct {
	host  string
	model string
	key   string
	http  *http.Client
}

// New builds a Client for host (base URL). timeout <= 0 uses the 600s default,
// matching the media client convention.
func New(host, model, key string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	return &Client{
		host:  strings.TrimRight(host, "/"),
		model: model,
		key:   key,
		http:  &http.Client{Timeout: timeout},
	}
}

// ChatOptions controls one chat completion. MaxTokens <= 0 omits the field;
// ResponseFormat nil omits it; Temperature is always sent.
type ChatOptions struct {
	Temperature    float64
	MaxTokens      int
	ResponseFormat json.RawMessage
}

// Chat POSTs one non-streaming completion and returns the trimmed assistant
// content of choices[0].
func (c *Client) Chat(ctx context.Context, prompt string, opts ChatOptions) (string, error) {
	payload := map[string]any{
		"model":       c.model,
		"stream":      false,
		"temperature": opts.Temperature,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
	}
	if opts.MaxTokens > 0 {
		payload["max_tokens"] = opts.MaxTokens
	}
	if opts.ResponseFormat != nil {
		payload["response_format"] = opts.ResponseFormat
	}
	data, err := c.post(ctx, "/v1/chat/completions", payload)
	if err != nil {
		return "", err
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("chat response has no choices")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

// Embed POSTs /v1/embeddings and returns one vector per input, placing each
// returned embedding at its OpenAI `index` position (the server may return them
// out of order). It errors when the response element count does not match the
// number of inputs, or when an index is out of range.
func (c *Client) Embed(ctx context.Context, model string, texts []string) ([][]float32, error) {
	data, err := c.post(ctx, "/v1/embeddings", map[string]any{
		"model": model,
		"input": texts,
	})
	if err != nil {
		return nil, err
	}
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decode embeddings response: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings response returned %d vectors for %d inputs", len(out.Data), len(texts))
	}
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(texts) {
			return nil, fmt.Errorf("embeddings response index %d out of range for %d inputs", d.Index, len(texts))
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
}

func (c *Client) post(ctx context.Context, path string, payload any) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.host+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request %s: %w", path, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s returned status %d", path, resp.StatusCode)
	}
	return data, nil
}
