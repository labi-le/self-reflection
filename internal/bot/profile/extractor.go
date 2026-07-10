package profile

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

const (
	defaultTimeout   = 120 * time.Second
	defaultMaxTokens = 400
)

// Extractor turns a prompt into text via an OpenAI-compatible chat endpoint. It
// is the seam the Builder depends on (a general instruct model in production, a
// fake in tests).
type Extractor interface {
	Extract(ctx context.Context, prompt string) (string, error)
}

// openaiExtractor calls a llama.cpp /v1/chat/completions endpoint. It caps the
// output (max_tokens) so a degenerate model can never run away, and uses a low
// temperature for stable extraction.
type openaiExtractor struct {
	host  string
	model string
	http  *http.Client
}

// NewExtractor builds an Extractor for host (base URL of a general instruct model
// — NOT the fine-tuned persona, which loops on structured extraction).
func NewExtractor(host, model string, timeout time.Duration) Extractor {
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	if model == "" {
		model = "default"
	}
	return &openaiExtractor{
		host:  strings.TrimRight(host, "/"),
		model: model,
		http:  &http.Client{Timeout: timeout},
	}
}

func (e *openaiExtractor) Extract(ctx context.Context, prompt string) (string, error) {
	body, err := json.Marshal(map[string]any{
		"model":       e.model,
		"stream":      false,
		"temperature": 0.2,
		"max_tokens":  defaultMaxTokens,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
	})
	if err != nil {
		return "", fmt.Errorf("marshal extract request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.host+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build extract request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("extract request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read extract response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("extractor returned status %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return "", fmt.Errorf("decode extract response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("extract response has no choices")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}
