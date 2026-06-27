package media

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/pkg/ctxlog"
)

// OpenAIDescriber describes photos via an OpenAI-compatible chat-completions
// vision endpoint (e.g. a llama.cpp server started with --mmproj). The context
// window is configured on the server (-c/LLAMA_ARG_CTX_SIZE), so there is no
// per-request num_ctx.
type OpenAIDescriber struct {
	model  string
	host   string
	prompt string
	client *http.Client
	logger zerolog.Logger
}

func NewOpenAIDescriber(model, host, prompt string, logger zerolog.Logger) *OpenAIDescriber {
	if model == "" {
		model = "default"
	}
	if host == "" {
		host = "http://localhost:8080"
	}
	return &OpenAIDescriber{
		model:  model,
		host:   host,
		prompt: prompt,
		client: &http.Client{Timeout: 600 * time.Second},
		logger: logger.With().Str("component", "media-describer").Logger(),
	}
}

type oaiImageURL struct {
	URL string `json:"url"`
}

type oaiContent struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *oaiImageURL `json:"image_url,omitempty"`
}

type oaiMessage struct {
	Role    string       `json:"role"`
	Content []oaiContent `json:"content"`
}

type oaiRequest struct {
	Model       string       `json:"model"`
	Stream      bool         `json:"stream"`
	Temperature float64      `json:"temperature"`
	Messages    []oaiMessage `json:"messages"`
}

type oaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Describe sends the image at absPath to the vision server and returns a
// single-line description, or "" on any error (unreadable file, transport
// failure, non-200 status, or an unparseable/empty response).
func (d *OpenAIDescriber) Describe(absPath string) string {
	log := ctxlog.Op(d.logger, "OpenAIDescriber.Describe")

	raw, err := os.ReadFile(absPath)
	if err != nil {
		log.Warn().Err(err).Str("path", absPath).Msg("read image file")
		return ""
	}
	b64 := base64.StdEncoding.EncodeToString(raw)

	reqBody := oaiRequest{
		Model:       d.model,
		Stream:      false,
		Temperature: 0,
		Messages: []oaiMessage{{
			Role: "user",
			Content: []oaiContent{
				{Type: "text", Text: d.prompt},
				{Type: "image_url", ImageURL: &oaiImageURL{URL: "data:image/jpeg;base64," + b64}},
			},
		}},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		log.Warn().Err(err).Msg("marshal request")
		return ""
	}

	url := strings.TrimRight(d.host, "/") + "/v1/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		log.Warn().Err(err).Msg("build request")
		return ""
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		log.Warn().Err(err).Str("host", d.host).Str("model", d.model).Msg("vision request failed")
		return ""
	}
	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warn().Err(err).Msg("read vision response")
		return ""
	}
	if resp.StatusCode != http.StatusOK {
		log.Warn().Int("status", resp.StatusCode).Str("body", string(respData)).Msg("vision server returned non-200")
		return ""
	}

	var result oaiResponse
	if err := json.Unmarshal(respData, &result); err != nil {
		log.Warn().Err(err).Msg("decode vision response")
		return ""
	}
	if len(result.Choices) == 0 {
		log.Warn().Msg("vision response has no choices")
		return ""
	}
	return OneLine(result.Choices[0].Message.Content)
}
