package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"tg2llm/internal/bot"
	"tg2llm/internal/redact"
	"tg2llm/pkg/ctxlog"
)

// Reply turns the conversation (oldest-first) into the persona's next reply: it
// generates a completion (persona-prepended, bounded by the client timeout) and
// scrubs PII by regenerating at a raised temperature up to maxRedactRetries,
// redacting as a guaranteed-clean fallback. The returned text never contains
// detected PII. Any transport, status, decode or empty-response failure on the
// FIRST generation is a returned error (never a hang, never a raw secret in the
// message).
func (c *Client) Reply(ctx context.Context, convo []bot.Message) (string, error) {
	reply, err := c.generate(ctx, convo, c.temperature)
	if err != nil {
		return "", err
	}
	if redact.Redact(reply) != reply {
		reply = c.regenerateClean(ctx, convo, reply)
	}
	return reply, nil
}

// regenerateClean re-runs generation at a progressively raised temperature until
// the model yields a PII-free reply or maxRedactRetries is exhausted. It NEVER
// returns text containing detected PII: the exhausted and transport-error paths
// fall back to redacting the last reply, which redact.Redact guarantees clean.
func (c *Client) regenerateClean(ctx context.Context, convo []bot.Message, dirty string) string {
	log := ctxlog.Op(c.logger, "Client.regenerateClean")
	last := dirty
	for attempt := 1; attempt <= c.maxRedactRetries; attempt++ {
		candidate, err := c.generate(ctx, convo, raisedTemperature(c.temperature, attempt))
		if err != nil {
			log.Warn().Err(err).Int("attempt", attempt).Msg("PII regeneration call failed; redacting last reply")
			return redact.Redact(last)
		}
		if redact.Redact(candidate) == candidate {
			return candidate
		}
		last = candidate
	}
	log.Warn().Int("retries", c.maxRedactRetries).Msg("reply still contained PII after retries; redacting")
	return redact.Redact(last)
}

// raisedTemperature nudges sampling up by 0.15 per retry so each regeneration
// diverges from the last, capped at 1.2 to stay in a coherent range.
func raisedTemperature(base float64, attempt int) float64 {
	const (
		step    = 0.15
		maxTemp = 1.2
	)
	if t := base + step*float64(attempt); t < maxTemp {
		return t
	}
	return maxTemp
}

// generate performs one persona completion at temperature: it builds the
// persona-prepended request, POSTs it under a per-request timeout, and returns
// the trimmed assistant content. An empty or failed response is an error.
func (c *Client) generate(ctx context.Context, convo []bot.Message, temperature float64) (string, error) {
	body, err := json.Marshal(c.buildRequest(convo, temperature))
	if err != nil {
		return "", fmt.Errorf("marshal persona request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.host+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build persona request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("persona request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read persona response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("persona server returned status %d", resp.StatusCode)
	}

	var result chatResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("decode persona response: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("persona response has no choices")
	}
	reply := strings.TrimSpace(result.Choices[0].Message.Content)
	if reply == "" {
		return "", fmt.Errorf("persona response is empty")
	}
	return reply, nil
}

// buildRequest assembles the OpenAI request body: the locked persona system turn
// followed by the conversation mapped 1:1, as a non-streaming completion at
// temperature.
func (c *Client) buildRequest(convo []bot.Message, temperature float64) chatRequest {
	msgs := make([]chatMessage, 0, len(convo)+1)
	msgs = append(msgs, chatMessage{Role: roleSystem, Content: c.persona})
	for _, m := range convo {
		msgs = append(msgs, chatMessage{Role: m.Role, Content: m.Content})
	}
	return chatRequest{
		Model:            c.model,
		Stream:           false,
		Temperature:      temperature,
		RepeatPenalty:    c.repeatPenalty,
		PresencePenalty:  c.presencePenalty,
		FrequencyPenalty: c.frequencyPenalty,
		RepeatLastN:      c.repeatLastN,
		Messages:         msgs,
	}
}
