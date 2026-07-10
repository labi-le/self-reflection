package botapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"tg2llm/internal/bot"
	"tg2llm/pkg/ctxlog"
)

// Updates starts the long-poll loop and returns a channel of allowed inbound
// updates. The channel closes when ctx is cancelled or the transport stops.
func (c *Client) Updates(ctx context.Context) (<-chan bot.Update, error) {
	out := make(chan bot.Update)
	go c.pollLoop(ctx, out)
	return out, nil
}

// pollLoop drives getUpdates. It advances the offset past every update it sees
// (so none is redelivered) but emits only those whose sender passes the
// allowlist; a transient poll failure is logged and retried after a backoff,
// never fatal.
func (c *Client) pollLoop(ctx context.Context, out chan<- bot.Update) {
	defer close(out)
	log := ctxlog.Op(c.logger, "Client.Updates")
	var offset int64
	for {
		if ctx.Err() != nil {
			return
		}
		batch, err := c.getUpdates(ctx, offset)
		if err != nil {
			if ctx.Err() != nil {
				return // cancelled, not a failure
			}
			log.Warn().Err(err).Dur("backoff", retryBackoff).Msg("getUpdates failed; backing off")
			if !c.sleep(ctx, retryBackoff) {
				return
			}
			continue
		}
		for _, u := range batch {
			offset = u.UpdateID + 1 // acknowledge, even if dropped below
			msg := u.Message
			if msg == nil || msg.From == nil {
				continue // non-message update: nothing to answer
			}
			if !c.allow(msg.From.ID) {
				continue // allowlist gate at the edge — never surface it
			}
			text := c.resolveText(ctx, msg)
			if strings.TrimSpace(text) == "" {
				continue // sticker/unsupported media/empty — nothing to answer
			}
			select {
			case out <- bot.Update{UpdateID: u.UpdateID, ChatID: msg.Chat.ID, UserID: msg.From.ID, Text: text}:
			case <-ctx.Done():
				return
			}
		}
	}
}

// getUpdates performs one long-poll. The per-call deadline is the poll hold plus
// slack, so a wedged connection cannot outlive the poll.
func (c *Client) getUpdates(ctx context.Context, offset int64) ([]apiUpdate, error) {
	reqCtx, cancel := context.WithTimeout(ctx, pollTimeout+pollSlack)
	defer cancel()
	req := getUpdatesRequest{Offset: offset, Timeout: int(pollTimeout / time.Second)}
	var resp getUpdatesResponse
	if err := c.postJSON(reqCtx, "getUpdates", req, &resp); err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, fmt.Errorf("getUpdates: telegram reported ok=false")
	}
	return resp.Result, nil
}

// Send delivers text to chatID. A persona burst reply (segments joined by
// U+2063) is split and each segment sent as its own message, each preceded by a
// typing indicator and a human-like pace delay so the bot never answers
// instantly. The first error stops the burst and is returned.
func (c *Client) Send(ctx context.Context, chatID int64, text string) error {
	for i, seg := range c.segments(text) {
		c.typing(ctx, chatID) // cosmetic; failures are non-fatal
		if !c.sleep(ctx, c.typingDelay(seg)) {
			return ctx.Err()
		}
		if err := c.sendMessage(ctx, chatID, seg); err != nil {
			return fmt.Errorf("send segment %d: %w", i, err)
		}
	}
	return nil
}

func (c *Client) sendMessage(ctx context.Context, chatID int64, text string) error {
	reqCtx, cancel := context.WithTimeout(ctx, sendTimeout)
	defer cancel()
	return c.postJSON(reqCtx, "sendMessage", sendMessageRequest{ChatID: chatID, Text: text}, nil)
}

// typing shows a "typing…" indicator. It is best-effort: a failure is logged
// (token-redacted) and swallowed, because the pacing indicator must never block
// message delivery.
func (c *Client) typing(ctx context.Context, chatID int64) {
	reqCtx, cancel := context.WithTimeout(ctx, sendTimeout)
	defer cancel()
	if err := c.postJSON(reqCtx, "sendChatAction", chatActionRequest{ChatID: chatID, Action: "typing"}, nil); err != nil {
		log := ctxlog.Op(c.logger, "Client.typing")
		log.Debug().Err(err).Msg("typing indicator failed")
	}
}

// postJSON marshals payload, POSTs it to method, and decodes a 200 body into out
// (nil to discard). Every error path that could contain the request URL — and
// thus the bot token — is redacted via sanitize before returning, so the token
// never escapes this package.
func (c *Client) postJSON(ctx context.Context, method string, payload, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("%s: marshal: %w", method, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.methodURL(method), bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("%s: build request: %s", method, c.sanitize(err.Error()))
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %s", method, c.sanitize(err.Error()))
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("%s: read body: %w", method, err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: status %d: %s", method, resp.StatusCode, c.sanitize(string(data)))
	}
	if out != nil {
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("%s: decode: %w", method, err)
		}
	}
	return nil
}
