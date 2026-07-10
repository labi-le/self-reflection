// Package botapi is a Telegram Bot API long-poll transport: the concrete
// bot.ChatTransport that carries messages to and from Telegram over plain HTTPS
// (getUpdates in, sendMessage out), using only the standard library — no
// third-party SDK. It is the Phase-1 chatbot ingress/egress: allowlist-gated at
// the edge, and it splits a persona burst reply (segments joined by U+2063) into
// separate, human-paced messages.
package botapi

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

// Compile-time proof that Client satisfies the transport port.
var _ bot.ChatTransport = (*Client)(nil)

const (
	defaultBaseURL = "https://api.telegram.org"
	// burstDelim joins consecutive persona segments in an outbound reply; each
	// segment is delivered as its own Telegram message. U+2063 INVISIBLE
	// SEPARATOR is the project-wide burst delimiter (see --burst-delim).
	burstDelim = "\u2063"
	// tokenRedaction masks the bot token anywhere it might surface in an error
	// string, so the secret never reaches a log or a returned error.
	tokenRedaction = "«token»"

	pollTimeout  = 25 * time.Second // getUpdates long-poll hold
	pollSlack    = 15 * time.Second // client deadline headroom over the hold
	sendTimeout  = 30 * time.Second // per outbound request deadline
	retryBackoff = 2 * time.Second  // pause after a transient getUpdates failure

	typingBase    = 1000 * time.Millisecond // base reading/thinking pause before a message
	typingPerRune = 55 * time.Millisecond   // human-like compose time per rune
	typingCap     = 5 * time.Second         // …but never dwell longer than this
	typingJitter  = 0.3                     // ± random fraction so pacing never feels metronomic
)

// Client is the Telegram Bot API transport. Construct it with New. The bot token
// is held privately and is NEVER logged: every error that could embed the token
// (all carry it in the request path) is redacted before it leaves this package.
type Client struct {
	token   string
	baseURL string
	allow   func(userID int64) bool
	http    *http.Client
	logger  zerolog.Logger
	// sleep is the cancellable delay seam (backoff + inter-segment pacing),
	// swapped for an instant recorder in tests. It returns false if ctx ended.
	sleep func(ctx context.Context, d time.Duration) bool
	// randFloat is the [0,1) randomness seam for probabilistic word-splitting,
	// swapped for a scripted source in tests. Defaults to rand.Float64.
	randFloat func() float64
	// splitWordProb is the break probability at each inter-word gap (0 disables
	// word-splitting); splitMaxMsgs caps the resulting message count (0 = none).
	splitWordProb float64
	splitMaxMsgs  int
	// decoder turns inbound voice/photo/video into text; nil leaves the bot
	// text-only (media messages fall back to a placeholder).
	decoder bot.MediaDecoder
}

// Option customises a Client at construction.
type Option func(*Client)

// WithBaseURL overrides the Telegram API base (default https://api.telegram.org),
// so tests can point the transport at an httptest.Server.
func WithBaseURL(base string) Option {
	return func(c *Client) {
		if base != "" {
			c.baseURL = strings.TrimRight(base, "/")
		}
	}
}

// WithMediaDecoder wires a decoder so inbound voice/photo/video are downloaded
// and turned into text; without it the transport stays text-only.
func WithMediaDecoder(d bot.MediaDecoder) Option {
	return func(c *Client) {
		if d != nil {
			c.decoder = d
		}
	}
}

// withSleep swaps the delay seam — test-only, so pacing is instant and
// observable instead of wall-clock.
func withSleep(fn func(ctx context.Context, d time.Duration) bool) Option {
	return func(c *Client) {
		if fn != nil {
			c.sleep = fn
		}
	}
}

// New builds a transport for the given bot token. allow is the allowlist gate
// consulted for every inbound update's sender; updates failing it are dropped at
// the edge and never surface (a nil allow fails closed). The token is stored but
// never logged.
func New(token string, allow func(userID int64) bool, logger zerolog.Logger, opts ...Option) *Client {
	if allow == nil {
		allow = func(int64) bool { return false }
	}
	c := &Client{
		token:     token,
		baseURL:   defaultBaseURL,
		allow:     allow,
		http:      &http.Client{}, // no global timeout; a per-request context bounds each call
		logger:    logger.With().Str("component", "bot-botapi").Logger(),
		sleep:     realSleep,
		randFloat: rand.Float64,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// realSleep waits d, returning early (false) if ctx is cancelled first.
func realSleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// methodURL builds the Bot API endpoint for method. The token lives in the path,
// so the result must never be logged or embedded verbatim in a returned error.
func (c *Client) methodURL(method string) string {
	return c.baseURL + "/bot" + c.token + "/" + method
}

// sanitize redacts the bot token from s so it is safe to log or return.
func (c *Client) sanitize(s string) string {
	if c.token == "" {
		return s
	}
	return strings.ReplaceAll(s, c.token, tokenRedaction)
}

// splitBurst divides an outbound reply into its burst segments on the U+2063
// delimiter, trimming each and dropping any left blank (Telegram rejects empty
// messages). A reply with no delimiter yields a single segment.
func splitBurst(text string) []string {
	var out []string
	for _, seg := range strings.Split(text, burstDelim) {
		if seg = strings.TrimSpace(seg); seg != "" {
			out = append(out, seg)
		}
	}
	return out
}

// typingDelay is the human-like pause before a segment: a base reading/thinking
// delay plus a per-rune compose time, randomly jittered (±typingJitter) so
// consecutive messages never arrive on a metronome, and capped so a long message
// never stalls the conversation.
func (c *Client) typingDelay(segment string) time.Duration {
	d := typingBase + time.Duration(len([]rune(segment)))*typingPerRune
	d = time.Duration(float64(d) * (1 + typingJitter*(2*c.randFloat()-1)))
	if d > typingCap {
		return typingCap
	}
	return d
}
