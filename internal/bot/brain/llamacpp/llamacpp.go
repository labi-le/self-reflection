// Package llamacpp is the persona Brain: the concrete bot.Brain that turns a
// conversation into the persona's next reply by calling a llama.cpp server over
// its OpenAI-compatible /v1/chat/completions endpoint. It is model-independent
// (it targets whatever fine-tuned model the persona chat server loads) and
// depends only on the standard library, zerolog and the bot port. The locked
// persona system prompt is injected as the leading system turn of every request
// and MUST stay byte-identical to the string used at fine-tune time.
package llamacpp

import (
	"net/http"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

// Compile-time proof that Client satisfies the Brain port.
var _ bot.Brain = (*Client)(nil)

const (
	defaultHost             = "http://localhost:8081"
	defaultModel            = "default"
	defaultTemperature      = 0.3
	defaultRepeatPenalty    = 1.1
	defaultPresencePenalty  = 0.3
	defaultFrequencyPenalty = 0.0
	defaultRepeatLastN      = 0
	defaultTimeout          = 120 * time.Second
	defaultMaxRedactRetries = 3
)

// Client is the llama.cpp persona Brain. Construct it with New. It carries no
// secret (the persona server needs no token), POSTs to host's OpenAI-compatible
// endpoint, and bounds every call with a per-request context timeout.
type Client struct {
	host             string
	persona          string
	model            string
	temperature      float64
	repeatPenalty    float64
	presencePenalty  float64
	frequencyPenalty float64
	repeatLastN      int
	timeout          time.Duration
	maxRedactRetries int
	http             *http.Client
	logger           zerolog.Logger
}

// Option customises a Client at construction.
type Option func(*Client)

// WithTemperature overrides the sampling temperature (default 0.3). Negative
// values are ignored.
func WithTemperature(t float64) Option {
	return func(c *Client) {
		if t >= 0 {
			c.temperature = t
		}
	}
}

// WithRepeatPenalty overrides the llama.cpp repeat_penalty (default 1.1), which
// curbs the short-reply repetition loop a bare temperature leaves unchecked.
// Negative values are ignored.
func WithRepeatPenalty(p float64) Option {
	return func(c *Client) {
		if p >= 0 {
			c.repeatPenalty = p
		}
	}
}

// WithPresencePenalty overrides the OpenAI-standard presence_penalty (default
// 0.3), which nudges the model off tokens it has already emitted. Negative
// values are ignored.
func WithPresencePenalty(p float64) Option {
	return func(c *Client) {
		if p >= 0 {
			c.presencePenalty = p
		}
	}
}

// WithFrequencyPenalty overrides the OpenAI-standard frequency_penalty (default
// 0, off). Unlike presence_penalty's flat nudge it scales with how often a token
// has already appeared, so it bites hardest on the runaway loops the persona
// falls into. Negative values are ignored.
func WithFrequencyPenalty(p float64) Option {
	return func(c *Client) {
		if p >= 0 {
			c.frequencyPenalty = p
		}
	}
}

// WithRepeatLastN sets how many recent tokens the repetition penalties look back
// over (llama.cpp repeat_last_n; default 0 keeps the server's own 64-token
// window). A wider window catches a phrase the persona repeated several turns
// ago, which the default window misses. Negative values are ignored.
func WithRepeatLastN(n int) Option {
	return func(c *Client) {
		if n >= 0 {
			c.repeatLastN = n
		}
	}
}

// WithTimeout overrides the per-request timeout (default 120s). Non-positive
// values are ignored.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		if d > 0 {
			c.timeout = d
		}
	}
}

// WithLogger attaches a logger (default: disabled).
func WithLogger(logger zerolog.Logger) Option {
	return func(c *Client) {
		c.logger = logger.With().Str("component", "bot-brain-llamacpp").Logger()
	}
}

// WithMaxRedactRetries sets how many times Reply regenerates at a raised
// temperature when the model returns PII before falling back to redacting the
// last reply (default 3). Negative values are ignored; 0 redacts a dirty reply
// immediately (no regeneration).
func WithMaxRedactRetries(n int) Option {
	return func(c *Client) {
		if n >= 0 {
			c.maxRedactRetries = n
		}
	}
}

// New builds a persona Brain targeting host's OpenAI-compatible endpoint. Blank
// host/personaSystem/model fall back to the package defaults (host :8081, the
// locked persona system prompt, model "default"). Options tune temperature,
// timeout and logging.
func New(host, personaSystem, model string, opts ...Option) *Client {
	if host == "" {
		host = defaultHost
	}
	if personaSystem == "" {
		personaSystem = bot.DefaultPersonaSystem
	}
	if model == "" {
		model = defaultModel
	}
	c := &Client{
		host:             strings.TrimRight(host, "/"),
		persona:          personaSystem,
		model:            model,
		temperature:      defaultTemperature,
		repeatPenalty:    defaultRepeatPenalty,
		presencePenalty:  defaultPresencePenalty,
		frequencyPenalty: defaultFrequencyPenalty,
		repeatLastN:      defaultRepeatLastN,
		timeout:          defaultTimeout,
		maxRedactRetries: defaultMaxRedactRetries,
		http:             &http.Client{}, // no global timeout; a per-request ctx bounds each call
		logger:           zerolog.Nop(),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}
