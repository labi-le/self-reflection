// Package chatmlsink implements an app.DialogueSink that converts the owner's
// Telegram dialogue into ChatML/sharegpt training examples (finetune/README.md
// contract). It buffers every message in memory keyed by ChatID and materialises
// the JSONL only at Close: one JSON object per line,
// {"messages":[{"role":...,"content":...},...]} with an optional leading system
// turn, strict user/assistant alternation, and an assistant-terminated target.
package chatmlsink

import (
	"bufio"
	"encoding/json"
	"io"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/app"
	"tg2llm/internal/domain/dialogue"
)

// Sink satisfies app.DialogueSink.
var _ app.DialogueSink = (*Sink)(nil)

const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"
)

// Options carries the resolved owner and the ParseOptions fields this sink needs.
// Grouping them keeps the constructor to a single value parameter.
type Options struct {
	OwnerFromID   string        // prefixed sender id treated as the assistant target
	PersonaSystem string        // optional system prompt injected as the first turn (empty = omit)
	BurstDelim    string        // delimiter joining consecutive same-sender messages inside one turn
	SessionGap    time.Duration // gap between consecutive messages that starts a new session (0 = never split)
	CtxTurns      int           // max preceding turns prepended as context for each target
	CutoffLen     int           // approximate-token cap that bounds context turns (0 = uncapped)
	MaxPerChat    int           // cap of emitted examples per ChatID (0 = unlimited)
	Logger        zerolog.Logger // optional; the zero value is a safe no-op (writes nowhere)
}

// turn and example marshal to the locked training schema.
type turn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type example struct {
	Messages []turn `json:"messages"`
}

// chatBuffer accumulates one chat's messages in arrival (chronological) order and
// remembers its ChatType alongside.
type chatBuffer struct {
	chatType string
	msgs     []dialogue.Message
}

// Sink buffers messages per ChatID and emits training JSONL at Close.
type Sink struct {
	w     io.Writer
	opts  Options
	order []string // ChatIDs in first-seen order (deterministic output)
	chats map[string]*chatBuffer
}

// NewSink builds a buffering ChatML sink writing to w.
func NewSink(w io.Writer, opts Options) *Sink {
	return &Sink{
		w:     w,
		opts:  opts,
		chats: make(map[string]*chatBuffer),
	}
}

// WriteMessage accumulates msg into the per-ChatID buffer. It never writes to the
// underlying writer (materialisation happens once, at Close). The formatted text
// produced by the service is intentionally ignored: a training turn's content is
// the message's ExtractedText, which the service pre-folds decoded media into
// (dialogue.PersonaText / PersonaOwnText, see app/service.go) so media messages
// are not dropped.
func (s *Sink) WriteMessage(msg dialogue.Message, _ string) error {
	b, ok := s.chats[msg.ChatID]
	if !ok {
		b = &chatBuffer{chatType: msg.ChatType}
		s.chats[msg.ChatID] = b
		s.order = append(s.order, msg.ChatID)
	}
	b.msgs = append(b.msgs, msg)
	return nil
}

// WriteDayHeader is a no-op: ChatML output has no day separators. It exists only
// to satisfy app.DialogueSink.
func (s *Sink) WriteDayHeader(string) error { return nil }

// Close transforms every buffered chat into training examples and writes them as
// JSONL (one object per line) to the underlying writer.
func (s *Sink) Close() error {
	var emitted []example
	for _, id := range s.order {
		emitted = append(emitted, s.chatExamples(s.chats[id])...)
	}
	s.logLengthHistogram(emitted)

	bw := bufio.NewWriter(s.w)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)
	for _, ex := range emitted {
		if err := enc.Encode(ex); err != nil {
			return err
		}
	}
	return bw.Flush()
}
