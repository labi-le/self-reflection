package app

import (
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
)

// MessageSource is a port for reading messages from a Telegram export.
type MessageSource interface {
	// ReadMessages reads and returns all messages from the source, along with
	// referencedIDs: the set of message IDs that some message replies to. The
	// pipeline keeps an otherwise-useless message when its ID is in this set
	// (the keep-if-referenced drop rule).
	ReadMessages() (messages []dialogue.Message, referencedIDs map[int]bool, err error)
}

// DialogueSink is a port for writing formatted dialogue text.
type DialogueSink interface {
	// WriteMessage writes a single formatted message line.
	WriteMessage(msg dialogue.Message, formattedText string) error
	// WriteDayHeader writes a day-separator header.
	WriteDayHeader(date string) error
	// Close flushes and closes the sink.
	Close() error
}

// MediaDecoder is a port for decoding media (voice transcription, photo description).
type MediaDecoder interface {
	// Decode attempts to decode a media message. Returns "" if decoding is not applicable or fails.
	Decode(msg dialogue.Message) string
	// Flush persists any pending work and closes the underlying cache, so it must
	// be called exactly once, after all Decode calls have completed.
	Flush() error
}

// ParseOptions configures the parsing pipeline.
type ParseOptions struct {
	ShowDateHeaders bool
	DateRange       dialogue.DateRange
	// Jobs sets the number of parallel media-decode workers (<=1 = sequential).
	// Output ordering is preserved regardless of Jobs.
	Jobs int

	// Dataset/identity knobs (wired in later tasks; unused by the base text pipeline).
	Format        string        // "text" (default) | "chatml"
	OwnerFromID   string        // prefixed sender id treated as the assistant when --format=chatml
	ChatScope     string        // "personal" (default) | "all" — filter by ChatType before writing
	SessionGap    time.Duration // gap that starts a new turn/session (0 = disabled)
	CtxTurns      int           // context turns to prepend when materialising a training sample
	CutoffLen     int           // per-turn character cap (0 = uncapped)
	MaxPerChat    int           // cap of samples per ChatID (0 = uncapped)
	PersonaSystem string        // persona/system prompt injected into --format=chatml
	BurstDelim    string        // delimiter joining consecutive same-sender messages inside a burst
}

// MarshalZerologObject implements zerolog.LogObjectMarshaler so ParseOptions can
// be logged structurally via Event.Object.
func (o ParseOptions) MarshalZerologObject(e *zerolog.Event) {
	e.Bool("show_date_headers", o.ShowDateHeaders).Int("jobs", o.Jobs)
	if o.DateRange.From != nil {
		e.Time("date_from", *o.DateRange.From)
	}
	if o.DateRange.To != nil {
		e.Time("date_to", *o.DateRange.To)
	}
	if o.Format != "" {
		e.Str("format", o.Format)
	}
	if o.OwnerFromID != "" {
		e.Str("owner_from_id", o.OwnerFromID)
	}
	if o.ChatScope != "" {
		e.Str("chat_scope", o.ChatScope)
	}
	if o.SessionGap != 0 {
		e.Dur("session_gap", o.SessionGap)
	}
	if o.CtxTurns != 0 {
		e.Int("ctx_turns", o.CtxTurns)
	}
	if o.CutoffLen != 0 {
		e.Int("cutoff_len", o.CutoffLen)
	}
	if o.MaxPerChat != 0 {
		e.Int("max_per_chat", o.MaxPerChat)
	}
	if o.PersonaSystem != "" {
		e.Str("persona_system", o.PersonaSystem)
	}
	if o.BurstDelim != "" {
		e.Str("burst_delim", o.BurstDelim)
	}
}
