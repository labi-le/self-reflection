package app

import (
	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
)

// MessageSource is a port for reading messages from a Telegram export.
type MessageSource interface {
	// ReadMessages reads and returns all messages from the source.
	// Also returns the set of referenced message IDs (used for drop-rule exemption).
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
	// Flush persists any pending work (e.g. cache writes).
	Flush() error
}

// ParseOptions configures the parsing pipeline.
type ParseOptions struct {
	ShowDateHeaders bool
	DateRange       dialogue.DateRange
	// Jobs sets the number of parallel media-decode workers (<=1 = sequential).
	// Output ordering is preserved regardless of Jobs.
	Jobs int
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
}
