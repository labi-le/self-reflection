package dialogue

import "github.com/rs/zerolog"

// MarshalZerologObject renders a Message as structured log fields, following the
// belphegor convention of logging domain objects via zerolog's .Object(...).
func (m Message) MarshalZerologObject(e *zerolog.Event) {
	e.Int("id", m.ID).
		Time("date", m.Date).
		Str("sender", m.Sender)
	if m.MediaType != "" {
		e.Str("media_type", m.MediaType)
	}
	if m.HasPhoto {
		e.Bool("has_photo", true)
	}
	if m.ReplyToMsgID != nil {
		e.Int("reply_to", *m.ReplyToMsgID)
	}
	if m.ForwardedFrom != "" {
		e.Str("forwarded_from", m.ForwardedFrom)
	}
}
