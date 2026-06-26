package dialogue

import "time"

// Message is a domain entity representing a single chat message from a Telegram export.
// It carries all fields needed by the parser pipeline: text extraction, clean-text rules,
// media-decoding, and output formatting. The anti-corruption layer (tgexport reader)
// populates these fields from the raw JSON.
type Message struct {
	ID              int
	Date            time.Time
	Sender          string
	ExtractedText   string // result of extract_text(text_field) — raw concatenation before newline collapsing
	ReplyToMsgID    *int
	ForwardedFromID string
	ForwardedFrom   string
	HasPhoto        bool   // original JSON object had a non-empty "photo" key
	MediaType       string
	StickerEmoji    string
	PhotoPath       string // value of "photo" key (relative path for media decode)
	FilePath        string // value of "file" key (relative path for voice decode)
}
