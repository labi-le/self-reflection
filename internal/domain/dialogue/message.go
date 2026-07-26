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
	FilePath        string // value of "file" key (relative path for voice/video decode)
	ThumbnailPath   string // value of "thumbnail" key (video-note frame for vision decode)

	// Identity & dataset fields (populated by Task 2/4; unused by the base text
	// pipeline). Kept as prefixed strings (e.g. "user123456789", "channel1001…")
	// because Telegram peer-id tails exceed int32 and the prefix is part of the
	// identity — never squash to a bare number.
	FromID   string // prefixed sender id from the export ("from_id")
	ActorID  string // service-message actor, same prefixed form ("actor_id")
	ChatType string // one of the 11 Telegram export chat types (e.g. "personal_chat")
	ChatID   string // stable per-chat key (export chat id as string); drives --chat-scope / --max-per-chat
}
