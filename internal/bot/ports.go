// Package bot defines the core ports (hexagonal boundaries) of the persona bot:
// a ChatTransport that carries messages to and from a chat backend, and a Brain
// that turns a conversation into the persona's next reply. Concrete adapters
// (the Telegram Bot API transport, the llama.cpp Brain) live in sub-packages and
// are wired together in cmd/bot; this package holds interfaces and value types
// only, so it depends on nothing but the standard library.
package bot

import "context"

// Update is a single inbound event delivered by a ChatTransport. It is the
// minimal shape the bot needs to gate and answer a message: which chat it
// belongs to, which user sent it, and the text. UpdateID lets a transport
// acknowledge/track its position (e.g. Bot API long-poll offsets).
type Update struct {
	UpdateID int64
	ChatID   int64
	UserID   int64
	Text     string
}

// OutgoingMessage is a single outbound text message a ChatTransport should
// deliver. A burst reply may be split into several of these before sending.
type OutgoingMessage struct {
	ChatID int64
	Text   string
}

// Message is one turn of a conversation: a role ("user", "assistant" or
// "system") and its text content. It is the shared vocabulary between the
// persisted conversation history and the Brain.
type Message struct {
	Role    string
	Content string
}

// ChatTransport abstracts a chat backend (e.g. the Telegram Bot API). Updates
// returns a channel of inbound events that closes when ctx is cancelled or the
// transport stops; Send delivers one outbound text message to a chat. No
// concrete transport is provided here — see internal/bot/transport/* .
type ChatTransport interface {
	Updates(ctx context.Context) (<-chan Update, error)
	Send(ctx context.Context, chatID int64, text string) error
}

// Brain turns a conversation history (oldest-first) into the persona's next
// reply. Implementations call the persona model; this package only declares the
// port — see internal/bot/brain/* .
type Brain interface {
	Reply(ctx context.Context, convo []Message) (string, error)
}

// MediaKind identifies the attachment on an inbound message that a MediaDecoder
// can turn into text. The zero value is intentionally invalid so an unset kind
// never silently decodes as voice.
type MediaKind int

const (
	MediaVoice MediaKind = iota + 1
	MediaPhoto
	MediaVideo
	MediaAudio
	MediaSticker
	MediaAnimation
)

// MediaDecoder turns downloaded media bytes into text the persona can read: a
// voice/video transcript or a photo/video-frame description. It returns "" when
// decoding is unavailable or fails, so the transport can fall back to a
// placeholder — see internal/bot/media . The transport owns the Telegram-specific
// download; this port owns only bytes-to-text.
type MediaDecoder interface {
	Decode(ctx context.Context, kind MediaKind, data []byte) string
}
