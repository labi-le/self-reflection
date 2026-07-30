package dialogue

import (
	"regexp"
	"strings"
)

var stickerRe = regexp.MustCompile(`^\[.*? Sticker\]$`)
var bareURLRe = regexp.MustCompile(`^https?://\S+$`)

// exactUseless holds the exact strings that are considered useless placeholders.
var exactUseless = map[string]bool{
	"[photo]":         true,
	"[voice_message]": true,
	"[video_file]":    true,
	"[video_message]": true,
	"[audio_file]":    true,
	"[animation]":     true,
}

// CleanText computes the display text for a message.
// ExtractedText is joined-and-collapsed; if empty, placeholders are derived from media metadata.
func CleanText(msg Message) string {
	text := strings.ReplaceAll(msg.ExtractedText, "\n", " ")
	text = strings.TrimSpace(text)
	if text != "" {
		return text
	}

	if msg.MediaType == "sticker" {
		return "[" + msg.StickerEmoji + " Sticker]"
	}
	if msg.HasPhoto || msg.MediaType != "" {
		return "[" + MediaLabel(msg) + "]"
	}
	return ""
}

// MediaLabel returns a message's media kind: MediaType, or "photo" when only a photo is present.
func MediaLabel(msg Message) string {
	if msg.MediaType != "" {
		return msg.MediaType
	}
	return "photo"
}

// FormatDecoded renders decoded media text as a "[label: decoded]" tag.
func FormatDecoded(msg Message, decoded string) string {
	return "[" + MediaLabel(msg) + ": " + decoded + "]"
}

// PersonaText renders decoded media for a turn the persona READS. It mirrors the
// bot's inference rendering for the three media the decoder handles
// (internal/bot/transport/botapi's formatMedia): a voice message IS the speaker's
// words, a video note and a photo become a Russian tag after any caption. A medium
// with no arm here keeps only the human caption — a decoder branch added later must
// gain an explicit arm in lock-step with formatMedia rather than leaking untagged
// machine prose into the corpus.
func PersonaText(msg Message, decoded string) string {
	decoded = strings.TrimSpace(decoded)
	switch {
	case msg.MediaType == "voice_message":
		return decoded
	case msg.MediaType == "video_message":
		return withCaption(Caption(msg), "[видео: "+decoded+"]")
	case msg.HasPhoto:
		return withCaption(Caption(msg), "[фото: "+decoded+"]")
	default:
		return Caption(msg)
	}
}

// SpeechMarker labels the spoken half of a video decode built by
// internal/infra/media.CombineVideo. CombineVideo emits it whenever there IS
// speech, so its absence unambiguously means "frame description only" — that is
// what lets the owner-side rendering keep the human half and drop the machine one.
const SpeechMarker = "Речь: "

// PersonaOwnText renders decoded media for a turn the persona WROTE. Owner turns are
// the supervised target, so only what the human contributed may appear: their spoken
// words plus any caption. A machine description of a photo or a video frame must
// never become something the model is trained to emit — the bot can only send text,
// so a generated "[фото: …]" would be delivered to the chat partner verbatim.
func PersonaOwnText(msg Message, decoded string) string {
	decoded = strings.TrimSpace(decoded)
	switch msg.MediaType {
	case "voice_message":
		return withCaption(Caption(msg), decoded)
	case "video_message":
		// Only the speech half is the persona's own. CombineVideo guarantees the
		// marker appears only as the separator — it neutralises the marker inside a
		// description that quotes it — so the first match is the split point.
		if _, speech, ok := strings.Cut(decoded, SpeechMarker); ok {
			return withCaption(Caption(msg), speech)
		}
	}
	return Caption(msg)
}

func withCaption(caption, tag string) string {
	if caption == "" {
		return tag
	}
	return caption + " " + tag
}

// Caption returns the extracted-and-flattened text suitable for a media caption.
func Caption(msg Message) string {
	return strings.TrimSpace(strings.ReplaceAll(msg.ExtractedText, "\n", " "))
}

// IsUseless reports whether text is a dead placeholder that should trigger message-dropping.
func IsUseless(text string) bool {
	if text == "" {
		return true
	}
	if exactUseless[text] {
		return true
	}
	if stickerRe.MatchString(text) {
		return true
	}
	if bareURLRe.MatchString(text) {
		return true
	}
	return false
}
