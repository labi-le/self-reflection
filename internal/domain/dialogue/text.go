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
