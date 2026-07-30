package media

import (
	"strings"

	"tg2llm/internal/domain/dialogue"
)

// CombineVideo joins a frame description and a speech transcript into one line,
// keeping whichever is non-empty. Shared by the export decoder (video notes) and
// the bot's inbound video path so both render video the same way.
//
// dialogue.SpeechMarker appears if and only if it separates the two halves: it
// rides along even with no description in front of it, and it is neutralised
// inside a description that quotes it (vision OCR of subtitles, say). That makes
// the split unambiguous for dialogue.PersonaOwnText, which must hand a training
// target the human half and never the machine one.
func CombineVideo(desc, transcript string) string {
	desc = strings.ReplaceAll(desc, dialogue.SpeechMarker, "речь: ")
	switch {
	case transcript == "":
		return desc
	case desc == "":
		return dialogue.SpeechMarker + transcript
	default:
		return desc + " " + dialogue.SpeechMarker + transcript
	}
}
