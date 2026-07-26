package media

// CombineVideo joins a frame description and a speech transcript into one line,
// keeping whichever is non-empty. Shared by the export decoder (video notes) and
// the bot's inbound video path so both render video the same way.
func CombineVideo(desc, transcript string) string {
	switch {
	case desc != "" && transcript != "":
		return desc + " Речь: " + transcript
	case desc != "":
		return desc
	default:
		return transcript
	}
}
