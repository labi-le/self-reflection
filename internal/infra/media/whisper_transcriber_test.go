package media

import (
	"path/filepath"
	"testing"
)

func TestNewTranscriberDefaults(t *testing.T) {
	tr := NewTranscriber("model.gguf", "", "", "", 0, testLogger)
	if tr.whisperBinary != "whisper-cli" {
		t.Errorf("whisperBinary = %q, want whisper-cli", tr.whisperBinary)
	}
	if tr.ffmpeg != "ffmpeg" {
		t.Errorf("ffmpeg = %q, want ffmpeg", tr.ffmpeg)
	}
	if tr.language != "ru" {
		t.Errorf("language = %q, want ru", tr.language)
	}
	if tr.timeoutSecs != 600 {
		t.Errorf("timeoutSecs = %d, want 600", tr.timeoutSecs)
	}
}

func TestTranscribeFfmpegMissingReturnsEmpty(t *testing.T) {
	tr := NewTranscriber("model.gguf", "whisper-cli-nope", "ru", "ffmpeg-does-not-exist-xyz", 30, testLogger)
	if got := tr.Transcribe(filepath.Join(t.TempDir(), "a.ogg")); got != "" {
		t.Fatalf("Transcribe with missing ffmpeg = %q, want empty", got)
	}
}
