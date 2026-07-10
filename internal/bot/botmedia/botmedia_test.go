package botmedia

import (
	"context"
	"os"
	"strings"
	"testing"

	"tg2llm/internal/bot"
)

// Test_Decode_voice_writes_a_temp_file_and_transcribes proves a voice payload is
// written to a temp file, passed to the transcriber, and its transcript trimmed.
func Test_Decode_voice_writes_a_temp_file_and_transcribes(t *testing.T) {
	// Given a decoder whose transcriber echoes the file's bytes
	var seen string
	d := &Decoder{transcribe: func(p string) string {
		b, _ := os.ReadFile(p)
		seen = string(b)
		return "  привет мир  "
	}}

	// When a voice payload is decoded
	got := d.Decode(context.Background(), bot.MediaVoice, []byte("OGGDATA"))

	// Then the transcript is returned trimmed and the temp file carried the bytes
	if got != "привет мир" {
		t.Fatalf("Decode voice = %q, want trimmed transcript", got)
	}
	if seen != "OGGDATA" {
		t.Fatalf("transcriber saw file bytes %q, want OGGDATA", seen)
	}
}

// Test_Decode_photo_describes proves a photo payload is passed to the describer.
func Test_Decode_photo_describes(t *testing.T) {
	// Given a decoder whose describer echoes the file's bytes
	d := &Decoder{describe: func(p string) string {
		b, _ := os.ReadFile(p)
		return "на фото " + string(b)
	}}

	// When a photo is decoded
	got := d.Decode(context.Background(), bot.MediaPhoto, []byte("JPG"))

	// Then the description is returned
	if got != "на фото JPG" {
		t.Fatalf("Decode photo = %q", got)
	}
}

// Test_Decode_video_combines_frame_description_and_transcript proves a video is
// both transcribed (audio) and described (first frame), combined into one line.
func Test_Decode_video_combines_frame_description_and_transcript(t *testing.T) {
	// Given a decoder with fake transcribe, frame-extract and describe
	d := &Decoder{
		transcribe: func(string) string { return "что-то сказал" },
		describe: func(p string) string {
			b, _ := os.ReadFile(p)
			return "на кадре " + string(b)
		},
		extractFrame: func(_ context.Context, _ string) (string, bool) {
			f, _ := os.CreateTemp("", "frame-*.jpg")
			_, _ = f.WriteString("FRAME")
			_ = f.Close()
			return f.Name(), true
		},
	}

	// When a video is decoded
	got := d.Decode(context.Background(), bot.MediaVideo, []byte("MP4"))

	// Then both the frame description and the transcript are present
	if !strings.Contains(got, "на кадре FRAME") || !strings.Contains(got, "что-то сказал") {
		t.Fatalf("Decode video = %q, want frame description + transcript", got)
	}
}

// Test_Decode_audio_transcribes proves a music/audio file is transcribed like a
// voice message (same whisper backend, MediaAudio kind).
func Test_Decode_audio_transcribes(t *testing.T) {
	// Given a decoder whose transcriber echoes the file's bytes
	d := &Decoder{transcribe: func(p string) string {
		b, _ := os.ReadFile(p)
		return "T:" + string(b)
	}}

	// When an audio payload is decoded
	got := d.Decode(context.Background(), bot.MediaAudio, []byte("MP3"))

	// Then it is transcribed
	if got != "T:MP3" {
		t.Fatalf("Decode audio = %q, want T:MP3", got)
	}
}

// Test_Decode_sticker_and_animation_describe_a_frame proves both kinds render a
// frame via extractFrame and describe it (shared frameDescribe path).
func Test_Decode_sticker_and_animation_describe_a_frame(t *testing.T) {
	// Given a decoder with fake frame-extract + describe
	d := &Decoder{
		describe: func(p string) string {
			b, _ := os.ReadFile(p)
			return "кадр " + string(b)
		},
		extractFrame: func(_ context.Context, _ string) (string, bool) {
			f, _ := os.CreateTemp("", "fr-*.jpg")
			_, _ = f.WriteString("FR")
			_ = f.Close()
			return f.Name(), true
		},
	}

	// When/Then a sticker and an animation both describe the frame
	for _, kind := range []bot.MediaKind{bot.MediaSticker, bot.MediaAnimation} {
		if got := d.Decode(context.Background(), kind, []byte("WEBP")); got != "кадр FR" {
			t.Fatalf("Decode(%d) = %q, want кадр FR", kind, got)
		}
	}
}

// Test_Decode_sticker_empty_when_no_frame proves an unrenderable sticker (TGS:
// extractFrame fails) yields "" without calling describe.
func Test_Decode_sticker_empty_when_no_frame(t *testing.T) {
	// Given a decoder whose frame-extract fails
	d := &Decoder{
		describe:     func(string) string { t.Fatal("describe called with no frame"); return "" },
		extractFrame: func(_ context.Context, _ string) (string, bool) { return "", false },
	}

	// When a TGS-like sticker is decoded
	if got := d.Decode(context.Background(), bot.MediaSticker, []byte("TGS")); got != "" {
		t.Fatalf("Decode sticker (no frame) = %q, want empty", got)
	}
}

// Test_Decode_returns_empty_when_backend_absent proves a kind whose backend is
// not wired decodes to "" (memory/prompt then just omit it).
func Test_Decode_returns_empty_when_backend_absent(t *testing.T) {
	// Given a decoder with no backends
	d := &Decoder{}

	// When/Then each kind yields no text
	if got := d.Decode(context.Background(), bot.MediaVoice, []byte("x")); got != "" {
		t.Fatalf("voice with no transcriber = %q, want empty", got)
	}
	if got := d.Decode(context.Background(), bot.MediaPhoto, []byte("x")); got != "" {
		t.Fatalf("photo with no describer = %q, want empty", got)
	}
}

// Test_Decode_empty_data_or_unknown_kind_is_empty proves the guards short-circuit.
func Test_Decode_empty_data_or_unknown_kind_is_empty(t *testing.T) {
	// Given a decoder whose transcriber would fail the test if called
	d := &Decoder{transcribe: func(string) string {
		t.Fatal("transcribe called for empty/unknown input")
		return ""
	}}

	// When data is empty
	if got := d.Decode(context.Background(), bot.MediaVoice, nil); got != "" {
		t.Fatalf("empty data = %q, want empty", got)
	}
	// And when the kind is unknown
	if got := d.Decode(context.Background(), bot.MediaKind(99), []byte("x")); got != "" {
		t.Fatalf("unknown kind = %q, want empty", got)
	}
}
