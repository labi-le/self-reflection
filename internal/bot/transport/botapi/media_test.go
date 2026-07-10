package botapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

// mediaDecoderFunc adapts a function to the bot.MediaDecoder port for tests.
type mediaDecoderFunc func(ctx context.Context, kind bot.MediaKind, data []byte) string

func (f mediaDecoderFunc) Decode(ctx context.Context, kind bot.MediaKind, data []byte) string {
	return f(ctx, kind, data)
}

// Test_resolveText_downloads_and_decodes_a_voice_message proves a voice message
// is resolved via getFile + download + decode into the transcript, with the
// decoder receiving the downloaded bytes and the voice kind.
func Test_resolveText_downloads_and_decodes_a_voice_message(t *testing.T) {
	// Given a Telegram server that resolves + serves a voice file and a decoder
	var gotKind bot.MediaKind
	var gotData string
	dec := mediaDecoderFunc(func(_ context.Context, kind bot.MediaKind, data []byte) string {
		gotKind, gotData = kind, string(data)
		return "  привет это транскрипт  "
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/getFile"):
			_ = json.NewEncoder(w).Encode(getFileResponse{OK: true, Result: apiFileInfo{FilePath: "voice/f.oga"}})
		case strings.Contains(r.URL.Path, "/file/"):
			_, _ = w.Write([]byte("OGGBYTES"))
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	defer srv.Close()
	c := New("TOKEN", func(int64) bool { return true }, zerolog.Nop(), WithBaseURL(srv.URL), WithMediaDecoder(dec))

	// When
	got := c.resolveText(context.Background(), &apiMessage{Voice: &apiFileRef{FileID: "v1"}})

	// Then the trimmed transcript is returned and the decoder saw the bytes+kind
	if got != "привет это транскрипт" {
		t.Fatalf("resolveText = %q, want the transcript", got)
	}
	if gotKind != bot.MediaVoice {
		t.Fatalf("decoder kind = %d, want MediaVoice", gotKind)
	}
	if gotData != "OGGBYTES" {
		t.Fatalf("decoder data = %q, want the downloaded bytes", gotData)
	}
}

// Test_resolveText_photo_with_caption_formats_a_tag proves a captioned photo is
// rendered as "caption [фото: description]".
func Test_resolveText_photo_with_caption_formats_a_tag(t *testing.T) {
	// Given a server serving a photo file and a describing decoder
	dec := mediaDecoderFunc(func(_ context.Context, _ bot.MediaKind, _ []byte) string {
		return "рыжий кот на диване"
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/getFile") {
			_ = json.NewEncoder(w).Encode(getFileResponse{OK: true, Result: apiFileInfo{FilePath: "photos/p.jpg"}})
			return
		}
		_, _ = w.Write([]byte("JPEG"))
	}))
	defer srv.Close()
	c := New("TOKEN", func(int64) bool { return true }, zerolog.Nop(), WithBaseURL(srv.URL), WithMediaDecoder(dec))

	// When a photo with a caption is resolved
	msg := &apiMessage{Caption: "смотри", Photo: []apiPhotoSize{{FileID: "small", FileSize: 100}, {FileID: "big", FileSize: 9000}}}
	got := c.resolveText(context.Background(), msg)

	// Then the caption prefixes the labelled description
	if got != "смотри [фото: рыжий кот на диване]" {
		t.Fatalf("resolveText = %q", got)
	}
}

// Test_resolveText_download_failure_falls_back_to_placeholder proves a failed
// download still yields a placeholder (the media is acknowledged, not dropped).
func Test_resolveText_download_failure_falls_back_to_placeholder(t *testing.T) {
	// Given a server whose getFile fails
	dec := mediaDecoderFunc(func(_ context.Context, _ bot.MediaKind, _ []byte) string {
		t.Fatal("decoder must not run when download fails")
		return ""
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusInternalServerError)
	}))
	defer srv.Close()
	c := New("TOKEN", func(int64) bool { return true }, zerolog.Nop(), WithBaseURL(srv.URL), WithMediaDecoder(dec))

	// When
	got := c.resolveText(context.Background(), &apiMessage{Voice: &apiFileRef{FileID: "v1"}})

	// Then
	if got != "[голосовое сообщение]" {
		t.Fatalf("resolveText = %q, want the voice placeholder", got)
	}
}

// Test_resolveText_returns_plain_text_without_a_decoder proves media is left as
// the raw text when no decoder is wired (so the loop's empty-skip drops it).
func Test_resolveText_returns_plain_text_without_a_decoder(t *testing.T) {
	// Given a client with no decoder
	c := New("TOKEN", func(int64) bool { return true }, zerolog.Nop())

	// When a voice message (no text) is resolved
	got := c.resolveText(context.Background(), &apiMessage{Voice: &apiFileRef{FileID: "v1"}, Text: ""})

	// Then it stays empty (no download attempted)
	if got != "" {
		t.Fatalf("resolveText = %q, want empty (no decoder)", got)
	}
}

// Test_kindForMime_routes_documents proves MIME types map to the right decoder
// kind, with image/gif routed to animation and non-media types rejected.
func Test_kindForMime_routes_documents(t *testing.T) {
	cases := []struct {
		mime string
		want bot.MediaKind
		ok   bool
	}{
		{"audio/mpeg", bot.MediaAudio, true},
		{"video/mp4", bot.MediaVideo, true},
		{"image/gif", bot.MediaAnimation, true},
		{"image/png", bot.MediaPhoto, true},
		{"application/pdf", 0, false},
		{"", 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.mime, func(t *testing.T) {
			got, ok := kindForMime(tc.mime)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("kindForMime(%q) = (%d,%v), want (%d,%v)", tc.mime, got, ok, tc.want, tc.ok)
			}
		})
	}
}

// Test_resolveSticker_combines_emoji_and_description proves a sticker resolves to
// its emoji plus a vision description of the rendered frame.
func Test_resolveSticker_combines_emoji_and_description(t *testing.T) {
	// Given a server serving the sticker file and a describing decoder
	dec := mediaDecoderFunc(func(_ context.Context, kind bot.MediaKind, _ []byte) string {
		if kind != bot.MediaSticker {
			t.Errorf("decoder kind = %d, want MediaSticker", kind)
		}
		return "рыжий кот"
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/getFile") {
			_ = json.NewEncoder(w).Encode(getFileResponse{OK: true, Result: apiFileInfo{FilePath: "stickers/s.webp"}})
			return
		}
		_, _ = w.Write([]byte("WEBP"))
	}))
	defer srv.Close()
	c := New("TOKEN", func(int64) bool { return true }, zerolog.Nop(), WithBaseURL(srv.URL), WithMediaDecoder(dec))

	// When
	got := c.resolveSticker(context.Background(), &apiSticker{FileID: "s1", Emoji: "😂"})

	// Then
	if got != "[стикер 😂: рыжий кот]" {
		t.Fatalf("resolveSticker = %q", got)
	}
}

// Test_resolveSticker_emoji_only_when_undescribable proves an unrenderable
// sticker (decoder returns "") degrades to the emoji alone.
func Test_resolveSticker_emoji_only_when_undescribable(t *testing.T) {
	// Given a decoder that can't describe the sticker (e.g. animated TGS)
	dec := mediaDecoderFunc(func(_ context.Context, _ bot.MediaKind, _ []byte) string { return "" })
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/getFile") {
			_ = json.NewEncoder(w).Encode(getFileResponse{OK: true, Result: apiFileInfo{FilePath: "stickers/s.tgs"}})
			return
		}
		_, _ = w.Write([]byte("TGS"))
	}))
	defer srv.Close()
	c := New("TOKEN", func(int64) bool { return true }, zerolog.Nop(), WithBaseURL(srv.URL), WithMediaDecoder(dec))

	// When
	got := c.resolveSticker(context.Background(), &apiSticker{FileID: "s1", Emoji: "🔥"})

	// Then only the emoji is rendered
	if got != "[стикер 🔥]" {
		t.Fatalf("resolveSticker = %q, want emoji-only", got)
	}
}

// Test_formatSticker_cases pins the sticker rendering for each combination.
func Test_formatSticker_cases(t *testing.T) {
	cases := []struct{ emoji, desc, want string }{
		{"😂", "кот", "[стикер 😂: кот]"},
		{"", "кот", "[стикер: кот]"},
		{"🔥", "", "[стикер 🔥]"},
		{"", "", "[стикер]"},
	}
	for _, tc := range cases {
		if got := formatSticker(tc.emoji, tc.desc); got != tc.want {
			t.Fatalf("formatSticker(%q,%q) = %q, want %q", tc.emoji, tc.desc, got, tc.want)
		}
	}
}

// Test_formatMedia_audio_and_animation pins the labels for the audio + animation kinds.
func Test_formatMedia_audio_and_animation(t *testing.T) {
	if got := formatMedia(bot.MediaAudio, "", "текст песни"); got != "[аудио: текст песни]" {
		t.Fatalf("audio = %q", got)
	}
	if got := formatMedia(bot.MediaAnimation, "лол", "кот прыгает"); got != "лол [гифка: кот прыгает]" {
		t.Fatalf("animation = %q", got)
	}
}

// Test_largestPhoto_picks_the_biggest proves the highest file_size wins.
func Test_largestPhoto_picks_the_biggest(t *testing.T) {
	got := largestPhoto([]apiPhotoSize{{FileID: "s", FileSize: 100}, {FileID: "m", FileSize: 500}, {FileID: "l", FileSize: 9000}})
	if got != "l" {
		t.Fatalf("largestPhoto = %q, want l", got)
	}
	if got := largestPhoto(nil); got != "" {
		t.Fatalf("largestPhoto(nil) = %q, want empty", got)
	}
}

// Test_formatMedia_cases pins the rendered text for each kind, with and without
// a decode result.
func Test_formatMedia_cases(t *testing.T) {
	cases := []struct {
		name    string
		kind    bot.MediaKind
		caption string
		decoded string
		want    string
	}{
		{"voice transcript", bot.MediaVoice, "", "привет", "привет"},
		{"voice empty", bot.MediaVoice, "", "", "[голосовое сообщение]"},
		{"photo with caption", bot.MediaPhoto, "гляди", "кот", "гляди [фото: кот]"},
		{"photo empty no caption", bot.MediaPhoto, "", "", "[фото]"},
		{"video decoded", bot.MediaVideo, "", "экшн", "[видео: экшн]"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := formatMedia(tc.kind, tc.caption, tc.decoded); got != tc.want {
				t.Fatalf("formatMedia = %q, want %q", got, tc.want)
			}
		})
	}
}
