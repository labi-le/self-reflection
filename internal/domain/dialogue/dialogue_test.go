package dialogue

import (
	"testing"
	"time"
)

func TestCleanText(t *testing.T) {
	cases := []struct {
		name string
		msg  Message
		want string
	}{
		{"plain text", Message{ExtractedText: "hello"}, "hello"},
		{"text collapses newlines", Message{ExtractedText: "a\nb\nc"}, "a b c"},
		{"text trimmed", Message{ExtractedText: "  spaced  "}, "spaced"},
		{"text wins over media", Message{ExtractedText: "caption", HasPhoto: true}, "caption"},
		{"sticker placeholder", Message{MediaType: "sticker", StickerEmoji: "😀"}, "[😀 Sticker]"},
		{"photo placeholder via HasPhoto", Message{HasPhoto: true}, "[photo]"},
		{"media_type placeholder", Message{MediaType: "voice_message"}, "[voice_message]"},
		{"video placeholder", Message{MediaType: "video_file"}, "[video_file]"},
		{"empty no media", Message{}, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := CleanText(c.msg); got != c.want {
				t.Fatalf("CleanText() = %q, want %q", got, c.want)
			}
		})
	}
}

func TestCaption(t *testing.T) {
	if got := Caption(Message{ExtractedText: "look\nhere "}); got != "look here" {
		t.Fatalf("Caption() = %q, want %q", got, "look here")
	}
	if got := Caption(Message{}); got != "" {
		t.Fatalf("Caption() empty = %q, want empty", got)
	}
}

func TestIsUseless(t *testing.T) {
	useless := []string{
		"", "[photo]", "[voice_message]", "[video_file]", "[audio_file]",
		"[animation]", "[😀 Sticker]", "[🔥 Sticker]", "https://example.com/x",
		"http://t.me/foo",
	}
	for _, s := range useless {
		if !IsUseless(s) {
			t.Errorf("IsUseless(%q) = false, want true", s)
		}
	}
	useful := []string{
		"hello", "[photo: a cat on a sofa]", "check https://example.com out",
		"[voice_message: привет]", "[Forwarded from X]: news",
	}
	for _, s := range useful {
		if IsUseless(s) {
			t.Errorf("IsUseless(%q) = true, want false", s)
		}
	}
}

func TestParseBound(t *testing.T) {
	cases := []struct {
		value string
		isEnd bool
		want  time.Time
	}{
		{"2026-06-19T10:00:00", false, time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)},
		{"2026-06-19 10:00:00", false, time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)},
		{"2026-06-19T10:00", false, time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)},
		{"2026-06-19", false, time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)},
		{"2026-06-19", true, time.Date(2026, 6, 19, 23, 59, 59, 0, time.UTC)},
		{"  2026-06-19  ", false, time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)},
	}
	for _, c := range cases {
		got, err := ParseBound(c.value, c.isEnd)
		if err != nil {
			t.Errorf("ParseBound(%q, %v) error: %v", c.value, c.isEnd, err)
			continue
		}
		if !got.Equal(c.want) {
			t.Errorf("ParseBound(%q, %v) = %v, want %v", c.value, c.isEnd, got, c.want)
		}
	}

	for _, bad := range []string{"", "not-a-date", "2026/06/19"} {
		if _, err := ParseBound(bad, false); err == nil {
			t.Errorf("ParseBound(%q) expected error, got nil", bad)
		}
	}
}

func TestDateRangeContains(t *testing.T) {
	from := time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 6, 20, 23, 59, 59, 0, time.UTC)
	inside := time.Date(2026, 6, 19, 12, 0, 0, 0, time.UTC)
	before := time.Date(2026, 6, 18, 12, 0, 0, 0, time.UTC)
	after := time.Date(2026, 6, 21, 12, 0, 0, 0, time.UTC)

	full := DateRange{From: &from, To: &to}
	if !full.Contains(inside) {
		t.Error("Contains(inside) = false, want true")
	}
	if full.Contains(before) {
		t.Error("Contains(before) = true, want false")
	}
	if full.Contains(after) {
		t.Error("Contains(after) = true, want false")
	}

	open := DateRange{}
	if !open.Contains(before) || !open.Contains(after) {
		t.Error("open DateRange should contain everything")
	}

	onlyFrom := DateRange{From: &from}
	if onlyFrom.Contains(before) {
		t.Error("onlyFrom.Contains(before) = true, want false")
	}
	if !onlyFrom.Contains(after) {
		t.Error("onlyFrom.Contains(after) = false, want true")
	}
}
