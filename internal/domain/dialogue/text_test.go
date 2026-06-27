package dialogue

import "testing"

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
