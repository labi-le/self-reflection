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
		{"video note placeholder", Message{MediaType: "video_message"}, "[video_message]"},
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
		"", "[photo]", "[voice_message]", "[video_file]", "[video_message]", "[audio_file]",
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
		"[voice_message: привет]", "[video_message: кадр Речь: привет]", "[Forwarded from X]: news",
	}
	for _, s := range useful {
		if IsUseless(s) {
			t.Errorf("IsUseless(%q) = true, want false", s)
		}
	}
}

// TestPersonaText pins the training-corpus rendering against what the bot feeds
// the model at inference (internal/bot/transport/botapi's formatMedia): a voice
// message is the words themselves, other media is a Russian tag after the caption.
func TestPersonaText(t *testing.T) {
	cases := []struct {
		name    string
		msg     Message
		decoded string
		want    string
	}{
		{"voice is the words", Message{MediaType: "voice_message"}, "привет как дела", "привет как дела"},
		{"video note", Message{MediaType: "video_message"}, "кадр Речь: ага", "[видео: кадр Речь: ага]"},
		{"video note with caption", Message{MediaType: "video_message", ExtractedText: "гляди"}, "кадр Речь: ага", "гляди [видео: кадр Речь: ага]"},
		// The read side wraps the decode verbatim, marker and all: the bot does the
		// same at inference, and the training input must match it byte for byte.
		{"video note with no frame", Message{MediaType: "video_message"}, "Речь: ага", "[видео: Речь: ага]"},
		{"photo", Message{HasPhoto: true}, "рыжий кот", "[фото: рыжий кот]"},
		{"photo with caption", Message{HasPhoto: true, ExtractedText: "смотри"}, "рыжий кот", "смотри [фото: рыжий кот]"},
		// No arm for this medium: the machine text is dropped rather than passed off
		// as something a human typed. A new decoder branch must add its own arm.
		{"unlabelled medium keeps only the caption", Message{MediaType: "audio_file", ExtractedText: "трек"}, "какой-то текст", "трек"},
		{"unlabelled medium without caption yields nothing", Message{MediaType: "audio_file"}, "какой-то текст", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := PersonaText(c.msg, c.decoded); got != c.want {
				t.Fatalf("PersonaText() = %q, want %q", got, c.want)
			}
		})
	}
}

// TestPersonaOwnText pins what may appear in a supervised target: the persona's own
// words, never a machine description of an image the bot cannot send.
func TestPersonaOwnText(t *testing.T) {
	cases := []struct {
		name    string
		msg     Message
		decoded string
		want    string
	}{
		{"own voice is the words", Message{MediaType: "voice_message"}, "я уже еду", "я уже еду"},
		{"own voice keeps its caption", Message{MediaType: "voice_message", ExtractedText: "слушай"}, "я уже еду", "слушай я уже еду"},
		{"own photo keeps only the caption", Message{HasPhoto: true, ExtractedText: "вот"}, "скриншот переписки", "вот"},
		{"own photo without caption yields nothing", Message{HasPhoto: true}, "селфи у окна", ""},
		{"own video note keeps the speech half", Message{MediaType: "video_message", ExtractedText: "гляди"}, "кадр Речь: ага", "гляди ага"},
		{"own video note with no frame keeps the speech", Message{MediaType: "video_message"}, "Речь: ага", "ага"},
		{"own video note without speech keeps the caption", Message{MediaType: "video_message", ExtractedText: "гляди"}, "просто кадр", "гляди"},
		// CombineVideo neutralises a marker quoted inside the description, so a
		// description-only decode can never look like it carries speech.
		{"description-only decode yields no speech", Message{MediaType: "video_message"}, "на кадре речь: надпись", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := PersonaOwnText(c.msg, c.decoded); got != c.want {
				t.Fatalf("PersonaOwnText() = %q, want %q", got, c.want)
			}
		})
	}
}
