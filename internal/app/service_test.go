package app_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"tg2llm/internal/app"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/infra/textsink"
)

type fakeSource struct {
	msgs []dialogue.Message
	refs map[int]bool
}

func (f fakeSource) ReadMessages() ([]dialogue.Message, map[int]bool, error) {
	return f.msgs, f.refs, nil
}

type fakeDecoder struct {
	out map[int]string
}

func (f fakeDecoder) Decode(msg dialogue.Message) string { return f.out[msg.ID] }
func (f fakeDecoder) Flush() error                       { return nil }

func runService(t *testing.T, src app.MessageSource, dec app.MediaDecoder, opts app.ParseOptions) string {
	t.Helper()
	out := filepath.Join(t.TempDir(), "out.txt")
	w, err := textsink.NewWriter(out)
	if err != nil {
		t.Fatalf("writer: %v", err)
	}
	if err := app.NewParseService(src, w, dec, opts).Run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read out: %v", err)
	}
	return string(b)
}

func ts(h, m, s int) time.Time {
	return time.Date(2026, 6, 19, h, m, s, 0, time.UTC)
}

func assertEqual(t *testing.T, got, want string) {
	t.Helper()
	if got != want {
		t.Fatalf("output mismatch:\n got: %q\nwant: %q", got, want)
	}
}

// FIXTURE: voice transcript, photo description, photo+caption combine, a decoded
// video note, and undecoded video / video note dropped as useless.
func TestServiceMediaIntegration(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 1, Date: ts(10, 0, 0), Sender: "A", MediaType: "voice_message", FilePath: "voice/a.ogg"},
		{ID: 2, Date: ts(10, 1, 0), Sender: "B", HasPhoto: true, PhotoPath: "photos/p.jpg"},
		{ID: 3, Date: ts(10, 2, 0), Sender: "A", HasPhoto: true, PhotoPath: "photos/q.jpg", ExtractedText: "смотри сюда"},
		{ID: 4, Date: ts(10, 3, 0), Sender: "B", MediaType: "video_file", FilePath: "video/v.mp4"},
		{ID: 5, Date: ts(10, 4, 0), Sender: "A", MediaType: "video_message", FilePath: "round/v.mp4", ThumbnailPath: "round/v.jpg"},
		{ID: 6, Date: ts(10, 5, 0), Sender: "B", MediaType: "video_message", FilePath: "round/w.mp4"},
	}
	// 4 and 6 decode to "" -> bare placeholders -> dropped.
	dec := fakeDecoder{out: map[int]string{1: "hello world", 2: "a cat", 3: "a dog", 5: "кадр Речь: привет"}}
	got := runService(t, fakeSource{msgs: msgs}, dec, app.ParseOptions{ShowDateHeaders: true})

	assertEqual(t, got, "\n=== 2026-06-19 ===\n"+
		"[1] [10:00:00] A: [voice_message: hello world]\n"+
		"[2] [10:01:00] B: [photo: a cat]\n"+
		"[3] [10:02:00] A: смотри сюда [photo: a dog]\n"+
		"[5] [10:04:00] A: [video_message: кадр Речь: привет]\n")
}

func TestServiceUselessUnlessReferenced(t *testing.T) {
	reply := 5
	msgs := []dialogue.Message{
		{ID: 5, Date: ts(10, 0, 0), Sender: "A", HasPhoto: true, PhotoPath: "photos/x.jpg"}, // [photo], referenced -> kept
		{ID: 6, Date: ts(10, 1, 0), Sender: "B", HasPhoto: true, PhotoPath: "photos/y.jpg"}, // [photo], not referenced -> dropped
		{ID: 7, Date: ts(10, 2, 0), Sender: "A", ExtractedText: "ответ", ReplyToMsgID: &reply},
	}
	got := runService(t, fakeSource{msgs: msgs, refs: map[int]bool{5: true}}, nil, app.ParseOptions{ShowDateHeaders: true})

	assertEqual(t, got, "\n=== 2026-06-19 ===\n"+
		"[5] [10:00:00] A: [photo]\n"+
		"[7] [10:02:00] A (reply to 5): ответ\n")
}

func TestServiceForwardWrapping(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 8, Date: ts(10, 0, 0), Sender: "A", ExtractedText: "breaking news", ForwardedFrom: "Some Channel"},
	}
	got := runService(t, fakeSource{msgs: msgs}, nil, app.ParseOptions{ShowDateHeaders: true})

	assertEqual(t, got, "\n=== 2026-06-19 ===\n"+
		"[8] [10:00:00] A: [Forwarded from Some Channel]: breaking news\n")
}

func TestServiceDateRangeFilter(t *testing.T) {
	from := time.Date(2026, 6, 19, 10, 1, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 9, Date: ts(10, 0, 0), Sender: "A", ExtractedText: "early"}, // before from -> filtered
		{ID: 10, Date: ts(10, 2, 0), Sender: "B", ExtractedText: "kept"},
	}
	opts := app.ParseOptions{ShowDateHeaders: true, DateRange: dialogue.DateRange{From: &from}}
	got := runService(t, fakeSource{msgs: msgs}, nil, opts)

	assertEqual(t, got, "\n=== 2026-06-19 ===\n[10] [10:02:00] B: kept\n")
}

func TestServiceMultiDayHeaders(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 1, Date: time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC), Sender: "A", ExtractedText: "day1"},
		{ID: 2, Date: time.Date(2026, 6, 20, 9, 0, 0, 0, time.UTC), Sender: "B", ExtractedText: "day2"},
	}
	got := runService(t, fakeSource{msgs: msgs}, nil, app.ParseOptions{ShowDateHeaders: true})

	assertEqual(t, got, "\n=== 2026-06-19 ===\n[1] [10:00:00] A: day1\n"+
		"\n=== 2026-06-20 ===\n[2] [09:00:00] B: day2\n")
}

func TestServiceNoHeaders(t *testing.T) {
	msgs := []dialogue.Message{{ID: 1, Date: ts(10, 0, 0), Sender: "A", ExtractedText: "hi"}}
	got := runService(t, fakeSource{msgs: msgs}, nil, app.ParseOptions{ShowDateHeaders: false})

	assertEqual(t, got, "[1] [10:00:00] A: hi\n")
}

// TestServiceParallelMatchesSequential verifies the output is byte-identical
// whether decoding runs sequentially (Jobs=1) or across many workers (Jobs=8),
// proving parallel decode never reorders or alters the result. Run with -race.
func TestServiceParallelMatchesSequential(t *testing.T) {
	day2 := func(h, m, s int) time.Time { return time.Date(2026, 6, 20, h, m, s, 0, time.UTC) }
	msgs := []dialogue.Message{
		{ID: 1, Date: ts(10, 0, 0), Sender: "A", MediaType: "voice_message", FilePath: "v/1.ogg"},
		{ID: 2, Date: ts(10, 1, 0), Sender: "B", HasPhoto: true, PhotoPath: "p/2.jpg"},
		{ID: 3, Date: ts(10, 2, 0), Sender: "A", ExtractedText: "просто текст"},
		{ID: 4, Date: ts(10, 3, 0), Sender: "B", HasPhoto: true, PhotoPath: "p/4.jpg", ExtractedText: "смотри"},
		{ID: 5, Date: day2(9, 0, 0), Sender: "A", MediaType: "voice_message", FilePath: "v/5.ogg"},
		{ID: 6, Date: day2(9, 1, 0), Sender: "B", ExtractedText: "ещё"},
	}
	decOut := map[int]string{1: "привет", 2: "кот", 4: "пёс", 5: "пока"}

	seq := runService(t, fakeSource{msgs: msgs}, fakeDecoder{out: decOut}, app.ParseOptions{ShowDateHeaders: true, Jobs: 1})
	par := runService(t, fakeSource{msgs: msgs}, fakeDecoder{out: decOut}, app.ParseOptions{ShowDateHeaders: true, Jobs: 8})

	assertEqual(t, par, seq)
	if seq == "" {
		t.Fatal("sequential output is empty")
	}
}

func TestServiceEmptyInput(t *testing.T) {
	got := runService(t, fakeSource{}, nil, app.ParseOptions{ShowDateHeaders: true})
	assertEqual(t, got, "")
}

// captureSink records what a run hands the sink: the message, whose ExtractedText
// the chatml sink turns into a training turn, and the formatted line, which only
// the text sink writes.
type captureSink struct {
	msgs      []dialogue.Message
	formatted []string
}

func (c *captureSink) WriteDayHeader(string) error { return nil }
func (c *captureSink) WriteMessage(m dialogue.Message, text string) error {
	c.msgs = append(c.msgs, m)
	c.formatted = append(c.formatted, text)
	return nil
}
func (c *captureSink) Close() error { return nil }

func runCapture(t *testing.T, dec app.MediaDecoder, opts app.ParseOptions, msgs []dialogue.Message) *captureSink {
	t.Helper()
	sink := &captureSink{}
	if err := app.NewParseService(fakeSource{msgs: msgs}, sink, dec, opts).Run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	return sink
}

// TestServiceChatmlFoldsDecodedMedia proves a chatml run carries decoded media in
// the message's own text — the sink builds turns from ExtractedText and skips empty
// ones, so without the fold every voice message and video note vanishes from the
// corpus — and that the rendering depends on the side. A turn the persona READS
// mirrors the bot's inference input; a turn it WROTE is the supervised target and
// may carry only what the human said, never a machine description the bot has no
// way to send.
func TestServiceChatmlFoldsDecodedMedia(t *testing.T) {
	const owner = "userOwner"
	msgs := []dialogue.Message{
		{ID: 1, Date: ts(10, 0, 0), FromID: "userOther", MediaType: "voice_message", FilePath: "v/1.ogg"},
		{ID: 2, Date: ts(10, 1, 0), FromID: "userOther", HasPhoto: true, PhotoPath: "p/2.jpg", ExtractedText: "гляди"},
		{ID: 3, Date: ts(10, 2, 0), FromID: "userOther", MediaType: "video_message", FilePath: "r/3.mp4"},
		{ID: 4, Date: ts(10, 3, 0), FromID: "userOther", MediaType: "voice_message", FilePath: "v/4.ogg", ForwardedFrom: "Some Channel"},
		{ID: 5, Date: ts(10, 4, 0), FromID: owner, MediaType: "voice_message", FilePath: "v/5.ogg"},
		{ID: 6, Date: ts(10, 5, 0), FromID: owner, HasPhoto: true, PhotoPath: "p/6.jpg", ExtractedText: "вот"},
		{ID: 7, Date: ts(10, 6, 0), FromID: owner, HasPhoto: true, PhotoPath: "p/7.jpg"},
	}
	dec := fakeDecoder{out: map[int]string{
		1: "привет как дела", 2: "рыжий кот", 3: "кадр Речь: ага", 4: "переслано голосом",
		5: "я уже еду", 6: "скриншот переписки", 7: "селфи у окна",
	}}

	sink := runCapture(t, dec, app.ParseOptions{Format: "chatml", ChatScope: "all", OwnerFromID: owner}, msgs)

	want := map[int]string{
		1: "привет как дела",
		2: "гляди [фото: рыжий кот]",
		3: "[видео: кадр Речь: ага]",
		4: "переслано голосом",
		5: "я уже еду", // owner voice: his own words are a legitimate target
		6: "вот",       // owner photo: the caption only, never the description
	}
	if len(sink.msgs) != len(want) {
		t.Fatalf("wrote %d messages, want %d (7 has no caption, so nothing of the owner's is left)", len(sink.msgs), len(want))
	}
	for i, m := range sink.msgs {
		if m.ExtractedText != want[m.ID] {
			t.Errorf("msg %d turn text = %q, want %q", m.ID, m.ExtractedText, want[m.ID])
		}
		if strings.Contains(m.ExtractedText, "Forwarded from") {
			t.Errorf("msg %d carries the forward prefix into a training turn: %q", m.ID, m.ExtractedText)
		}
		// The prefix belongs to the formatted line the text sink writes, and is the
		// reason the two strings must stay decoupled.
		if m.ID == 4 && !strings.HasPrefix(sink.formatted[i], "[Forwarded from Some Channel]: ") {
			t.Errorf("formatted line for msg 4 = %q, want the forward prefix", sink.formatted[i])
		}
	}
}

// TestServiceTextFormatKeepsExtractedText proves the fold is chatml-only: the text
// pipeline still renders media into the output line and leaves the message alone.
func TestServiceTextFormatKeepsExtractedText(t *testing.T) {
	msgs := []dialogue.Message{{ID: 1, Date: ts(10, 0, 0), Sender: "A", MediaType: "voice_message", FilePath: "v/1.ogg"}}
	dec := fakeDecoder{out: map[int]string{1: "привет"}}

	sink := runCapture(t, dec, app.ParseOptions{}, msgs)
	if len(sink.msgs) != 1 || sink.msgs[0].ExtractedText != "" {
		t.Fatalf("text mode message = %+v, want ExtractedText untouched", sink.msgs)
	}
	if sink.formatted[0] != "[voice_message: привет]" {
		t.Fatalf("text mode line = %q, want the reading tag", sink.formatted[0])
	}
}

// chatAwareDecoder answers per chat, so a test can tell which chat's decode landed
// on a message.
type chatAwareDecoder struct{}

func (chatAwareDecoder) Decode(msg dialogue.Message) string { return "decode of chat " + msg.ChatID }
func (chatAwareDecoder) Flush() error                       { return nil }

// TestServiceDecodesPerChatNotPerID proves decodes are keyed by chat AND id. A
// Telegram message id is unique only inside its chat, and a full-account export
// flattens every chat into one slice, so an id-only key lets one chat's transcript
// surface on another chat's message — in chatml mode, straight into a training turn.
func TestServiceDecodesPerChatNotPerID(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 42, Date: ts(10, 0, 0), Sender: "A", ChatID: "111", ChatType: "personal_chat", MediaType: "voice_message", FilePath: "v/a.ogg"},
		{ID: 42, Date: ts(10, 1, 0), Sender: "B", ChatID: "222", ChatType: "personal_chat", MediaType: "voice_message", FilePath: "v/b.ogg"},
	}
	for _, jobs := range []int{1, 4} {
		sink := runCapture(t, chatAwareDecoder{}, app.ParseOptions{Format: "chatml", ChatScope: "all", Jobs: jobs}, msgs)
		if len(sink.msgs) != 2 {
			t.Fatalf("jobs=%d: wrote %d messages, want 2", jobs, len(sink.msgs))
		}
		for _, m := range sink.msgs {
			if want := "decode of chat " + m.ChatID; m.ExtractedText != want {
				t.Errorf("jobs=%d: chat %s got %q, want %q", jobs, m.ChatID, m.ExtractedText, want)
			}
		}
	}
}

// countingDecoder records how many messages a run actually decoded.
type countingDecoder struct{ calls *int }

func (c countingDecoder) Decode(dialogue.Message) string { *c.calls++; return "расшифровка" }
func (c countingDecoder) Flush() error                   { return nil }

// TestServiceDecodesOnlyInScopeChats proves the decoder is not spent on messages the
// run then drops. A full-account export carries tens of thousands of media messages
// in groups and channels; with --chat-scope personal every one of them is a whisper
// or vision call for a message that never reaches the corpus.
func TestServiceDecodesOnlyInScopeChats(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 1, Date: ts(10, 0, 0), ChatID: "1", ChatType: "personal_chat", MediaType: "voice_message", FilePath: "v/1.ogg"},
		{ID: 2, Date: ts(10, 1, 0), ChatID: "2", ChatType: "public_supergroup", MediaType: "voice_message", FilePath: "v/2.ogg"},
	}
	for _, jobs := range []int{1, 4} {
		calls := 0
		runCapture(t, countingDecoder{calls: &calls}, app.ParseOptions{Format: "chatml", ChatScope: "personal", Jobs: jobs}, msgs)
		if calls != 1 {
			t.Errorf("jobs=%d: decoded %d messages, want 1 (the supergroup is out of scope)", jobs, calls)
		}
	}
	// The text pipeline has no scope concept and must keep decoding everything.
	calls := 0
	runCapture(t, countingDecoder{calls: &calls}, app.ParseOptions{ChatScope: "personal"}, msgs)
	if calls != 2 {
		t.Errorf("text mode decoded %d messages, want 2 (scope is chatml-only)", calls)
	}
}
