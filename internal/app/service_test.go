package app_test

import (
	"os"
	"path/filepath"
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

// FIXTURE: voice transcript, photo description,
// photo+caption combine, and an undecoded video dropped as useless.
func TestServiceMediaIntegration(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 1, Date: ts(10, 0, 0), Sender: "A", MediaType: "voice_message", FilePath: "voice/a.ogg"},
		{ID: 2, Date: ts(10, 1, 0), Sender: "B", HasPhoto: true, PhotoPath: "photos/p.jpg"},
		{ID: 3, Date: ts(10, 2, 0), Sender: "A", HasPhoto: true, PhotoPath: "photos/q.jpg", ExtractedText: "смотри сюда"},
		{ID: 4, Date: ts(10, 3, 0), Sender: "B", MediaType: "video_file", FilePath: "video/v.mp4"},
	}
	dec := fakeDecoder{out: map[int]string{1: "hello world", 2: "a cat", 3: "a dog"}} // 4 -> "" (video skipped)
	got := runService(t, fakeSource{msgs: msgs}, dec, app.ParseOptions{ShowDateHeaders: true})

	assertEqual(t, got, "\n=== 2026-06-19 ===\n"+
		"[1] [10:00:00] A: [voice_message: hello world]\n"+
		"[2] [10:01:00] B: [photo: a cat]\n"+
		"[3] [10:02:00] A: смотри сюда [photo: a dog]\n")
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
