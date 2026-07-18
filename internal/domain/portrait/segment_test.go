package portrait

import (
	"strings"
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

func TestSegment_Empty(t *testing.T) {
	if got := Segment(nil, 30*time.Minute, 1000); len(got) != 0 {
		t.Fatalf("empty input => %d chunks, want 0", len(got))
	}
}

func TestSegment_SessionBoundary(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 1, FromID: "userA", Sender: "A", Date: base, ExtractedText: "hi"},
		{ID: 2, FromID: "userB", Sender: "B", Date: base.Add(time.Minute), ExtractedText: "hey"},
		{ID: 3, FromID: "userA", Sender: "A", Date: base.Add(90 * time.Minute), ExtractedText: "back"},
	}
	chunks := Segment(msgs, 30*time.Minute, 10000)
	if len(chunks) < 2 {
		t.Fatalf("want >=2 chunks across a session boundary, got %d", len(chunks))
	}
	for i, c := range chunks {
		if c.Index != i {
			t.Fatalf("chunk[%d].Index = %d, want %d", i, c.Index, i)
		}
	}
}

func TestSegment_OversizedSessionHardSplit(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	var msgs []dialogue.Message
	for i := 1; i <= 20; i++ {
		msgs = append(msgs, dialogue.Message{
			ID: i, FromID: "userA", Sender: "A",
			Date:          base.Add(time.Duration(i) * time.Second),
			ExtractedText: "message number filler text",
		})
	}
	const chunkChars = 80
	chunks := Segment(msgs, 30*time.Minute, chunkChars)
	if len(chunks) < 2 {
		t.Fatalf("oversized single session should split, got %d chunk(s)", len(chunks))
	}
	seen := map[int]bool{}
	for i, c := range chunks {
		if c.Index != i {
			t.Fatalf("chunk[%d].Index = %d", i, c.Index)
		}
		if n := len([]rune(c.Text)); n > chunkChars {
			t.Fatalf("chunk %d text len %d exceeds %d", i, n, chunkChars)
		}
		for id := range c.MemberIDs {
			seen[id] = true
		}
	}
	if len(seen) != 20 {
		t.Fatalf("union of MemberIDs = %d, want 20", len(seen))
	}
}

func TestSegment_TextFormat(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 7, FromID: "userA", Sender: "Alice", Date: base, ExtractedText: "hello world"},
	}
	chunks := Segment(msgs, 30*time.Minute, 10000)
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "[7] userA (Alice): hello world") {
		t.Fatalf("text = %q, want rendered line", chunks[0].Text)
	}
	if !chunks[0].MemberIDs[7] {
		t.Fatalf("MemberIDs missing id 7: %+v", chunks[0].MemberIDs)
	}
}
