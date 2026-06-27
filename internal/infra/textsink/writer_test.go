package textsink

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

func TestWriterFormats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")
	w, err := NewWriter(path)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if w.Path() != path {
		t.Errorf("Path() = %q, want %q", w.Path(), path)
	}
	if err := w.WriteDayHeader("2026-06-19"); err != nil {
		t.Fatalf("WriteDayHeader: %v", err)
	}
	reply := 5
	if err := w.WriteMessage(dialogue.Message{ID: 1, Date: time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC), Sender: "A"}, "hello"); err != nil {
		t.Fatalf("WriteMessage plain: %v", err)
	}
	if err := w.WriteMessage(dialogue.Message{ID: 2, Date: time.Date(2026, 6, 19, 10, 1, 0, 0, time.UTC), Sender: "B", ReplyToMsgID: &reply}, "reply text"); err != nil {
		t.Fatalf("WriteMessage reply: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	want := "\n=== 2026-06-19 ===\n" +
		"[1] [10:00:00] A: hello\n" +
		"[2] [10:01:00] B (reply to 5): reply text\n"
	if string(got) != want {
		t.Fatalf("output =\n%q\nwant\n%q", string(got), want)
	}
}

func TestNewWriterError(t *testing.T) {
	_, err := NewWriter(filepath.Join(t.TempDir(), "nope", "out.txt"))
	if err == nil {
		t.Fatal("NewWriter under nonexistent dir: err = nil, want error")
	}
}
