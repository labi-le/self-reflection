package tgexport

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"
)

func writeJSON(t *testing.T, payload string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "export.json")
	if err := os.WriteFile(p, []byte(payload), 0o644); err != nil {
		t.Fatalf("write json: %v", err)
	}
	return p
}

func TestReadMessagesObjectShape(t *testing.T) {
	payload := `{"messages":[
		{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Alice","text":"hello"},
		{"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Bob","text":[{"type":"link","text":"http://x"},"!"],"reply_to_message_id":1},
		{"id":3,"type":"service","date":"2026-06-19T10:02:00"},
		{"id":4,"type":"message","date":"2026-06-19T10:03:00","from":"Alice","photo":"photos/p.jpg","media_type":"photo"}
	]}`
	msgs, refs, err := NewReader(writeJSON(t, payload), zerolog.New(io.Discard)).ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 3 {
		t.Fatalf("len(msgs) = %d, want 3 (service message skipped)", len(msgs))
	}
	if msgs[0].ID != 1 || msgs[0].Sender != "Alice" || msgs[0].ExtractedText != "hello" {
		t.Errorf("msg0 = %+v", msgs[0])
	}
	if msgs[1].ExtractedText != "http://x!" {
		t.Errorf("msg1 text = %q, want %q", msgs[1].ExtractedText, "http://x!")
	}
	if msgs[1].ReplyToMsgID == nil || *msgs[1].ReplyToMsgID != 1 {
		t.Errorf("msg1 reply = %v, want 1", msgs[1].ReplyToMsgID)
	}
	if !msgs[2].HasPhoto || msgs[2].PhotoPath != "photos/p.jpg" {
		t.Errorf("msg2 photo = %+v", msgs[2])
	}
	if !refs[1] {
		t.Errorf("refs[1] missing: %v", refs)
	}
}

func TestReadMessagesBareArrayAndUnknownSender(t *testing.T) {
	payload := `[{"id":7,"type":"message","date":"2026-06-19 09:00:00"}]`
	msgs, _, err := NewReader(writeJSON(t, payload), zerolog.New(io.Discard)).ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}
	if msgs[0].Sender != "Unknown" {
		t.Errorf("sender = %q, want Unknown", msgs[0].Sender)
	}
}

func TestReadMessagesFileMissing(t *testing.T) {
	_, _, err := NewReader(filepath.Join(t.TempDir(), "nope.json"), zerolog.New(io.Discard)).ReadMessages()
	if err == nil {
		t.Fatal("missing file: err = nil, want error")
	}
}

func TestReadMessagesInvalidJSON(t *testing.T) {
	_, _, err := NewReader(writeJSON(t, "{not json"), zerolog.New(io.Discard)).ReadMessages()
	if err == nil {
		t.Fatal("invalid JSON: err = nil, want error")
	}
}
