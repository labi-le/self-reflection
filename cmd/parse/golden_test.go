package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestTextModeGoldenUnchanged pins the byte-exact output of the default
// (text-mode) pipeline. The golden bytes are the pre-Task-4 output for
// this fixture; after adding the new flags, `--format text` (the default)
// must reproduce this exactly.
func TestTextModeGoldenUnchanged(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "export.json")
	output := filepath.Join(dir, "out.txt")
	payload := `{"messages":[
		{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Alice","from_id":"user111","text":"hello"},
		{"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Bob","from_id":"user222","text":"world","reply_to_message_id":1},
		{"id":3,"type":"message","date":"2026-06-20T09:30:00","from":"Alice","from_id":"user111","text":"next day"}
	]}`
	if err := os.WriteFile(input, []byte(payload), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := run([]string{input, output}); err != nil {
		t.Fatalf("run: %v", err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	// Golden bytes come from the pre-change binary; ordering / punctuation /
	// day-header blank lines are baked in intentionally.
	want := "\n=== 2026-06-19 ===\n" +
		"[1] [10:00:00] Alice: hello\n" +
		"[2] [10:01:00] Bob (reply to 1): world\n" +
		"\n=== 2026-06-20 ===\n" +
		"[3] [09:30:00] Alice: next day\n"
	if string(got) != want {
		t.Errorf("text-mode output drifted from golden.\n--- got ---\n%s\n--- want ---\n%s", string(got), want)
	}
}
