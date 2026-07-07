package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fullAccountChatML is a minimal full-account export: owner (user777001) and a
// counterparty in one personal_chat, so chatml mode yields one user->assistant
// example with the owner mapped to "assistant".
const fullAccountChatML = `{
  "personal_information": {"user_id": 777001},
  "chats": {"list": [
    {"name":"Them","type":"personal_chat","id":200200,"messages":[
      {"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Them","from_id":"user200200","text":"hi there"},
      {"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Owner","from_id":"user777001","text":"hello back"}
    ]}
  ]}
}`

type chatmlTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatmlExample struct {
	Messages []chatmlTurn `json:"messages"`
}

// TestRunChatMLFormatSelectsSink proves --format chatml selects the chatml sink:
// the output is JSONL training examples (NOT the text dialogue), and the account
// owner's message maps to role "assistant" (auto-detected owner user777001).
func TestRunChatMLFormatSelectsSink(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "export.json")
	output := filepath.Join(dir, "out.jsonl")
	if err := os.WriteFile(input, []byte(fullAccountChatML), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := run([]string{"--format", "chatml", input, output}); err != nil {
		t.Fatalf("run: %v", err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	s := string(got)

	// chatml output must NOT be the text dialogue (no day headers).
	if strings.Contains(s, "=== ") {
		t.Fatalf("chatml mode produced text output (found a day header):\n%s", s)
	}

	trimmed := strings.TrimRight(s, "\n")
	if trimmed == "" {
		t.Fatalf("chatml mode produced no JSONL output")
	}

	sawAssistant := false
	for _, line := range strings.Split(trimmed, "\n") {
		var ex chatmlExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			t.Fatalf("chatml line is not valid JSON: %q: %v", line, err)
		}
		if len(ex.Messages) < 2 {
			t.Fatalf("chatml example too short (need user+assistant): %q", line)
		}
		for _, m := range ex.Messages {
			switch m.Content {
			case "hello back":
				if m.Role != "assistant" {
					t.Fatalf("owner text %q mapped to role %q, want assistant", m.Content, m.Role)
				}
				sawAssistant = true
			case "hi there":
				if m.Role != "user" {
					t.Fatalf("counterparty text %q mapped to role %q, want user", m.Content, m.Role)
				}
			}
		}
	}
	if !sawAssistant {
		t.Fatalf("owner message 'hello back' never appeared as assistant:\n%s", s)
	}
}
