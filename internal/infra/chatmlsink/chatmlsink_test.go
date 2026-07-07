package chatmlsink

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

// outTurn/outEx mirror the LOCKED training contract (finetune/README.md) so the
// tests parse the emitted JSONL strictly by the same key names training uses.
type outTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type outEx struct {
	Messages []outTurn `json:"messages"`
}

func mkMsg(id int, chatID, chatType, fromID, text string, ts time.Time) dialogue.Message {
	return dialogue.Message{
		ID:            id,
		Date:          ts,
		ChatID:        chatID,
		ChatType:      chatType,
		FromID:        fromID,
		ExtractedText: text,
	}
}

func parseJSONL(t *testing.T, data string) []outEx {
	t.Helper()
	trimmed := strings.TrimRight(data, "\n")
	if trimmed == "" {
		return nil
	}
	var out []outEx
	for _, line := range strings.Split(trimmed, "\n") {
		if line == "" {
			t.Fatalf("blank line inside JSONL output")
		}
		var e outEx
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("line is not valid JSON: %q: %v", line, err)
		}
		out = append(out, e)
	}
	return out
}

// assertContractValid enforces criterion (c): lowercase roles, an optional single
// leading system, then strict user↔assistant alternation ending with assistant,
// every content non-empty.
func assertContractValid(t *testing.T, e outEx) {
	t.Helper()
	msgs := e.Messages
	i := 0
	if len(msgs) > 0 && msgs[0].Role == "system" {
		i = 1
	}
	body := msgs[i:]
	if len(body) < 2 {
		t.Fatalf("example body too short (need >=1 user and the assistant target): %+v", msgs)
	}
	if body[0].Role != "user" {
		t.Fatalf("example must start (after optional system) with user: %+v", msgs)
	}
	if body[len(body)-1].Role != "assistant" {
		t.Fatalf("example must end with assistant: %+v", msgs)
	}
	for j, m := range body {
		if m.Content == "" {
			t.Fatalf("empty content turn: %+v", msgs)
		}
		want := "user"
		if j%2 == 1 {
			want = "assistant"
		}
		if m.Role != want {
			t.Fatalf("turn %d role = %q, want %q (strict alternation): %+v", j, m.Role, want, msgs)
		}
	}
}

func hasContent(e outEx, content string) bool {
	for _, m := range e.Messages {
		if m.Content == content {
			return true
		}
	}
	return false
}

func roleOf(e outEx, content string) string {
	for _, m := range e.Messages {
		if m.Content == content {
			return m.Role
		}
	}
	return ""
}

const owner = "user111"

// scenario builds two chats: chat "100" with two sessions (a >gap jump splits it)
// and chat "200" with a single session, an owner + one counterparty alternating.
func scenario() []dialogue.Message {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	min := func(n int) time.Time { return t0.Add(time.Duration(n) * time.Minute) }
	return []dialogue.Message{
		mkMsg(1, "100", "personal_chat", "user222", "hi", min(0)),
		mkMsg(2, "100", "personal_chat", owner, "hello there", min(1)),
		mkMsg(3, "100", "personal_chat", "user222", "how are you", min(2)),
		mkMsg(4, "100", "personal_chat", owner, "good", min(3)),
		mkMsg(5, "100", "personal_chat", "user222", "still there?", t0.Add(3*time.Hour)),
		mkMsg(6, "100", "personal_chat", owner, "yeah", t0.Add(3*time.Hour+time.Minute)),
		mkMsg(7, "200", "personal_chat", "user333", "yo", min(0)),
		mkMsg(8, "200", "personal_chat", owner, "sup", min(1)),
	}
}

func feed(t *testing.T, s *Sink, msgs []dialogue.Message) {
	t.Helper()
	for _, m := range msgs {
		if err := s.WriteMessage(m, "IGNORED_FORMATTED"); err != nil {
			t.Fatalf("WriteMessage: %v", err)
		}
	}
}

// (a) WriteMessage accumulates: nothing is written to the sink before Close.
func TestBuffersUntilClose(t *testing.T) {
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, PersonaSystem: "SYS"})
	feed(t, s, scenario())
	if buf.Len() != 0 {
		t.Fatalf("sink wrote %d bytes before Close, want 0 (must buffer): %q", buf.Len(), buf.String())
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatalf("Close produced no output, want >=1 JSONL line")
	}
}

// (b,c,d,g) valid JSONL, strict alternation, correct role-map, leading system.
func TestEmitsValidJSONL(t *testing.T) {
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, PersonaSystem: "SYS"})
	feed(t, s, scenario())
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	exs := parseJSONL(t, buf.String())
	if len(exs) == 0 {
		t.Fatalf("no examples emitted")
	}

	assistantSaid := map[string]bool{"hello there": true, "good": true, "yeah": true, "sup": true}
	userSaid := map[string]bool{"hi": true, "how are you": true, "still there?": true, "yo": true}

	for _, e := range exs {
		assertContractValid(t, e)
		if e.Messages[0].Role != "system" || e.Messages[0].Content != "SYS" {
			t.Fatalf("example must start with system 'SYS': %+v", e.Messages)
		}
		for _, m := range e.Messages {
			if m.Role == "system" {
				continue
			}
			if assistantSaid[m.Content] && m.Role != "assistant" {
				t.Fatalf("owner text %q mapped to %q, want assistant", m.Content, m.Role)
			}
			if userSaid[m.Content] && m.Role != "user" {
				t.Fatalf("counterparty text %q mapped to %q, want user", m.Content, m.Role)
			}
		}
	}
}
