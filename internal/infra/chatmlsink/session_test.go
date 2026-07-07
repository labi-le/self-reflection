package chatmlsink

import (
	"bytes"
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

// (e) a > SessionGap jump splits sessions: the "yeah" target's context must be
// only "still there?" and must NOT reach across the 2h gap into session 1.
func TestSessionGapSplits(t *testing.T) {
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, PersonaSystem: "SYS"})
	feed(t, s, scenario())
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())

	var yeah *outEx
	for i := range exs {
		if roleOf(exs[i], "yeah") == "assistant" {
			yeah = &exs[i]
			break
		}
	}
	if yeah == nil {
		t.Fatalf("no example targets the owner 'yeah' message")
	}
	for _, forbidden := range []string{"hi", "hello there", "how are you", "good"} {
		if hasContent(*yeah, forbidden) {
			t.Fatalf("session boundary leaked: 'yeah' example contains %q from session 1: %+v", forbidden, yeah.Messages)
		}
	}
	if !hasContent(*yeah, "still there?") {
		t.Fatalf("'yeah' example lost its in-session context 'still there?': %+v", yeah.Messages)
	}
}

// (f) no example mixes two ChatIDs.
func TestNoChatMixing(t *testing.T) {
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, PersonaSystem: "SYS"})
	feed(t, s, scenario())
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())

	chatA := map[string]bool{"hi": true, "hello there": true, "how are you": true, "good": true, "still there?": true, "yeah": true}
	chatB := map[string]bool{"yo": true, "sup": true}
	for _, e := range exs {
		var seenA, seenB bool
		for _, m := range e.Messages {
			if chatA[m.Content] {
				seenA = true
			}
			if chatB[m.Content] {
				seenB = true
			}
		}
		if seenA && seenB {
			t.Fatalf("example mixes chat 100 and chat 200 content: %+v", e.Messages)
		}
	}
}

// negative: a session of ONLY owner messages must not panic and must not emit any
// example lacking a user turn (or with empty content).
func TestOnlyOwnerNoInvalidExample(t *testing.T) {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		mkMsg(1, "300", "personal_chat", owner, "monologue one", t0),
		mkMsg(2, "300", "personal_chat", owner, "monologue two", t0.Add(time.Minute)),
		mkMsg(3, "300", "personal_chat", owner, "monologue three", t0.Add(2*time.Minute)),
	}
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, PersonaSystem: "SYS"})
	feed(t, s, msgs)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())
	for _, e := range exs {
		hasUser := false
		for _, m := range e.Messages {
			if m.Role == "user" {
				hasUser = true
			}
			if m.Content == "" {
				t.Fatalf("emitted an empty-content turn: %+v", e.Messages)
			}
		}
		if !hasUser {
			t.Fatalf("emitted an example with no user turn: %+v", e.Messages)
		}
	}
}

// (g complement) with no PersonaSystem, examples must NOT carry a system turn.
func TestNoPersonaOmitsSystemTurn(t *testing.T) {
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6})
	feed(t, s, scenario())
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())
	if len(exs) == 0 {
		t.Fatalf("no examples emitted")
	}
	for _, e := range exs {
		assertContractValid(t, e)
		for _, m := range e.Messages {
			if m.Role == "system" {
				t.Fatalf("system turn present despite empty PersonaSystem: %+v", e.Messages)
			}
		}
		if e.Messages[0].Role != "user" {
			t.Fatalf("first turn must be user when no persona: %+v", e.Messages)
		}
	}
}
