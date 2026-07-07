package chatmlsink

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
)

// burstDelim is a deliberately distinctive, non-default delimiter so every burst
// assertion also proves the sink honours Options.BurstDelim (nothing hardcoded).
const burstDelim = "<<D>>"

func atMin(base time.Time, n int) time.Time { return base.Add(time.Duration(n) * time.Minute) }

// Burst preservation, owner side: N consecutive owner messages collapse into ONE
// assistant target whose content is the segments joined by Options.BurstDelim.
func TestBurstJoinsOwnerBurst(t *testing.T) {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		mkMsg(1, "100", "personal_chat", "user222", "how are you", atMin(t0, 0)),
		mkMsg(2, "100", "personal_chat", owner, "im", atMin(t0, 1)),
		mkMsg(3, "100", "personal_chat", owner, "great", atMin(t0, 2)),
		mkMsg(4, "100", "personal_chat", owner, "thanks", atMin(t0, 3)),
	}
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, BurstDelim: burstDelim})
	feed(t, s, msgs)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())
	if len(exs) != 1 {
		t.Fatalf("one owner burst = one target: want 1 example, got %d: %+v", len(exs), exs)
	}
	assertContractValid(t, exs[0])
	last := exs[0].Messages[len(exs[0].Messages)-1]
	want := "im" + burstDelim + "great" + burstDelim + "thanks"
	if last.Role != "assistant" || last.Content != want {
		t.Fatalf("owner burst not joined by BurstDelim:\n got role=%q content=%q\nwant role=assistant content=%q", last.Role, last.Content, want)
	}
	if !strings.Contains(last.Content, burstDelim) {
		t.Fatalf("assistant burst target must contain the delimiter between segments: %q", last.Content)
	}
}

// Burst preservation, user side: consecutive counterparty messages collapse into
// ONE user turn joined by the delimiter (never dropped, never concatenated raw).
func TestBurstJoinsUserBurst(t *testing.T) {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		mkMsg(1, "100", "personal_chat", "user222", "hey", atMin(t0, 0)),
		mkMsg(2, "100", "personal_chat", "user222", "you there", atMin(t0, 1)),
		mkMsg(3, "100", "personal_chat", "user222", "hello?", atMin(t0, 2)),
		mkMsg(4, "100", "personal_chat", owner, "yes im here", atMin(t0, 3)),
	}
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, BurstDelim: burstDelim})
	feed(t, s, msgs)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())
	if len(exs) != 1 {
		t.Fatalf("want 1 example, got %d: %+v", len(exs), exs)
	}
	assertContractValid(t, exs[0])
	want := "hey" + burstDelim + "you there" + burstDelim + "hello?"
	if !hasContent(exs[0], want) {
		t.Fatalf("user burst not joined into one delimited turn %q; got %+v", want, exs[0].Messages)
	}
	if roleOf(exs[0], want) != "user" {
		t.Fatalf("joined burst turn must be user, got %q", roleOf(exs[0], want))
	}
}

// A saved_messages chat yields ZERO examples — excluded purely by chat type, even
// when its messages would otherwise form a trainable user->assistant pair. A
// normal chat in the same run still emits (proves the exclusion is type-scoped).
func TestSavedMessagesExcluded(t *testing.T) {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	saved := []dialogue.Message{
		mkMsg(1, "999", "saved_messages", "user222", "note to self q", atMin(t0, 0)),
		mkMsg(2, "999", "saved_messages", owner, "note to self a", atMin(t0, 1)),
	}
	normal := []dialogue.Message{
		mkMsg(3, "100", "personal_chat", "user222", "real question", atMin(t0, 0)),
		mkMsg(4, "100", "personal_chat", owner, "real answer", atMin(t0, 1)),
	}
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6})
	feed(t, s, append(saved, normal...))
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())
	for _, e := range exs {
		for _, m := range e.Messages {
			if strings.Contains(m.Content, "note to self") {
				t.Fatalf("saved_messages content leaked into the dataset: %+v", e.Messages)
			}
		}
	}
	var sawNormal bool
	for _, e := range exs {
		if hasContent(e, "real answer") {
			sawNormal = true
		}
	}
	if !sawNormal {
		t.Fatalf("control personal_chat example missing; exclusion must be type-scoped, not a global no-op")
	}
}

// --max-per-chat caps emitted examples per ChatID, deterministically keeping the
// first N in chronological order; two runs on the same input are byte-identical.
func TestMaxPerChatCapDeterministic(t *testing.T) {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	var msgs []dialogue.Message
	for i := 0; i < 10; i++ {
		msgs = append(msgs,
			mkMsg(2*i+1, "100", "personal_chat", "user222", fmt.Sprintf("q%d", i), atMin(t0, 2*i)),
			mkMsg(2*i+2, "100", "personal_chat", owner, fmt.Sprintf("a%d", i), atMin(t0, 2*i+1)),
		)
	}
	const capN = 3
	runStr := func(maxPerChat int) string {
		var buf bytes.Buffer
		s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, MaxPerChat: maxPerChat})
		feed(t, s, msgs)
		if err := s.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		return buf.String()
	}
	uncapped := parseJSONL(t, runStr(0))
	if len(uncapped) <= capN {
		t.Fatalf("weak setup: uncapped=%d must exceed cap=%d so the cap is proven to do the reducing", len(uncapped), capN)
	}
	out1, out2 := runStr(capN), runStr(capN)
	if out1 != out2 {
		t.Fatalf("max-per-chat selection not deterministic:\n run1=%q\n run2=%q", out1, out2)
	}
	exs := parseJSONL(t, out1)
	if len(exs) != capN {
		t.Fatalf("want exactly %d examples after cap, got %d", capN, len(exs))
	}
	for i, e := range exs {
		want := fmt.Sprintf("a%d", i)
		last := e.Messages[len(e.Messages)-1]
		if last.Role != "assistant" || last.Content != want {
			t.Fatalf("example %d target = %q, want the chronological first-N target %q", i, last.Content, want)
		}
	}
}

// --cutoff-len bounds context by trimming the OLDEST context turns first; the
// final assistant target is never shortened or dropped.
func TestCutoffTrimsOldestKeepsTarget(t *testing.T) {
	t0 := time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)
	long := func(tag string) string { return tag + ": " + strings.Repeat("x", 40) }
	msgs := []dialogue.Message{
		mkMsg(1, "100", "personal_chat", "user222", long("q0"), atMin(t0, 0)),
		mkMsg(2, "100", "personal_chat", owner, long("a0"), atMin(t0, 1)),
		mkMsg(3, "100", "personal_chat", "user222", long("q1"), atMin(t0, 2)),
		mkMsg(4, "100", "personal_chat", owner, long("a1"), atMin(t0, 3)),
		mkMsg(5, "100", "personal_chat", "user222", long("q2"), atMin(t0, 4)),
		mkMsg(6, "100", "personal_chat", owner, long("a2"), atMin(t0, 5)),
	}
	const cutoff = 30 // approx tokens; far below the full 6-turn window
	var buf bytes.Buffer
	s := NewSink(&buf, Options{OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6, CutoffLen: cutoff})
	feed(t, s, msgs)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	exs := parseJSONL(t, buf.String())
	var target *outEx
	for i := range exs {
		if roleOf(exs[i], long("a2")) == "assistant" {
			target = &exs[i]
			break
		}
	}
	if target == nil {
		t.Fatalf("no example targets the final owner message a2")
	}
	assertContractValid(t, *target)
	last := target.Messages[len(target.Messages)-1]
	if last.Content != long("a2") {
		t.Fatalf("final assistant target was altered by cutoff: %q", last.Content)
	}
	if hasContent(*target, long("q0")) || hasContent(*target, long("a0")) {
		t.Fatalf("oldest context (q0/a0) not trimmed under cutoff: %+v", target.Messages)
	}
	if !hasContent(*target, long("q2")) {
		t.Fatalf("cutoff over-trimmed — lost the immediate user turn q2: %+v", target.Messages)
	}
}

// At Close the sink logs an approximate length histogram (zerolog) across the
// emitted examples.
func TestLengthHistogramLogged(t *testing.T) {
	var logBuf bytes.Buffer
	var out bytes.Buffer
	s := NewSink(&out, Options{
		OwnerFromID: owner, SessionGap: 30 * time.Minute, CtxTurns: 6,
		CutoffLen: 1024, Logger: zerolog.New(&logBuf),
	})
	feed(t, s, scenario())
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	logs := logBuf.String()
	if !strings.Contains(logs, "histogram") || !strings.Contains(logs, "approx") {
		t.Fatalf("expected an approximate length histogram log line, got: %q", logs)
	}
}
