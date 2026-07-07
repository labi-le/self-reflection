package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"tg2llm/internal/bot"
)

// This file is the Wave-2 integration GATE for the --format chatml dataset
// builder: it drives the REAL pipeline (run() -> buildPipeline -> service.Run ->
// chatmlsink) exactly as the CLI does and asserts the emitted JSONL satisfies the
// LOCKED training contract in finetune/README.md — one JSON object per line,
// {"messages":[{"role","content"}...]}, an optional single leading system turn,
// then strictly alternating user<->assistant ending in assistant, every content
// non-empty. Prior tasks already implement this, so these assertions PASS today;
// they are written to FAIL loudly if the pipeline ever regresses (concrete
// expected values, exact example counts), which is the point of a gate.
//
// The chatmlTurn / chatmlExample decode types live in chatml_test.go (same
// package) and are reused here.

// personaSystemPrompt is the known --persona-system value; every emitted example
// must carry it byte-identical as the single leading system turn.
const personaSystemPrompt = "I am Denis."

// burstDelimiter is the default --burst-delim (U+2063 INVISIBLE SEPARATOR) that
// joins consecutive same-sender messages inside one merged turn.
const burstDelimiter = "\u2063"

// fixtureFullAccount is the committed full-account export fixture (owner
// user777001; a personal_chat, a public_supergroup, a left private_group, a
// service message, a photo-only message and a unixtime-only message). Resolved
// relative to this package dir (cmd/parse) — the Go test working directory.
func fixtureFullAccount(t *testing.T) string {
	t.Helper()
	p := filepath.Join("..", "..", "internal", "infra", "tgexport", "testdata", "full_account_mini.json")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("committed fixture missing at %s: %v", p, err)
	}
	return p
}

// runChatMLPipeline drives the real chatml pipeline through run() and returns the
// decoded JSONL examples. It fails the test if any line is not valid JSON, so
// assertion (a) — every output line is valid JSON — is enforced here for free.
func runChatMLPipeline(t *testing.T, inputPath string, extraFlags ...string) []chatmlExample {
	t.Helper()
	out := filepath.Join(t.TempDir(), "train.jsonl")
	args := append([]string{"--format", "chatml"}, extraFlags...)
	args = append(args, inputPath, out)
	if err := run(args); err != nil {
		t.Fatalf("run(%v): %v", args, err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output %s: %v", out, err)
	}
	trimmed := strings.TrimRight(string(raw), "\n")
	if trimmed == "" {
		return nil
	}
	var exs []chatmlExample
	for i, line := range strings.Split(trimmed, "\n") {
		var ex chatmlExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			t.Fatalf("output line %d is not valid JSON: %q: %v", i+1, line, err)
		}
		exs = append(exs, ex)
	}
	return exs
}

// assertContractValid enforces the LOCKED finetune/README.md schema on one
// example: (b) an optional single leading system, then strict user<->assistant
// with the first non-system role user and the last role assistant; (c) no turn
// has empty content.
func assertContractValid(t *testing.T, ex chatmlExample) {
	t.Helper()
	msgs := ex.Messages
	if len(msgs) == 0 {
		t.Fatalf("example has no messages")
	}
	body := msgs
	if msgs[0].Role == "system" {
		body = msgs[1:]
		for j := 1; j < len(msgs); j++ {
			if msgs[j].Role == "system" {
				t.Fatalf("a second system turn at position %d (only one leading system allowed): %+v", j, msgs)
			}
		}
	}
	if len(body) < 2 {
		t.Fatalf("example body too short (need user+assistant): %+v", msgs)
	}
	if body[len(body)-1].Role != "assistant" {
		t.Fatalf("example must end with assistant, got %q: %+v", body[len(body)-1].Role, msgs)
	}
	for j, m := range body {
		if m.Content == "" {
			t.Fatalf("empty content at body turn %d: %+v", j, msgs)
		}
		want := "user"
		if j%2 == 1 {
			want = "assistant"
		}
		if m.Role != want {
			t.Fatalf("body turn %d role = %q, want %q (strict alternation): %+v", j, m.Role, want, msgs)
		}
	}
}

func equalTurns(got, want []chatmlTurn) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func allContent(exs []chatmlExample) string {
	var b strings.Builder
	for _, ex := range exs {
		for _, m := range ex.Messages {
			b.WriteString(m.Content)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func writeTempExport(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "export.json")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write temp export: %v", err)
	}
	return p
}

// TestDatasetPipelineFullAccountFixtureContractValid is the primary end-to-end
// gate: the committed full-account fixture, --chat-scope all, --persona-system
// set, owner auto-detected from personal_information (777001). Only the
// personal_chat has an owner turn, so exactly one contract-valid example is
// emitted; the group + left chats (no owner turn) and the photo/service/unixtime
// records contribute nothing.
func TestDatasetPipelineFullAccountFixtureContractValid(t *testing.T) {
	exs := runChatMLPipeline(t, fixtureFullAccount(t),
		"--persona-system", personaSystemPrompt,
		"--chat-scope", "all",
	)

	if len(exs) != 1 {
		t.Fatalf("want exactly 1 example from the fixture, got %d: %+v", len(exs), exs)
	}
	ex := exs[0]
	assertContractValid(t, ex)

	// (d) the leading system turn is byte-identical to --persona-system.
	if ex.Messages[0].Role != "system" || ex.Messages[0].Content != personaSystemPrompt {
		t.Fatalf("leading system turn = %+v, want {system, %q}", ex.Messages[0], personaSystemPrompt)
	}

	// Concrete role mapping: the account owner (user777001) is the assistant, the
	// counterparty (user200200) is the user. Regressing owner detection or the
	// role map would break this exact shape.
	want := []chatmlTurn{
		{Role: "system", Content: personaSystemPrompt},
		{Role: "user", Content: "hi there"},
		{Role: "assistant", Content: "hello back"},
	}
	if !equalTurns(ex.Messages, want) {
		t.Fatalf("fixture example = %+v, want %+v", ex.Messages, want)
	}

	// (f) no example is derived from a chat where the owner never speaks. The
	// fixture has no saved_messages chat, but the same invariant is visible here:
	// the public_supergroup and left private_group produced nothing, so their
	// content must be absent from the training output.
	joined := allContent(exs)
	for _, forbidden := range []string{"group message", "unixtime survivor", "large id survivor"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("content %q from an owner-silent chat leaked into training output:\n%s", forbidden, joined)
		}
	}
}

// capExportJSON: a single personal_chat with three owner-answered exchanges would
// window into three examples (one per owner turn); --max-per-chat 2 must cap the
// chat at two.
const capExportJSON = `{
  "personal_information": {"user_id": 777001},
  "chats": {"list": [
    {"name":"Them","type":"personal_chat","id":900001,"messages":[
      {"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Them","from_id":"user200200","text":"u1"},
      {"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Me","from_id":"user777001","text":"a1"},
      {"id":3,"type":"message","date":"2026-06-19T10:02:00","from":"Them","from_id":"user200200","text":"u2"},
      {"id":4,"type":"message","date":"2026-06-19T10:03:00","from":"Me","from_id":"user777001","text":"a2"},
      {"id":5,"type":"message","date":"2026-06-19T10:04:00","from":"Them","from_id":"user200200","text":"u3"},
      {"id":6,"type":"message","date":"2026-06-19T10:05:00","from":"Me","from_id":"user777001","text":"a3"}
    ]}
  ]}
}`

// TestDatasetPipelineMaxPerChatCapHolds proves assertion (e): with one chat, the
// total emitted example count equals the per-chat count, so a --max-per-chat 2
// cap on a chat that would otherwise yield three examples emits exactly two.
func TestDatasetPipelineMaxPerChatCapHolds(t *testing.T) {
	input := writeTempExport(t, capExportJSON)

	// Baseline: uncapped (--max-per-chat 0) yields the full three examples, so the
	// cap below is demonstrably doing the trimming (not an accidentally-empty run).
	full := runChatMLPipeline(t, input,
		"--owner-user-id", "777001",
		"--chat-scope", "all",
		"--session-gap", "24h",
		"--max-per-chat", "0",
	)
	if len(full) != 3 {
		t.Fatalf("uncapped run should yield 3 examples, got %d: %+v", len(full), full)
	}

	capped := runChatMLPipeline(t, input,
		"--owner-user-id", "777001",
		"--chat-scope", "all",
		"--session-gap", "24h",
		"--max-per-chat", "2",
	)
	if len(capped) != 2 {
		t.Fatalf("--max-per-chat 2 on a single chat must emit exactly 2 examples, got %d: %+v", len(capped), capped)
	}
	for _, ex := range capped {
		assertContractValid(t, ex)
	}
}

// burstExportJSON: two consecutive owner messages form a burst answered to one
// preceding user message; the burst merges into one assistant turn.
const burstExportJSON = `{
  "personal_information": {"user_id": 777001},
  "chats": {"list": [
    {"name":"Them","type":"personal_chat","id":900002,"messages":[
      {"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Them","from_id":"user200200","text":"are you there"},
      {"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Me","from_id":"user777001","text":"yes"},
      {"id":3,"type":"message","date":"2026-06-19T10:02:00","from":"Me","from_id":"user777001","text":"one moment"}
    ]}
  ]}
}`

// TestDatasetPipelineBurstDelimiterJoinsSameSender proves assertion (g): a burst
// of consecutive same-sender messages is joined by U+2063 inside the merged turn,
// and that delimiter survives into the emitted example.
func TestDatasetPipelineBurstDelimiterJoinsSameSender(t *testing.T) {
	input := writeTempExport(t, burstExportJSON)
	exs := runChatMLPipeline(t, input,
		"--owner-user-id", "777001",
		"--chat-scope", "all",
	)
	if len(exs) != 1 {
		t.Fatalf("burst export must yield exactly 1 example, got %d: %+v", len(exs), exs)
	}
	ex := exs[0]
	assertContractValid(t, ex)

	last := ex.Messages[len(ex.Messages)-1]
	if last.Role != "assistant" {
		t.Fatalf("last turn must be assistant, got %q: %+v", last.Role, ex.Messages)
	}
	if !strings.Contains(last.Content, burstDelimiter) {
		t.Fatalf("assistant burst turn %q missing the U+2063 burst delimiter", last.Content)
	}
	if want := "yes" + burstDelimiter + "one moment"; last.Content != want {
		t.Fatalf("merged burst content = %q, want %q", last.Content, want)
	}
}

// savedExportJSON: a saved_messages chat (owner talking to owner) plus a normal
// personal_chat. --chat-scope all lets the saved chat reach the sink; the sink
// must still emit nothing for it.
const savedExportJSON = `{
  "personal_information": {"user_id": 777001},
  "chats": {"list": [
    {"name":"Saved Messages","type":"saved_messages","id":777001,"messages":[
      {"id":1,"type":"message","date":"2026-06-19T09:00:00","from":"Me","from_id":"user777001","text":"secret note alpha"},
      {"id":2,"type":"message","date":"2026-06-19T09:01:00","from":"Me","from_id":"user777001","text":"secret note beta"}
    ]},
    {"name":"Them","type":"personal_chat","id":900003,"messages":[
      {"id":3,"type":"message","date":"2026-06-19T10:00:00","from":"Them","from_id":"user200200","text":"hey"},
      {"id":4,"type":"message","date":"2026-06-19T10:01:00","from":"Me","from_id":"user777001","text":"hi back"}
    ]}
  ]}
}`

// TestDatasetPipelineSavedMessagesYieldsNoExample proves assertion (f): even when
// --chat-scope all routes a saved_messages chat all the way to the sink, the sink
// emits no example from it (both participants are the owner), so the saved
// content never leaks into training and only the personal_chat exchange survives.
func TestDatasetPipelineSavedMessagesYieldsNoExample(t *testing.T) {
	input := writeTempExport(t, savedExportJSON)
	exs := runChatMLPipeline(t, input,
		"--owner-user-id", "777001",
		"--chat-scope", "all",
	)
	if len(exs) != 1 {
		t.Fatalf("saved_messages must contribute 0 examples; want 1 total (the personal_chat), got %d: %+v", len(exs), exs)
	}
	assertContractValid(t, exs[0])

	joined := allContent(exs)
	for _, forbidden := range []string{"secret note alpha", "secret note beta"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("saved_messages content %q leaked into training output:\n%s", forbidden, joined)
		}
	}

	want := []chatmlTurn{
		{Role: "system", Content: bot.DefaultPersonaSystem},
		{Role: "user", Content: "hey"},
		{Role: "assistant", Content: "hi back"},
	}
	if !equalTurns(exs[0].Messages, want) {
		t.Fatalf("surviving example = %+v, want %+v", exs[0].Messages, want)
	}
}
