package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// buildJSONL builds a synthetic ChatML/"messages" dataset of k examples spread
// across 3 chat ids and 3 coarse time buckets. Even-indexed examples carry an
// assistant turn (eligible for held-out); odd-indexed do not.
func buildJSONL(k int) (data string, eligible int) {
	chats := []string{"100", "200", "300"}
	months := []string{"2026-05", "2026-06", "2026-07"}
	var b strings.Builder
	for i := 0; i < k; i++ {
		chat := chats[i%len(chats)]
		month := months[(i/2)%len(months)]
		date := month + "-01T10:00:00"
		if i%2 == 0 {
			eligible++
			fmt.Fprintf(&b, `{"chat_id":%s,"date":%q,"messages":[{"role":"user","content":"q%d"},{"role":"assistant","content":"a%d"}]}`+"\n", chat, date, i, i)
		} else {
			fmt.Fprintf(&b, `{"chat_id":%s,"date":%q,"messages":[{"role":"user","content":"q%d"}]}`+"\n", chat, date, i)
		}
	}
	return b.String(), eligible
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// assertAssistant asserts on the CONTRACT (a message with role "assistant"),
// not merely that the field is non-empty.
func assertAssistant(t *testing.T, raw string) {
	t.Helper()
	var top struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(raw), &top); err != nil {
		t.Fatalf("held example is not valid JSON: %v (%s)", err, raw)
	}
	for _, m := range top.Messages {
		if m.Role == "assistant" {
			return
		}
	}
	t.Fatalf("held example has no assistant turn: %s", raw)
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}

// TestSplit_PartitionDisjointAndComplete: train+held == total, disjoint by
// example identity, held has the requested size, every held example is eligible.
func TestSplit_PartitionDisjointAndComplete(t *testing.T) {
	const k = 60
	data, _ := buildJSONL(k)
	exs, err := parseExamples(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parseExamples: %v", err)
	}
	if len(exs) != k {
		t.Fatalf("len(exs) = %d, want %d", len(exs), k)
	}

	train, held := splitExamples(exs, 42, 10, nil)

	if len(train)+len(held) != len(exs) {
		t.Fatalf("train+held = %d, want %d", len(train)+len(held), len(exs))
	}
	if len(held) != 10 {
		t.Fatalf("len(held) = %d, want 10", len(held))
	}

	seen := make(map[int]bool, len(exs))
	for _, i := range train {
		if seen[i] {
			t.Fatalf("duplicate index %d in train", i)
		}
		seen[i] = true
	}
	for _, i := range held {
		if seen[i] {
			t.Fatalf("index %d present in BOTH train and held (overlap)", i)
		}
		seen[i] = true
	}
	if len(seen) != len(exs) {
		t.Fatalf("union size = %d, want %d (partition incomplete)", len(seen), len(exs))
	}
	for _, i := range held {
		if !exs[i].hasAssistant {
			t.Fatalf("held example index %d has no assistant turn", i)
		}
		assertAssistant(t, exs[i].raw)
	}
}

// TestSplit_Deterministic: same seed => identical partition; a different seed
// still yields a valid, correctly-sized held-out set.
func TestSplit_Deterministic(t *testing.T) {
	data, _ := buildJSONL(60)
	exs1, err := parseExamples(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parseExamples: %v", err)
	}
	exs2, err := parseExamples(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parseExamples: %v", err)
	}

	tr1, h1 := splitExamples(exs1, 42, 10, nil)
	tr2, h2 := splitExamples(exs2, 42, 10, nil)
	if !equalInts(tr1, tr2) || !equalInts(h1, h2) {
		t.Fatalf("split is non-deterministic for seed=42:\n  held1=%v\n  held2=%v", h1, h2)
	}

	_, hOther := splitExamples(exs1, 7, 10, nil)
	if len(hOther) != 10 {
		t.Fatalf("seed=7 held size = %d, want 10", len(hOther))
	}
}

// TestSplit_HeldoutAlwaysContainsAssistant: no matter the seed, every held-out
// example is an eligible (assistant-bearing) one.
func TestSplit_HeldoutAlwaysContainsAssistant(t *testing.T) {
	data, _ := buildJSONL(40)
	exs, err := parseExamples(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parseExamples: %v", err)
	}
	for _, seed := range []int64{1, 42, 7, 999} {
		_, held := splitExamples(exs, seed, 15, nil)
		for _, i := range held {
			assertAssistant(t, exs[i].raw)
		}
	}
}

// TestSplit_CapsWhenHeldoutExceedsEligible: request > eligible => cap at
// floor(0.1*total) and warn.
func TestSplit_CapsWhenHeldoutExceedsEligible(t *testing.T) {
	const k = 20 // eligible = 10, floor(0.1*20) = 2
	data, elig := buildJSONL(k)
	if elig != 10 {
		t.Fatalf("fixture eligible = %d, want 10", elig)
	}
	exs, err := parseExamples(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parseExamples: %v", err)
	}
	var warned bool
	_, held := splitExamples(exs, 42, 999, func(string, ...any) { warned = true })
	if !warned {
		t.Fatalf("expected a warning when heldout(999) > eligible(10)")
	}
	if len(held) != 2 {
		t.Fatalf("capped held size = %d, want 2 (floor(0.1*20))", len(held))
	}
	for _, i := range held {
		assertAssistant(t, exs[i].raw)
	}
}

// TestParse_MalformedLineErrors: a malformed JSON line must error (not be
// silently skipped) and the message must point at the offending line.
func TestParse_MalformedLineErrors(t *testing.T) {
	data := `{"messages":[{"role":"assistant","content":"ok"}]}` + "\n" + `{bad json}` + "\n"
	_, err := parseExamples(strings.NewReader(data))
	if err == nil {
		t.Fatalf("expected error on malformed JSON line, got nil")
	}
	if !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("error should identify line 2, got: %v", err)
	}
}

// TestParse_MissingMessagesErrors: a line lacking the "messages" array violates
// the dataset contract and must error.
func TestParse_MissingMessagesErrors(t *testing.T) {
	data := `{"foo":1}` + "\n"
	_, err := parseExamples(strings.NewReader(data))
	if err == nil {
		t.Fatalf("expected error on line missing \"messages\", got nil")
	}
}

// TestRun_MalformedInputNonZeroExit: run() returns an error (=> non-zero exit)
// on a malformed input file.
func TestRun_MalformedInputNonZeroExit(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "in.jsonl")
	content := `{"messages":[{"role":"assistant","content":"x"}]}` + "\n" + `{nope}` + "\n"
	if err := os.WriteFile(in, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	err := run([]string{
		"--in", in,
		"--train-out", filepath.Join(dir, "t.jsonl"),
		"--heldout-out", filepath.Join(dir, "h.jsonl"),
		"--heldout", "1",
	})
	if err == nil {
		t.Fatalf("run() should return an error on malformed input")
	}
}

// TestRun_ByteIdenticalReRun: same seed => byte-identical output files
// (idempotent / stale-state safe).
func TestRun_ByteIdenticalReRun(t *testing.T) {
	dir := t.TempDir()
	data, _ := buildJSONL(50)
	in := filepath.Join(dir, "in.jsonl")
	if err := os.WriteFile(in, []byte(data), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	t1, h1 := filepath.Join(dir, "t1.jsonl"), filepath.Join(dir, "h1.jsonl")
	t2, h2 := filepath.Join(dir, "t2.jsonl"), filepath.Join(dir, "h2.jsonl")

	args1 := []string{"--in", in, "--seed", "42", "--heldout", "8", "--train-out", t1, "--heldout-out", h1}
	args2 := []string{"--in", in, "--seed", "42", "--heldout", "8", "--train-out", t2, "--heldout-out", h2}
	if err := run(args1); err != nil {
		t.Fatalf("run #1: %v", err)
	}
	if err := run(args2); err != nil {
		t.Fatalf("run #2: %v", err)
	}

	if string(readFile(t, t1)) != string(readFile(t, t2)) {
		t.Fatalf("train outputs differ between identical-seed runs")
	}
	if string(readFile(t, h1)) != string(readFile(t, h2)) {
		t.Fatalf("heldout outputs differ between identical-seed runs")
	}

	// And prove the run-level partition is complete + disjoint on real files.
	trainLines := nonEmptyLines(string(readFile(t, t1)))
	heldLines := nonEmptyLines(string(readFile(t, h1)))
	if len(trainLines)+len(heldLines) != 50 {
		t.Fatalf("train+held lines = %d, want 50", len(trainLines)+len(heldLines))
	}
	set := make(map[string]bool, 50)
	for _, l := range trainLines {
		set[l] = true
	}
	for _, l := range heldLines {
		if set[l] {
			t.Fatalf("line present in both train and held: %s", l)
		}
	}
	for _, l := range heldLines {
		assertAssistant(t, l)
	}
}

func nonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}
