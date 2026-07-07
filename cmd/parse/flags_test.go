package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"tg2llm/internal/bot"
)

func TestParseFlagsDefaults(t *testing.T) {
	opts, positional, err := parseFlags([]string{"in.json"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if len(positional) != 1 || positional[0] != "in.json" {
		t.Fatalf("positional = %v, want [in.json]", positional)
	}
	if opts.Format != "text" {
		t.Errorf("Format = %q, want %q", opts.Format, "text")
	}
	if opts.OwnerFromID != "" {
		t.Errorf("OwnerFromID = %q, want empty", opts.OwnerFromID)
	}
	if opts.ChatScope != "personal" {
		t.Errorf("ChatScope = %q, want %q", opts.ChatScope, "personal")
	}
	if opts.SessionGap != 30*time.Minute {
		t.Errorf("SessionGap = %v, want 30m", opts.SessionGap)
	}
	if opts.CtxTurns != 6 {
		t.Errorf("CtxTurns = %d, want 6", opts.CtxTurns)
	}
	if opts.CutoffLen != 1024 {
		t.Errorf("CutoffLen = %d, want 1024", opts.CutoffLen)
	}
	if opts.MaxPerChat != 4000 {
		t.Errorf("MaxPerChat = %d, want 4000", opts.MaxPerChat)
	}
	if opts.PersonaSystem != bot.DefaultPersonaSystem {
		t.Errorf("PersonaSystem = %q, want the shared default %q", opts.PersonaSystem, bot.DefaultPersonaSystem)
	}
	if opts.BurstDelim != "\u2063" {
		t.Errorf("BurstDelim = %q, want U+2063", opts.BurstDelim)
	}
}

func TestParseFlagsOverrides(t *testing.T) {
	args := []string{
		"--format", "chatml",
		"--chat-scope", "all",
		"--session-gap", "45m",
		"--ctx-turns", "8",
		"--cutoff-len", "2048",
		"--max-per-chat", "0",
		"--burst-delim", "|",
		"--persona-system", "you are a helpful assistant",
		"in.json", "out.txt",
	}
	opts, positional, err := parseFlags(args)
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if len(positional) != 2 || positional[0] != "in.json" || positional[1] != "out.txt" {
		t.Fatalf("positional = %v, want [in.json out.txt]", positional)
	}
	if opts.Format != "chatml" {
		t.Errorf("Format = %q, want chatml", opts.Format)
	}
	if opts.ChatScope != "all" {
		t.Errorf("ChatScope = %q, want all", opts.ChatScope)
	}
	if opts.SessionGap != 45*time.Minute {
		t.Errorf("SessionGap = %v, want 45m", opts.SessionGap)
	}
	if opts.CtxTurns != 8 {
		t.Errorf("CtxTurns = %d, want 8", opts.CtxTurns)
	}
	if opts.CutoffLen != 2048 {
		t.Errorf("CutoffLen = %d, want 2048", opts.CutoffLen)
	}
	if opts.MaxPerChat != 0 {
		t.Errorf("MaxPerChat = %d, want 0", opts.MaxPerChat)
	}
	if opts.BurstDelim != "|" {
		t.Errorf("BurstDelim = %q, want |", opts.BurstDelim)
	}
	if opts.PersonaSystem != "you are a helpful assistant" {
		t.Errorf("PersonaSystem = %q, want literal string", opts.PersonaSystem)
	}
}

func TestParseFlagsOwnerUserIDBareNumber(t *testing.T) {
	opts, _, err := parseFlags([]string{"--owner-user-id", "12345", "in.json"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if opts.OwnerFromID != "user12345" {
		t.Errorf("OwnerFromID = %q, want user12345", opts.OwnerFromID)
	}
}

func TestParseFlagsOwnerUserIDAlreadyPrefixed(t *testing.T) {
	opts, _, err := parseFlags([]string{"--owner-user-id", "user123", "in.json"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if opts.OwnerFromID != "user123" {
		t.Errorf("OwnerFromID = %q, want passthrough user123", opts.OwnerFromID)
	}
}

func TestParseFlagsPersonaSystemTildeExpansion(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home dir: %v", err)
	}
	// Create an actual file at ~/tg2llm-test-persona.txt so the path-heuristic branch triggers.
	rel := ".tg2llm-persona-test-" + t.Name() + ".txt"
	tildePath := "~/" + rel
	absPath := filepath.Join(home, rel)
	body := "SYSTEM PROMPT FROM FILE\n"
	if err := os.WriteFile(absPath, []byte(body), 0o644); err != nil {
		t.Fatalf("write persona file: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(absPath) })

	opts, _, err := parseFlags([]string{"--persona-system", tildePath, "in.json"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	// Path exists → contents loaded; the leading '~' must have been expanded.
	if opts.PersonaSystem != body {
		t.Errorf("PersonaSystem = %q, want file body %q", opts.PersonaSystem, body)
	}
}

func TestParseFlagsPersonaSystemLiteralWhenNotAFile(t *testing.T) {
	// A string that does not resolve to an existing file must pass through verbatim.
	literal := "act as Alice, respond concisely"
	opts, _, err := parseFlags([]string{"--persona-system", literal, "in.json"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if opts.PersonaSystem != literal {
		t.Errorf("PersonaSystem = %q, want literal %q", opts.PersonaSystem, literal)
	}
}

func TestParseFlagsExistingFlagsStillWork(t *testing.T) {
	// Guard against accidentally reordering / dropping the pre-existing flags.
	args := []string{
		"--jobs", "5",
		"--date-from", "2026-06-19",
		"--verbose",
		"in.json",
	}
	opts, positional, err := parseFlags(args)
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if opts.Jobs != 5 {
		t.Errorf("Jobs = %d, want 5", opts.Jobs)
	}
	if opts.DateRange.From == nil {
		t.Errorf("DateRange.From is nil, want set")
	}
	if len(positional) != 1 || positional[0] != "in.json" {
		t.Errorf("positional = %v, want [in.json]", positional)
	}
}
