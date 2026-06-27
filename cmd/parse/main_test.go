package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunEndToEnd(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "export.json")
	output := filepath.Join(dir, "out.txt")
	payload := `{"messages":[
		{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Alice","text":"hello"},
		{"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Bob","text":"world"}
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
	s := string(got)
	if !strings.Contains(s, "[1] [10:00:00] Alice: hello") {
		t.Errorf("output missing first message:\n%s", s)
	}
	if !strings.Contains(s, "=== 2026-06-19 ===") {
		t.Errorf("output missing day header:\n%s", s)
	}
}

func TestRunMissingInput(t *testing.T) {
	if err := run([]string{filepath.Join(t.TempDir(), "nope.json")}); err == nil {
		t.Fatal("run with missing input: err = nil, want error")
	}
}

func TestRunDefaultOutput(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	input := filepath.Join(dir, "export.json")
	if err := os.WriteFile(input, []byte(`[{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"A","text":"hi"}]`), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := run([]string{input}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, outputTxt)); err != nil {
		t.Fatalf("default output %q not created: %v", outputTxt, err)
	}
}

func TestAbspath(t *testing.T) {
	if got := abspath(""); got != "" {
		t.Errorf("abspath(\"\") = %q, want empty", got)
	}
	if got := abspath("x.txt"); !filepath.IsAbs(got) {
		t.Errorf("abspath(\"x.txt\") = %q, want absolute", got)
	}
}

func TestInitLogger(t *testing.T) {
	infoLogger := initLogger(false)
	infoLogger.Info().Msg("info-mode")
	traceLogger := initLogger(true)
	traceLogger.Trace().Msg("trace-mode")
}
