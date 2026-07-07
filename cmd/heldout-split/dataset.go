package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// example is one dataset line plus the metadata needed to split it.
type example struct {
	index        int    // stable identity: 0-based position in the input
	raw          string // exact JSON line, written back verbatim
	chatID       string // stratification key part 1
	bucket       string // stratification key part 2 (coarse time bucket)
	hasAssistant bool   // eligible for held-out iff true
}

// parseExamples reads the JSONL dataset. A malformed JSON line, or a line
// missing the "messages" array, is a hard error (never silently skipped).
// Blank lines are ignored.
func parseExamples(r io.Reader) ([]example, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 64*1024*1024) // tolerate long lines

	var out []example
	lineNo := 0
	for sc.Scan() {
		lineNo++
		raw := sc.Text()
		if strings.TrimSpace(raw) == "" {
			continue
		}
		ex, err := parseLine(lineNo, raw)
		if err != nil {
			return nil, err
		}
		ex.index = len(out)
		out = append(out, ex)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("reading input: %w", err)
	}
	return out, nil
}

func parseLine(lineNo int, raw string) (example, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &top); err != nil {
		return example{}, fmt.Errorf("line %d: malformed JSON: %w", lineNo, err)
	}
	msgsRaw, ok := top["messages"]
	if !ok {
		return example{}, fmt.Errorf("line %d: missing \"messages\" array", lineNo)
	}
	var msgs []map[string]json.RawMessage
	if err := json.Unmarshal(msgsRaw, &msgs); err != nil {
		return example{}, fmt.Errorf("line %d: \"messages\" is not an array of objects: %w", lineNo, err)
	}

	ex := example{raw: raw}
	for _, m := range msgs {
		if jsonScalar(m["role"]) == "assistant" {
			ex.hasAssistant = true
			break
		}
	}
	ex.chatID = extractChatID(top, msgs)
	ex.bucket = extractBucket(top, msgs)
	return ex, nil
}

// jsonScalar renders a JSON string or number field as a string; anything else
// (or absent) yields "".
func jsonScalar(r json.RawMessage) string {
	if len(r) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(r, &s); err == nil {
		return s
	}
	var n json.Number
	if err := json.Unmarshal(r, &n); err == nil {
		return n.String()
	}
	return ""
}

// extractChatID prefers a top-level "chat_id"; otherwise the first message's
// "chat_id"; otherwise "unknown".
func extractChatID(top map[string]json.RawMessage, msgs []map[string]json.RawMessage) string {
	if id := jsonScalar(top["chat_id"]); id != "" {
		return id
	}
	for _, m := range msgs {
		if id := jsonScalar(m["chat_id"]); id != "" {
			return id
		}
	}
	return "unknown"
}

// extractBucket derives a coarse (year-month) time bucket from the first usable
// date/timestamp it finds; "nobucket" when none parse.
func extractBucket(top map[string]json.RawMessage, msgs []map[string]json.RawMessage) string {
	if b := bucketOf(jsonScalar(top["date"])); b != "" {
		return b
	}
	if b := bucketOf(jsonScalar(top["timestamp"])); b != "" {
		return b
	}
	for _, m := range msgs {
		if b := bucketOf(jsonScalar(m["date"])); b != "" {
			return b
		}
	}
	return "nobucket"
}

func bucketOf(s string) string {
	if s == "" {
		return ""
	}
	for _, layout := range []string{
		"2006-01-02T15:04:05", "2006-01-02 15:04:05", time.RFC3339,
		"2006-01-02T15:04", "2006-01-02 15:04", "2006-01-02",
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC().Format("2006-01")
		}
	}
	return ""
}
