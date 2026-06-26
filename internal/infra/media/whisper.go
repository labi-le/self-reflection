package media

import (
	"crypto/sha1"
	"fmt"
)

// Signature returns the first 10 hex chars of SHA-1 of the parts joined by "|".
func Signature(parts ...string) string {
	raw := ""
	for i, p := range parts {
		if i > 0 {
			raw += "|"
		}
		raw += p
	}
	h := sha1.Sum([]byte(raw))
	return fmt.Sprintf("%x", h)[:10]
}

// OneLine splits text into lines, strips each, drops empty ones, and joins with a single space.
func OneLine(raw string) string {
	lines := splitLines(raw)
	result := ""
	for _, ln := range lines {
		trimmed := trimStr(ln)
		if trimmed == "" {
			continue
		}
		if result != "" {
			result += " "
		}
		result += trimmed
	}
	return result
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	lines = append(lines, s[start:])
	return lines
}

func trimStr(s string) string {
	start, end := 0, len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\r') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\r') {
		end--
	}
	return s[start:end]
}
