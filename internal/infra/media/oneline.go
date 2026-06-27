package media

import "strings"

// OneLine splits text into lines, strips each, drops empty ones, and joins with a single space.
func OneLine(raw string) string {
	var parts []string
	for _, ln := range strings.Split(raw, "\n") {
		if trimmed := strings.TrimSpace(ln); trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	return strings.Join(parts, " ")
}
