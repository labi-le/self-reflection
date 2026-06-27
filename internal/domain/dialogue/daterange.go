package dialogue

import (
	"fmt"
	"strings"
	"time"
)

// DateRange represents an optional time window for filtering messages.
type DateRange struct {
	From *time.Time
	To   *time.Time
}

// Contains reports whether t falls within the range.
func (dr DateRange) Contains(t time.Time) bool {
	if dr.From != nil && t.Before(*dr.From) {
		return false
	}
	if dr.To != nil && t.After(*dr.To) {
		return false
	}
	return true
}

// ParseBound parses a date string like "2026-06-19T10:00:00" or "2026-06-19" into a time.Time.
// When isEnd is true and the format is date-only,
// the time is set to 23:59:59.
func ParseBound(value string, isEnd bool) (time.Time, error) {
	if value == "" {
		return time.Time{}, fmt.Errorf("empty date")
	}
	value = strings.TrimSpace(value)
	formats := []string{
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
		"2006-01-02T15:04",
		"2006-01-02 15:04",
		"2006-01-02",
	}
	for _, layout := range formats {
		t, err := time.Parse(layout, value)
		if err != nil {
			continue
		}
		if layout == "2006-01-02" && isEnd {
			t = time.Date(t.Year(), t.Month(), t.Day(), 23, 59, 59, 0, time.UTC)
		}
		return t, nil
	}
	return time.Time{}, fmt.Errorf("invalid date format: %s", value)
}
