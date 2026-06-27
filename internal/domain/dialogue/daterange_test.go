package dialogue

import (
	"testing"
	"time"
)

func TestParseBound(t *testing.T) {
	cases := []struct {
		value string
		isEnd bool
		want  time.Time
	}{
		{"2026-06-19T10:00:00", false, time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)},
		{"2026-06-19 10:00:00", false, time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)},
		{"2026-06-19T10:00", false, time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC)},
		{"2026-06-19", false, time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)},
		{"2026-06-19", true, time.Date(2026, 6, 19, 23, 59, 59, 0, time.UTC)},
		{"  2026-06-19  ", false, time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)},
	}
	for _, c := range cases {
		got, err := ParseBound(c.value, c.isEnd)
		if err != nil {
			t.Errorf("ParseBound(%q, %v) error: %v", c.value, c.isEnd, err)
			continue
		}
		if !got.Equal(c.want) {
			t.Errorf("ParseBound(%q, %v) = %v, want %v", c.value, c.isEnd, got, c.want)
		}
	}

	for _, bad := range []string{"", "not-a-date", "2026/06/19"} {
		if _, err := ParseBound(bad, false); err == nil {
			t.Errorf("ParseBound(%q) expected error, got nil", bad)
		}
	}
}

func TestDateRangeContains(t *testing.T) {
	from := time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 6, 20, 23, 59, 59, 0, time.UTC)
	inside := time.Date(2026, 6, 19, 12, 0, 0, 0, time.UTC)
	before := time.Date(2026, 6, 18, 12, 0, 0, 0, time.UTC)
	after := time.Date(2026, 6, 21, 12, 0, 0, 0, time.UTC)

	full := DateRange{From: &from, To: &to}
	if !full.Contains(inside) {
		t.Error("Contains(inside) = false, want true")
	}
	if full.Contains(before) {
		t.Error("Contains(before) = true, want false")
	}
	if full.Contains(after) {
		t.Error("Contains(after) = true, want false")
	}

	open := DateRange{}
	if !open.Contains(before) || !open.Contains(after) {
		t.Error("open DateRange should contain everything")
	}

	onlyFrom := DateRange{From: &from}
	if onlyFrom.Contains(before) {
		t.Error("onlyFrom.Contains(before) = true, want false")
	}
	if !onlyFrom.Contains(after) {
		t.Error("onlyFrom.Contains(after) = false, want true")
	}
}
