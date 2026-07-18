package portrait

import (
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

func TestWindowKey(t *testing.T) {
	tm := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	if got := WindowKey(tm, "month"); got != "2026-06" {
		t.Fatalf("month key = %q, want 2026-06", got)
	}
	if got := WindowKey(tm, "week"); len(got) != 8 || got[4] != '-' || got[5] != 'W' {
		t.Fatalf("week key = %q, want form 2026-Wnn", got)
	}
}

func TestReduceWindowsBucketsByMonth(t *testing.T) {
	s6, s7 := 0.4, 0.8
	chunks := []Chunk{
		{Index: 0, Messages: []dialogue.Message{{ID: 1, Date: time.Date(2026, 6, 10, 0, 0, 0, 0, time.UTC)}}},
		{Index: 1, Messages: []dialogue.Message{{ID: 2, Date: time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC)}}},
	}
	frags := []Fragment{
		{ChunkIndex: 0, Target: "u", Person: PersonPortrait{BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: &s6, Confidence: ConfMedium, Evidence: []int{1}}}}},
		{ChunkIndex: 1, Target: "u", Person: PersonPortrait{BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: &s7, Confidence: ConfHigh, Evidence: []int{2}}}}},
	}
	tl := ReduceWindows("u", frags, chunks, "month")
	if len(tl) != 2 {
		t.Fatalf("windows = %d, want 2", len(tl))
	}
	if tl[0].Window != "2026-06" || tl[1].Window != "2026-07" {
		t.Fatalf("windows = %q,%q", tl[0].Window, tl[1].Window)
	}
	o0 := tl[0].Person.BigFive.Openness
	o1 := tl[1].Person.BigFive.Openness
	if o0.Score == nil || *o0.Score != 0.4 {
		t.Fatalf("june openness = %v, want 0.4", o0.Score)
	}
	if o1.Score == nil || *o1.Score != 0.8 {
		t.Fatalf("july openness = %v, want 0.8", o1.Score)
	}
	if tl[0].Messages != 1 {
		t.Fatalf("june messages = %d, want 1", tl[0].Messages)
	}
}
