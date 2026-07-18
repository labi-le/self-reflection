package portrait

import (
	"fmt"
	"sort"
	"time"
)

// WindowPortrait is the target's portrait reduced over a single time bucket, so
// the caller can chart how a person changes across months/weeks. It reuses the
// same evidence-gated PersonPortrait as the overall portrait.
type WindowPortrait struct {
	Window   string         `json:"window"`
	Messages int            `json:"messages"`
	Person   PersonPortrait `json:"person"`
}

// WindowKey buckets a timestamp: "2006-01" for month (default) or "2006-Www"
// (ISO week) for week.
func WindowKey(t time.Time, by string) string {
	if by == "week" {
		y, w := t.ISOWeek()
		return fmt.Sprintf("%d-W%02d", y, w)
	}
	return t.Format("2006-01")
}

// ReduceWindows tags each fragment with its chunk's time window and reduces the
// fragments per window (pure Go — the LLM map runs once, so temporal analysis is
// free). Windows are returned ascending. frags is aligned to chunks by index:
// frags[i] is the fragment for chunks[i].
func ReduceWindows(target string, frags []Fragment, chunks []Chunk, by string) []WindowPortrait {
	groups := map[string][]Fragment{}
	counts := map[string]int{}
	for i, c := range chunks {
		if i >= len(frags) {
			break
		}
		f := frags[i]
		if f.Target != target {
			continue
		}
		key := WindowKey(chunkDate(c), by)
		groups[key] = append(groups[key], f)
		counts[key] += len(c.Messages)
	}

	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	out := make([]WindowPortrait, 0, len(keys))
	for _, k := range keys {
		out = append(out, WindowPortrait{
			Window:   k,
			Messages: counts[k],
			Person:   Reduce(target, groups[k]),
		})
	}
	return out
}

// chunkDate is the window anchor for a chunk: the timestamp of its first message
// (chunks are time-ordered), falling back to zero time for an empty chunk.
func chunkDate(c Chunk) time.Time {
	if len(c.Messages) == 0 {
		return time.Time{}
	}
	return c.Messages[0].Date
}

// OceanTrend is one big-five dimension's score across the timeline, for charting
// or a delta table. Score is nil where the window did not assess that dimension.
type OceanTrend struct {
	Window string   `json:"window"`
	Scores []*float64 `json:"scores"` // aligned to OCEAN order
}

// OceanTrends projects the timeline onto a per-window OCEAN score matrix (nil for
// not-assessed), so a renderer or dashboard can show trends and deltas without
// re-walking PersonPortrait.
func OceanTrends(timeline []WindowPortrait) []OceanTrend {
	out := make([]OceanTrend, 0, len(timeline))
	for _, w := range timeline {
		b := w.Person.BigFive
		dims := []TraitScore{b.Openness, b.Conscientiousness, b.Extraversion, b.Agreeableness, b.Neuroticism}
		scores := make([]*float64, len(dims))
		for i, d := range dims {
			if d.Assessed {
				scores[i] = d.Score
			}
		}
		out = append(out, OceanTrend{Window: w.Window, Scores: scores})
	}
	return out
}
