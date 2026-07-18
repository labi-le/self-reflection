package portrait

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"tg2llm/internal/domain/dialogue"
)

// Segment groups time-sorted messages into chunks. A gap greater than sessionGap
// starts a new session; any session whose rendered text exceeds chunkChars is
// hard-split at message boundaries so no chunk's Text exceeds the budget (a lone
// oversized message is unavoidably its own chunk). Chunk indices run 0..n-1.
func Segment(msgs []dialogue.Message, sessionGap time.Duration, chunkChars int) []Chunk {
	chunks := []Chunk{}
	if len(msgs) == 0 {
		return chunks
	}

	sorted := make([]dialogue.Message, len(msgs))
	copy(sorted, msgs)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Date.Equal(sorted[j].Date) {
			return sorted[i].ID < sorted[j].ID
		}
		return sorted[i].Date.Before(sorted[j].Date)
	})

	var cur []dialogue.Message
	curLen := 0
	flush := func() {
		if len(cur) == 0 {
			return
		}
		chunks = append(chunks, newChunk(len(chunks), cur))
		cur = nil
		curLen = 0
	}

	for i, m := range sorted {
		if i > 0 && m.Date.Sub(sorted[i-1].Date) > sessionGap {
			flush()
		}
		lineLen := len([]rune(renderLine(m)))
		add := lineLen
		if len(cur) > 0 {
			add++ // newline separator
		}
		if chunkChars > 0 && len(cur) > 0 && curLen+add > chunkChars {
			flush()
			add = lineLen
		}
		cur = append(cur, m)
		curLen += add
	}
	flush()
	return chunks
}

func newChunk(index int, msgs []dialogue.Message) Chunk {
	members := make(map[int]bool, len(msgs))
	lines := make([]string, len(msgs))
	batch := make([]dialogue.Message, len(msgs))
	copy(batch, msgs)
	for i, m := range batch {
		members[m.ID] = true
		lines[i] = renderLine(m)
	}
	return Chunk{
		Index:     index,
		Messages:  batch,
		MemberIDs: members,
		Text:      strings.Join(lines, "\n"),
	}
}

func renderLine(m dialogue.Message) string {
	if m.FromID == "" {
		return fmt.Sprintf("[%d] %s: %s", m.ID, m.Sender, dialogue.CleanText(m))
	}
	return fmt.Sprintf("[%d] %s (%s): %s", m.ID, m.FromID, m.Sender, dialogue.CleanText(m))
}
