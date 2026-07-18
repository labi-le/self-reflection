package portrait

import (
	"sort"
	"strings"
	"time"
	"unicode"

	"tg2llm/internal/domain/dialogue"
)

// stopwords is a small RU/EN function-word set stripped before top-term ranking,
// so frequency lists surface content words rather than grammatical glue.
var stopwords = map[string]bool{
	// RU
	"и": true, "в": true, "во": true, "не": true, "что": true, "он": true,
	"на": true, "я": true, "с": true, "со": true, "как": true, "а": true,
	"то": true, "все": true, "она": true, "так": true, "его": true, "но": true,
	"да": true, "ты": true, "к": true, "у": true, "же": true, "вы": true,
	"за": true, "бы": true, "по": true, "ее": true, "мне": true, "было": true,
	"вот": true, "от": true, "меня": true, "еще": true, "нет": true, "о": true,
	"из": true, "ему": true, "теперь": true, "они": true, "мы": true, "это": true,
	"эт": true, "или": true, "ни": true, "быть": true, "был": true, "него": true,
	// EN
	"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
	"is": true, "are": true, "was": true, "were": true, "be": true, "to": true,
	"of": true, "in": true, "on": true, "for": true, "it": true, "this": true,
	"that": true, "with": true, "as": true, "at": true, "by": true, "i": true,
	"you": true, "he": true, "she": true, "we": true, "they": true, "so": true,
	"do": true, "not": true, "no": true, "yes": true, "me": true, "my": true,
	"your": true, "its": true, "if": true, "then": true, "than": true,
}

// ComputeStats derives the exact, LLM-free statistics block from a message slice.
// Pairwise is populated only when exactly two distinct FromIDs are present.
func ComputeStats(msgs []dialogue.Message, target string, sessionGap time.Duration) Stats {
	stats := Stats{PerSender: map[string]SenderStats{}}
	if len(msgs) == 0 {
		return stats
	}

	sorted := make([]dialogue.Message, len(msgs))
	copy(sorted, msgs)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Date.Equal(sorted[j].Date) {
			return sorted[i].ID < sorted[j].ID
		}
		return sorted[i].Date.Before(sorted[j].Date)
	})

	type acc struct {
		display   string
		msgLens   []int
		chars     int
		questions int
		stickers  int
		hours     [24]int
		weekdays  [7]int
		emoji     map[string]int
		terms     map[string]int
		latencies []int
		monologue int
		inits     int
		enders    int
	}
	accs := map[string]*acc{}
	get := func(id, display string) *acc {
		a := accs[id]
		if a == nil {
			a = &acc{emoji: map[string]int{}, terms: map[string]int{}}
			accs[id] = a
		}
		if display != "" {
			a.display = display
		}
		return a
	}

	// Session boundaries over the time-sorted stream.
	for i, m := range sorted {
		a := get(m.FromID, m.Sender)
		text := dialogue.CleanText(m)
		a.msgLens = append(a.msgLens, len([]rune(text)))
		a.chars += len([]rune(text))
		if strings.Contains(text, "?") {
			a.questions++
		}
		a.hours[m.Date.Hour()]++
		a.weekdays[int(m.Date.Weekday())]++
		if m.MediaType == "sticker" {
			a.stickers++
		}
		for _, r := range text {
			if isEmoji(r) {
				a.emoji[string(r)]++
			}
		}
		for _, term := range topTerms(text) {
			a.terms[term]++
		}

		newSession := i == 0 || m.Date.Sub(sorted[i-1].Date) > sessionGap
		if newSession {
			a.inits++
		}
		if i == len(sorted)-1 || sorted[i+1].Date.Sub(m.Date) > sessionGap {
			a.enders++
		}
		if i > 0 {
			prev := sorted[i-1]
			switch {
			case prev.FromID == m.FromID:
				a.monologue++
			case !newSession:
				a.latencies = append(a.latencies, int(m.Date.Sub(prev.Date).Seconds()))
			}
		}
	}

	for id, a := range accs {
		ss := SenderStats{
			DisplayName:            a.display,
			Messages:               len(a.msgLens),
			Chars:                  a.chars,
			MedianMsgLen:           medianInt(a.msgLens),
			ResponseLatencyMedianS: medianInt(a.latencies),
			Initiations:            a.inits,
			ConversationEnders:     a.enders,
			ActiveHoursHistogram:   a.hours,
			WeekdayHistogram:       a.weekdays,
			StickerCount:           a.stickers,
			EmojiTop:               rankEmoji(a.emoji),
			TopTerms:               rankTerms(a.terms),
		}
		if ss.Messages > 0 {
			ss.AvgMsgLen = float64(a.chars) / float64(ss.Messages)
			ss.QuestionsRatio = float64(a.questions) / float64(ss.Messages)
			ss.MonologueRatio = float64(a.monologue) / float64(ss.Messages)
		}
		stats.PerSender[id] = ss
	}

	if pw := computePairwise(sorted, target, sessionGap); pw != nil {
		stats.Pairwise = pw
	}
	return stats
}

func computePairwise(sorted []dialogue.Message, target string, sessionGap time.Duration) *Pairwise {
	ids := map[string]bool{}
	for _, m := range sorted {
		ids[m.FromID] = true
	}
	if len(ids) != 2 {
		return nil
	}

	var targetInits, otherInits, switches, replyGapSum, replyGaps int
	var gaps []SilenceGap
	for i, m := range sorted {
		newSession := i == 0 || m.Date.Sub(sorted[i-1].Date) > sessionGap
		if newSession {
			if m.FromID == target {
				targetInits++
			} else {
				otherInits++
			}
		}
		if i > 0 {
			prev := sorted[i-1]
			gap := m.Date.Sub(prev.Date)
			if gap > sessionGap {
				gaps = append(gaps, SilenceGap{From: prev.Date, To: m.Date, Seconds: int(gap.Seconds())})
			}
			if prev.FromID != m.FromID {
				switches++
				if !newSession {
					replyGapSum += int(gap.Seconds())
					replyGaps++
				}
			}
		}
	}

	pw := &Pairwise{SilenceGaps: gaps}
	if totalInits := targetInits + otherInits; totalInits > 0 {
		pw.InitiativeBalance = float64(targetInits-otherInits) / float64(totalInits)
	}
	if len(sorted) > 1 {
		pw.Reciprocity = float64(switches) / float64(len(sorted)-1)
	}
	if replyGaps > 0 {
		pw.AvgReplyGapS = replyGapSum / replyGaps
	}
	return pw
}

func medianInt(v []int) int {
	if len(v) == 0 {
		return 0
	}
	s := make([]int, len(v))
	copy(s, v)
	sort.Ints(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func topTerms(text string) []string {
	var out []string
	for _, field := range strings.Fields(strings.ToLower(text)) {
		term := strings.TrimFunc(field, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		})
		if len([]rune(term)) < 2 || stopwords[term] {
			continue
		}
		out = append(out, term)
	}
	return out
}

func rankTerms(counts map[string]int) []TermCount {
	out := make([]TermCount, 0, len(counts))
	for term, c := range counts {
		out = append(out, TermCount{Term: term, Count: c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Term < out[j].Term
	})
	if len(out) > 20 {
		out = out[:20]
	}
	return out
}

func rankEmoji(counts map[string]int) []EmojiCount {
	out := make([]EmojiCount, 0, len(counts))
	for e, c := range counts {
		out = append(out, EmojiCount{Emoji: e, Count: c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Emoji < out[j].Emoji
	})
	if len(out) > 20 {
		out = out[:20]
	}
	return out
}

// isEmoji reports whether a rune falls in a pictographic Unicode block. It is a
// deterministic range scan, not a grapheme-cluster segmenter: modifiers and
// variation selectors are ignored so a base emoji counts once.
func isEmoji(r rune) bool {
	switch {
	case r >= 0x1F300 && r <= 0x1FAFF:
		return true
	case r >= 0x1F000 && r <= 0x1F0FF:
		return true
	case r >= 0x1F1E6 && r <= 0x1F1FF:
		return true
	case r >= 0x2600 && r <= 0x27BF:
		return true
	case r >= 0x2300 && r <= 0x23FF:
		return true
	case r >= 0x2B00 && r <= 0x2BFF:
		return true
	default:
		return false
	}
}
