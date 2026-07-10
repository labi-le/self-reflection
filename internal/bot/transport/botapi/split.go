package botapi

import "strings"

// WithWordSplit enables probabilistic word-level splitting. prob is the break
// probability applied at each inter-word gap (<=0 disables it: one message per
// U+2063 burst segment, the prior behaviour). maxMsgs caps fragmentation — a
// segment that would split into more than maxMsgs messages is sent whole instead
// (0 = uncapped) — so a long reply never becomes a message flood.
func WithWordSplit(prob float64, maxMsgs int) Option {
	return func(c *Client) {
		c.splitWordProb = prob
		c.splitMaxMsgs = maxMsgs
	}
}

// withRandFloat swaps the [0,1) randomness seam so split decisions are
// deterministic and observable in tests instead of random.
func withRandFloat(fn func() float64) Option {
	return func(c *Client) {
		if fn != nil {
			c.randFloat = fn
		}
	}
}

// segments is the outbound reply's delivery plan: the U+2063 burst split first,
// then probabilistic word-chunking of each resulting segment.
func (c *Client) segments(text string) []string {
	var out []string
	for _, seg := range splitBurst(text) {
		out = append(out, c.chunkWords(seg)...)
	}
	return out
}

// chunkWords breaks seg at inter-word gaps, each an independent break fired with
// probability splitWordProb, so the segment is sometimes delivered word-by-word
// and sometimes whole — mimicking a person firing off several short messages. A
// single word, a disabled probability, or a result exceeding splitMaxMsgs
// messages yields the segment unsplit.
func (c *Client) chunkWords(seg string) []string {
	if c.splitWordProb <= 0 {
		return []string{seg}
	}
	words := strings.Fields(seg)
	if len(words) <= 1 {
		return []string{seg}
	}
	var chunks []string
	start := 0
	for i := 0; i < len(words)-1; i++ { // never break after the final word
		if c.randFloat() < c.splitWordProb {
			chunks = append(chunks, strings.Join(words[start:i+1], " "))
			start = i + 1
		}
	}
	chunks = append(chunks, strings.Join(words[start:], " "))
	if c.splitMaxMsgs > 0 && len(chunks) > c.splitMaxMsgs {
		return []string{seg} // too fragmented — deliver as one message
	}
	return chunks
}
