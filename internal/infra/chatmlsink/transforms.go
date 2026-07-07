package chatmlsink

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"tg2llm/internal/domain/dialogue"
)

const chatTypeSavedMessages = "saved_messages"

// mergeBursts collapses each run of consecutive same-sender messages into one turn
// whose segments are joined by Options.BurstDelim. The join preserves the burst's
// rapid-fire style (the bot transport later splits on the same delimiter) and makes
// the returned turns strictly alternate: adjacent turns always differ in role, so
// windowing them yields contract-valid examples.
func (s *Sink) mergeBursts(msgs []dialogue.Message) []turn {
	var turns []turn
	for _, m := range msgs {
		text := strings.TrimSpace(m.ExtractedText)
		if text == "" {
			continue
		}
		r := s.role(m)
		if n := len(turns); n > 0 && turns[n-1].Role == r {
			turns[n-1].Content += s.opts.BurstDelim + text
			continue
		}
		turns = append(turns, turn{Role: r, Content: text})
	}
	return turns
}

// repairAlternation makes a window body satisfy the training contract: it collapses
// any consecutive same-role turns (joining segments with the delimiter, never
// dropping one) and drops a leading assistant turn that has no user to answer. The
// result strictly alternates; a body that cannot reach >=2 alternating turns ending
// in assistant is rejected later by validAlternation.
func (s *Sink) repairAlternation(body []turn) []turn {
	var out []turn
	for _, t := range body {
		if n := len(out); n > 0 && out[n-1].Role == t.Role {
			out[n-1].Content += s.opts.BurstDelim + t.Content
			continue
		}
		out = append(out, t)
	}
	if len(out) > 0 && out[0].Role == roleAssistant {
		out = out[1:]
	}
	return out
}

// fitCutoff trims the oldest context turn-pairs until the example's approximate
// token count fits CutoffLen. Turns are removed two at a time (one user + its
// assistant reply) so the body keeps starting with user and ending with the
// assistant target. The final user->assistant pair is never trimmed: even a target
// that alone exceeds CutoffLen is emitted intact, because it is the supervised
// signal and must never be shortened or dropped.
func (s *Sink) fitCutoff(body []turn) []turn {
	max := s.opts.CutoffLen
	if max <= 0 {
		return body
	}
	sysTok := approxTokens(s.systemTurns())
	for len(body) > 2 && sysTok+approxTokens(body) > max {
		body = body[2:]
	}
	return body
}

func (s *Sink) systemTurns() []turn {
	if s.opts.PersonaSystem == "" {
		return nil
	}
	return []turn{{Role: roleSystem, Content: s.opts.PersonaSystem}}
}

func (s *Sink) withSystem(body []turn) []turn {
	sys := s.systemTurns()
	if len(sys) == 0 {
		return body
	}
	return append(sys, body...)
}

// approxTokens is a deliberately rough estimate — one token per 4 runes of content
// plus a small per-turn overhead for chat-template role markers. It is NOT a real
// tokenizer; it only bounds context growth against CutoffLen and shapes the length
// histogram.
func approxTokens(turns []turn) int {
	n := 0
	for _, t := range turns {
		n += utf8.RuneCountInString(t.Content)/4 + 3
	}
	return n
}

// capExamples keeps at most max examples, the chronological first ones (max <= 0 =
// unlimited). Deterministic: identical input yields the identical prefix on every
// run, so a couple of high-volume chats cannot dominate the corpus.
func capExamples(exs []example, max int) []example {
	if max > 0 && len(exs) > max {
		return exs[:max]
	}
	return exs
}

// chatExamples materialises one buffered chat into capped training examples. A
// saved_messages chat yields none: both participants are the owner, so every window
// is degenerate (no genuine user turn) and would only teach the model to talk to
// itself.
func (s *Sink) chatExamples(cb *chatBuffer) []example {
	if cb.chatType == chatTypeSavedMessages {
		return nil
	}
	return capExamples(s.buildExamples(cb.msgs), s.opts.MaxPerChat)
}

// logLengthHistogram emits an approximate token-length distribution over the emitted
// examples so an operator can see whether CutoffLen is biting.
func (s *Sink) logLengthHistogram(exs []example) {
	if len(exs) == 0 {
		return
	}
	lens := make([]int, len(exs))
	for i, ex := range exs {
		lens[i] = approxTokens(ex.Messages)
	}
	sort.Ints(lens)

	edges := []int{128, 256, 512, 1024}
	buckets := make([]int, len(edges)+1)
	over := 0
	for _, v := range lens {
		buckets[bucketOf(v, edges)]++
		if s.opts.CutoffLen > 0 && v > s.opts.CutoffLen {
			over++
		}
	}
	s.opts.Logger.Info().
		Int("examples", len(exs)).
		Int("approx_tokens_p50", lens[len(lens)/2]).
		Int("approx_tokens_p95", lens[(len(lens)*95)/100]).
		Int("approx_tokens_max", lens[len(lens)-1]).
		Int("over_cutoff", over).
		Int("cutoff_len", s.opts.CutoffLen).
		Str("buckets", histBuckets(edges, buckets)).
		Str("token_estimate", "approximate: runes/4 + 3 per turn, not a real tokenizer").
		Msg("chatml dataset approx length histogram")
}

func bucketOf(v int, edges []int) int {
	for i, e := range edges {
		if v < e {
			return i
		}
	}
	return len(edges)
}

func histBuckets(edges, counts []int) string {
	var b strings.Builder
	lo := 0
	for i, c := range counts {
		if i > 0 {
			b.WriteByte(' ')
		}
		if i < len(edges) {
			fmt.Fprintf(&b, "%d-%d:%d", lo, edges[i], c)
			lo = edges[i]
			continue
		}
		fmt.Fprintf(&b, "%d+:%d", lo, c)
	}
	return b.String()
}
