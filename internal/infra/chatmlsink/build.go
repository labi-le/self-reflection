package chatmlsink

import "tg2llm/internal/domain/dialogue"

func (s *Sink) buildExamples(msgs []dialogue.Message) []example {
	var out []example
	for _, session := range s.segmentSessions(msgs) {
		out = append(out, s.windowSession(session)...)
	}
	return out
}

func (s *Sink) segmentSessions(msgs []dialogue.Message) [][]dialogue.Message {
	if len(msgs) == 0 {
		return nil
	}
	var sessions [][]dialogue.Message
	cur := []dialogue.Message{msgs[0]}
	for i := 1; i < len(msgs); i++ {
		gap := msgs[i].Date.Sub(msgs[i-1].Date)
		if s.opts.SessionGap > 0 && gap > s.opts.SessionGap {
			sessions = append(sessions, cur)
			cur = nil
		}
		cur = append(cur, msgs[i])
	}
	sessions = append(sessions, cur)
	return sessions
}

// windowSession merges the session into strictly alternating turns (bursts joined)
// and emits one example per owner turn, using the turns preceding it as context.
func (s *Sink) windowSession(session []dialogue.Message) []example {
	turns := s.mergeBursts(session)
	var out []example
	for i := range turns {
		if turns[i].Role != roleAssistant {
			continue
		}
		if ex, ok := s.buildExample(turns[:i], turns[i]); ok {
			out = append(out, ex)
		}
	}
	return out
}

// buildExample assembles context + target, repairs the alternation, bounds it by
// CutoffLen, prepends the optional persona system turn, and keeps the result only
// if it satisfies the locked training contract.
func (s *Sink) buildExample(preceding []turn, target turn) (example, bool) {
	ctx := s.contextTurns(preceding)
	body := make([]turn, 0, len(ctx)+1)
	body = append(body, ctx...)
	body = append(body, target)

	body = s.repairAlternation(body)
	body = s.fitCutoff(body)

	turns := s.withSystem(body)
	if !validAlternation(turns) {
		return example{}, false
	}
	return example{Messages: turns}, true
}

// contextTurns returns the last CtxTurns turns preceding a target (already
// burst-merged, so each is one alternating turn). CtxTurns <= 0 yields no context.
func (s *Sink) contextTurns(preceding []turn) []turn {
	limit := s.opts.CtxTurns
	if limit <= 0 || len(preceding) == 0 {
		return nil
	}
	start := len(preceding) - limit
	if start < 0 {
		start = 0
	}
	return preceding[start:]
}

func (s *Sink) role(m dialogue.Message) string {
	if m.FromID == s.opts.OwnerFromID {
		return roleAssistant
	}
	return roleUser
}

// validAlternation gates each example against the locked finetune contract:
// optional leading system, then strict user↔assistant ending in assistant.
func validAlternation(turns []turn) bool {
	i := 0
	if len(turns) > 0 && turns[0].Role == roleSystem {
		i = 1
	}
	body := turns[i:]
	if len(body) < 2 {
		return false
	}
	if body[len(body)-1].Role != roleAssistant {
		return false
	}
	for j, t := range body {
		if t.Content == "" {
			return false
		}
		want := roleUser
		if j%2 == 1 {
			want = roleAssistant
		}
		if t.Role != want {
			return false
		}
	}
	return true
}
