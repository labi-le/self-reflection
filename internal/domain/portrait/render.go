package portrait

import (
	"sort"
	"strconv"
	"strings"
)

// RenderMarkdown renders a portrait as a human-readable report. The first line is
// the disclaimer; NotAssessed trait scores are omitted (never shown as a real
// 0.0); each cited message id is resolved to its quote via p.EvidenceIndex.
func RenderMarkdown(p Portrait) string {
	var sb strings.Builder
	sb.WriteString(p.Meta.Disclaimer)
	sb.WriteString("\n\n# Psychological Portrait\n")
	if p.Meta.Model != "" {
		sb.WriteString("\nModel: " + p.Meta.Model + "\n")
	}

	ids := make([]string, 0, len(p.Portrait))
	for id := range p.Portrait {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		person := p.Portrait[id]
		name := id
		if ss, ok := p.Stats.PerSender[id]; ok && ss.DisplayName != "" {
			name = ss.DisplayName
		}
		sb.WriteString("\n## " + name + " (" + id + ")\n")
		renderBigFive(&sb, person.BigFive, p.EvidenceIndex)
		renderStyle(&sb, person.CommunicationStyle, p.EvidenceIndex)
		renderAccentuations(&sb, person.Accentuations, p.EvidenceIndex)
		renderTopics(&sb, person.RecurringTopics, p.EvidenceIndex)
		renderClaims(&sb, "Values & Motivations", person.ValuesMotivations, p.EvidenceIndex)
		renderEmotional(&sb, person.EmotionalPatterns, p.EvidenceIndex)
		renderPredictions(&sb, person.WhatToExpect, p.EvidenceIndex)
	}
	renderDynamics(&sb, p.Dynamics, p.Stats, p.EvidenceIndex)
	renderTimeline(&sb, p.Timeline)
	return sb.String()
}

var oceanDims = []struct {
	title string
	pick  func(BigFive) TraitScore
}{
	{"Openness", func(b BigFive) TraitScore { return b.Openness }},
	{"Conscientiousness", func(b BigFive) TraitScore { return b.Conscientiousness }},
	{"Extraversion", func(b BigFive) TraitScore { return b.Extraversion }},
	{"Agreeableness", func(b BigFive) TraitScore { return b.Agreeableness }},
	{"Neuroticism", func(b BigFive) TraitScore { return b.Neuroticism }},
}

func renderBigFive(sb *strings.Builder, b BigFive, idx map[string]EvidenceRef) {
	var shown []struct {
		title string
		ts    TraitScore
	}
	for _, d := range oceanDims {
		if ts := d.pick(b); ts.Assessed && ts.Score != nil {
			shown = append(shown, struct {
				title string
				ts    TraitScore
			}{d.title, ts})
		}
	}
	if len(shown) == 0 {
		return
	}
	sb.WriteString("\n### Big Five (OCEAN)\n\n")
	for _, s := range shown {
		sb.WriteString("- **" + s.title + "**: " + fmtScore(*s.ts.Score) + " (" + string(s.ts.Confidence) + ")\n")
		renderEvidence(sb, s.ts.Evidence, idx)
	}
}

func renderStyle(sb *strings.Builder, cs CommunicationStyle, idx map[string]EvidenceRef) {
	if styleEmpty(cs) {
		return
	}
	sb.WriteString("\n### Communication Style\n\n")
	for _, f := range []struct{ label, val string }{
		{"Tone", cs.Tone},
		{"Formality", cs.Formality},
		{"Directness", cs.Directness},
		{"Humor", cs.Humor},
		{"Emotional Expressiveness", cs.EmotionalExpressiveness},
		{"Conflict Style", cs.ConflictStyle},
	} {
		if f.val == "" {
			continue
		}
		sb.WriteString("- **" + f.label + "**: " + f.val + " (" + string(cs.Confidence) + ")\n")
	}
	renderEvidence(sb, cs.Evidence, idx)
}

func renderTopics(sb *strings.Builder, topics []Topic, idx map[string]EvidenceRef) {
	if len(topics) == 0 {
		return
	}
	sb.WriteString("\n### Recurring Topics\n\n")
	for _, t := range topics {
		sb.WriteString("- **" + t.Topic + "** (weight " + fmtScore(t.Weight) + ", " + string(t.Confidence) + ")\n")
		renderEvidence(sb, t.Evidence, idx)
	}
}

func renderClaims(sb *strings.Builder, heading string, claims []Claim, idx map[string]EvidenceRef) {
	if len(claims) == 0 {
		return
	}
	sb.WriteString("\n### " + heading + "\n\n")
	for _, c := range claims {
		sb.WriteString("- " + c.Claim + " (" + string(c.Confidence) + ")\n")
		renderEvidence(sb, c.Evidence, idx)
	}
}

func renderEvidence(sb *strings.Builder, ids []int, idx map[string]EvidenceRef) {
	for _, id := range ids {
		key := strconv.Itoa(id)
		if ref, ok := idx[key]; ok {
			sb.WriteString("  - [" + key + "] " + ref.FromID + ": \"" + ref.Text + "\"\n")
		} else {
			sb.WriteString("  - [" + key + "] (unresolved)\n")
		}
	}
}

func fmtScore(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) }

func renderAccentuations(sb *strings.Builder, accs []Accentuation, idx map[string]EvidenceRef) {
	if len(accs) == 0 {
		return
	}
	sb.WriteString("\n### Accentuations\n\n")
	for _, a := range accs {
		sb.WriteString("- **" + a.Type + "** (intensity " + fmtScore(a.Intensity) + ", " + string(a.Confidence) + ")\n")
		renderEvidence(sb, a.Evidence, idx)
	}
}

func renderEmotional(sb *strings.Builder, patterns []EmotionalPattern, idx map[string]EvidenceRef) {
	if len(patterns) == 0 {
		return
	}
	sb.WriteString("\n### Emotional Patterns\n\n")
	for _, e := range patterns {
		line := "- **" + e.Pattern + "**"
		if len(e.Triggers) > 0 {
			line += " (triggers: " + strings.Join(e.Triggers, ", ") + ")"
		}
		sb.WriteString(line + " (" + string(e.Confidence) + ")\n")
		renderEvidence(sb, e.Evidence, idx)
	}
}

func renderPredictions(sb *strings.Builder, preds []Prediction, idx map[string]EvidenceRef) {
	if len(preds) == 0 {
		return
	}
	sb.WriteString("\n### What To Expect\n\n")
	for _, p := range preds {
		sb.WriteString("- " + p.Prediction + " (" + string(p.Confidence) + ")\n")
		renderEvidence(sb, p.Evidence, idx)
	}
}

func renderDynamics(sb *strings.Builder, d *Dynamics, stats Stats, idx map[string]EvidenceRef) {
	if d == nil {
		return
	}
	sb.WriteString("\n## Relationship Dynamics\n")
	if len(d.KarpmanRoles) > 0 {
		who := make([]string, 0, len(d.KarpmanRoles))
		for k := range d.KarpmanRoles {
			who = append(who, k)
		}
		sort.Strings(who)
		sb.WriteString("\n### Karpman Roles\n\n")
		for _, w := range who {
			name := w
			if ss, ok := stats.PerSender[w]; ok && ss.DisplayName != "" {
				name = ss.DisplayName
			}
			sb.WriteString("**" + name + " (" + w + ")**\n\n")
			for _, r := range d.KarpmanRoles[w] {
				sb.WriteString("- **" + r.Role + "** (tendency " + fmtScore(r.Tendency) + ", " + string(r.Confidence) + ")\n")
				renderEvidence(sb, r.Evidence, idx)
			}
		}
	}
	if d.PowerBalance.Claim != "" {
		sb.WriteString("\n### Power Balance\n\n")
		sb.WriteString("- " + d.PowerBalance.Claim + " (" + string(d.PowerBalance.Confidence) + ")\n")
		renderEvidence(sb, d.PowerBalance.Evidence, idx)
	}
	if len(d.RelationshipArc) > 0 {
		sb.WriteString("\n### Relationship Arc\n\n")
		for _, w := range d.RelationshipArc {
			sb.WriteString("- **" + w.Window + "**: " + w.Summary + " (" + string(w.Confidence) + ")\n")
			renderEvidence(sb, w.Evidence, idx)
		}
	}
}

// renderTimeline prints a compact per-window OCEAN table with deltas vs the
// previous window. It is evidence-light by design (scores + counts, no quote
// dumps) so the report stays readable across many windows.
func renderTimeline(sb *strings.Builder, timeline []WindowPortrait) {
	if len(timeline) == 0 {
		return
	}
	sb.WriteString("\n## Dynamics Over Time\n\n")
	sb.WriteString("Per-window OCEAN score (blank = not assessed); Δ vs previous window.\n\n")
	sb.WriteString("| Window | Msgs | O | C | E | A | N |\n")
	sb.WriteString("|---|---|---|---|---|---|---|\n")
	var prev *BigFive
	for i := range timeline {
		w := timeline[i]
		b := w.Person.BigFive
		sb.WriteString("| " + w.Window + " | " + strconv.Itoa(w.Messages))
		dims := oceanScores(b)
		var prevDims []TraitScore
		if prev != nil {
			prevDims = oceanScores(*prev)
		}
		for j, d := range dims {
			cell := "—"
			if d.Assessed && d.Score != nil {
				cell = fmtScore(*d.Score)
				if prevDims != nil && prevDims[j].Assessed && prevDims[j].Score != nil {
					cell += " (" + fmtDelta(*d.Score-*prevDims[j].Score) + ")"
				}
			}
			sb.WriteString(" | " + cell)
		}
		sb.WriteString(" |\n")
		bb := b
		prev = &bb
	}
}

func oceanScores(b BigFive) []TraitScore {
	return []TraitScore{b.Openness, b.Conscientiousness, b.Extraversion, b.Agreeableness, b.Neuroticism}
}

func fmtDelta(d float64) string {
	s := strconv.FormatFloat(d, 'f', 2, 64)
	if d > 0 {
		return "+" + s
	}
	return s
}
