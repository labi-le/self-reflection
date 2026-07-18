package portrait

import (
	"sort"
	"strings"
)

const topK = 10

// Reduce merges per-chunk fragments for one target into a single portrait using
// deterministic, evidence-weighted aggregation: numeric scores are means weighted
// by evidence count, list claims are grouped by normalized key and capped at the
// top-K by evidence, and any dimension no fragment assessed stays NotAssessed().
func Reduce(target string, frags []Fragment) PersonPortrait {
	var people []PersonPortrait
	for _, f := range frags {
		if f.Target == target {
			people = append(people, f.Person)
		}
	}

	var out PersonPortrait
	out.BigFive = BigFive{
		Openness:          reduceTrait(collectTrait(people, func(b BigFive) TraitScore { return b.Openness })),
		Conscientiousness: reduceTrait(collectTrait(people, func(b BigFive) TraitScore { return b.Conscientiousness })),
		Extraversion:      reduceTrait(collectTrait(people, func(b BigFive) TraitScore { return b.Extraversion })),
		Agreeableness:     reduceTrait(collectTrait(people, func(b BigFive) TraitScore { return b.Agreeableness })),
		Neuroticism:       reduceTrait(collectTrait(people, func(b BigFive) TraitScore { return b.Neuroticism })),
	}
	out.CommunicationStyle = reduceStyle(people)
	out.RecurringTopics = reduceTopics(people)
	out.ValuesMotivations = reduceClaims(people)
	out.EmotionalPatterns = reduceEmotional(people)
	out.WhatToExpect = reducePredictions(people)
	out.Accentuations = reduceAccentuations(people)
	return out
}

func collectTrait(people []PersonPortrait, pick func(BigFive) TraitScore) []TraitScore {
	var out []TraitScore
	for _, p := range people {
		out = append(out, pick(p.BigFive))
	}
	return out
}

func reduceTrait(scores []TraitScore) TraitScore {
	var sum, sumW, plainSum float64
	var plainN int
	var ev []int
	conf := ConfLow
	assessed := false
	for _, s := range scores {
		if !s.Assessed {
			continue
		}
		assessed = true
		ev = append(ev, s.Evidence...)
		conf = maxConf(conf, s.Confidence)
		if s.Score != nil {
			w := float64(len(s.Evidence))
			sum += *s.Score * w
			sumW += w
			plainSum += *s.Score
			plainN++
		}
	}
	if !assessed {
		return NotAssessed()
	}
	var score float64
	switch {
	case sumW > 0:
		score = sum / sumW
	case plainN > 0:
		score = plainSum / float64(plainN)
	}
	return TraitScore{Assessed: true, Score: new(score), Confidence: conf, Evidence: unionSorted(ev)}
}

func reduceStyle(people []PersonPortrait) CommunicationStyle {
	var best CommunicationStyle
	bestEv := -1
	conf := ConfLow
	var ev []int
	for _, p := range people {
		cs := p.CommunicationStyle
		if styleEmpty(cs) {
			continue
		}
		ev = append(ev, cs.Evidence...)
		conf = maxConf(conf, cs.Confidence)
		if len(cs.Evidence) > bestEv {
			bestEv = len(cs.Evidence)
			best = cs
		}
	}
	if bestEv < 0 {
		return CommunicationStyle{}
	}
	best.Evidence = unionSorted(ev)
	best.Confidence = conf
	return best
}

func styleEmpty(cs CommunicationStyle) bool {
	return cs.Tone == "" && cs.Formality == "" && cs.Directness == "" &&
		cs.Humor == "" && cs.EmotionalExpressiveness == "" && cs.ConflictStyle == "" &&
		len(cs.Evidence) == 0
}

type numAgg struct {
	key      string
	order    int
	sum, w   float64
	plainSum float64
	plainN   int
	ev       []int
	conf     Confidence
}

func (a *numAgg) add(num float64, evidence []int, c Confidence) {
	w := float64(len(evidence))
	a.sum += num * w
	a.w += w
	a.plainSum += num
	a.plainN++
	a.ev = append(a.ev, evidence...)
	a.conf = maxConf(a.conf, c)
}

func (a *numAgg) mean() float64 {
	switch {
	case a.w > 0:
		return a.sum / a.w
	case a.plainN > 0:
		return a.plainSum / float64(a.plainN)
	default:
		return 0
	}
}

func rankNum(groups map[string]*numAgg) []*numAgg {
	out := make([]*numAgg, 0, len(groups))
	for _, g := range groups {
		out = append(out, g)
	}
	sort.Slice(out, func(i, j int) bool {
		if len(out[i].ev) != len(out[j].ev) {
			return len(out[i].ev) > len(out[j].ev)
		}
		if out[i].order != out[j].order {
			return out[i].order < out[j].order
		}
		return out[i].key < out[j].key
	})
	if len(out) > topK {
		out = out[:topK]
	}
	return out
}

func reduceTopics(people []PersonPortrait) []Topic {
	groups := map[string]*numAgg{}
	orig := map[string]string{}
	for _, p := range people {
		for _, t := range p.RecurringTopics {
			k := normKey(t.Topic)
			g := groups[k]
			if g == nil {
				g = &numAgg{key: k, order: len(groups), conf: ConfLow}
				groups[k] = g
				orig[k] = strings.TrimSpace(t.Topic)
			}
			g.add(t.Weight, t.Evidence, t.Confidence)
		}
	}
	var out []Topic
	for _, g := range rankNum(groups) {
		out = append(out, Topic{Topic: orig[g.key], Weight: g.mean(), Confidence: g.conf, Evidence: unionSorted(g.ev)})
	}
	return out
}

func reduceAccentuations(people []PersonPortrait) []Accentuation {
	groups := map[string]*numAgg{}
	orig := map[string]string{}
	for _, p := range people {
		for _, a := range p.Accentuations {
			k := normKey(a.Type)
			g := groups[k]
			if g == nil {
				g = &numAgg{key: k, order: len(groups), conf: ConfLow}
				groups[k] = g
				orig[k] = strings.TrimSpace(a.Type)
			}
			g.add(a.Intensity, a.Evidence, a.Confidence)
		}
	}
	var out []Accentuation
	for _, g := range rankNum(groups) {
		out = append(out, Accentuation{Type: orig[g.key], Intensity: g.mean(), Confidence: g.conf, Evidence: unionSorted(g.ev)})
	}
	return out
}

type claimAgg struct {
	key   string
	order int
	ev    []int
	conf  Confidence
}

func rankClaims(groups map[string]*claimAgg) []*claimAgg {
	out := make([]*claimAgg, 0, len(groups))
	for _, g := range groups {
		out = append(out, g)
	}
	sort.Slice(out, func(i, j int) bool {
		if len(out[i].ev) != len(out[j].ev) {
			return len(out[i].ev) > len(out[j].ev)
		}
		if out[i].order != out[j].order {
			return out[i].order < out[j].order
		}
		return out[i].key < out[j].key
	})
	if len(out) > topK {
		out = out[:topK]
	}
	return out
}

func reduceClaims(people []PersonPortrait) []Claim {
	groups := map[string]*claimAgg{}
	orig := map[string]string{}
	for _, p := range people {
		for _, c := range p.ValuesMotivations {
			k := normKey(c.Claim)
			g := groups[k]
			if g == nil {
				g = &claimAgg{key: k, order: len(groups), conf: ConfLow}
				groups[k] = g
				orig[k] = strings.TrimSpace(c.Claim)
			}
			g.ev = append(g.ev, c.Evidence...)
			g.conf = maxConf(g.conf, c.Confidence)
		}
	}
	var out []Claim
	for _, g := range rankClaims(groups) {
		out = append(out, Claim{Claim: orig[g.key], Confidence: g.conf, Evidence: unionSorted(g.ev)})
	}
	return out
}

func reducePredictions(people []PersonPortrait) []Prediction {
	groups := map[string]*claimAgg{}
	orig := map[string]string{}
	for _, p := range people {
		for _, pr := range p.WhatToExpect {
			k := normKey(pr.Prediction)
			g := groups[k]
			if g == nil {
				g = &claimAgg{key: k, order: len(groups), conf: ConfLow}
				groups[k] = g
				orig[k] = strings.TrimSpace(pr.Prediction)
			}
			g.ev = append(g.ev, pr.Evidence...)
			g.conf = maxConf(g.conf, pr.Confidence)
		}
	}
	var out []Prediction
	for _, g := range rankClaims(groups) {
		out = append(out, Prediction{Prediction: orig[g.key], Confidence: g.conf, Evidence: unionSorted(g.ev)})
	}
	return out
}

func reduceEmotional(people []PersonPortrait) []EmotionalPattern {
	groups := map[string]*claimAgg{}
	orig := map[string]string{}
	triggers := map[string][]string{}
	for _, p := range people {
		for _, ep := range p.EmotionalPatterns {
			k := normKey(ep.Pattern)
			g := groups[k]
			if g == nil {
				g = &claimAgg{key: k, order: len(groups), conf: ConfLow}
				groups[k] = g
				orig[k] = strings.TrimSpace(ep.Pattern)
			}
			g.ev = append(g.ev, ep.Evidence...)
			g.conf = maxConf(g.conf, ep.Confidence)
			triggers[k] = append(triggers[k], ep.Triggers...)
		}
	}
	var out []EmotionalPattern
	for _, g := range rankClaims(groups) {
		out = append(out, EmotionalPattern{
			Pattern:    orig[g.key],
			Triggers:   uniqueStrings(triggers[g.key]),
			Confidence: g.conf,
			Evidence:   unionSorted(g.ev),
		})
	}
	return out
}

func normKey(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

func confRank(c Confidence) int {
	switch c {
	case ConfHigh:
		return 3
	case ConfMedium:
		return 2
	default:
		return 1
	}
}

func maxConf(a, b Confidence) Confidence {
	if confRank(b) > confRank(a) {
		return b
	}
	return a
}

func unionSorted(ids []int) []int {
	if len(ids) == 0 {
		return nil
	}
	seen := map[int]bool{}
	var out []int
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	sort.Ints(out)
	return out
}

func uniqueStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
