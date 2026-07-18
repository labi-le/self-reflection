package portrait

import (
	"strconv"

	"tg2llm/internal/domain/dialogue"
)

// Assemble composes the final Portrait from the deterministic stats, the reduced
// person portrait, optional 1:1 dynamics, and the source messages. It performs
// no I/O and no inference: EvidenceIndex is built to cover exactly the message
// ids cited anywhere in person/dyn/timeline, each resolved to its content so
// portrait.json is self-contained. Ids with no matching message are omitted.
func Assemble(target string, stats Stats, person PersonPortrait, dyn *Dynamics, timeline []WindowPortrait, msgs []dialogue.Message, meta Meta) Portrait {
	byID := make(map[int]dialogue.Message, len(msgs))
	for _, m := range msgs {
		byID[m.ID] = m
	}

	index := map[string]EvidenceRef{}
	add := func(ids []int) {
		for _, id := range ids {
			key := strconv.Itoa(id)
			if _, ok := index[key]; ok {
				continue
			}
			m, ok := byID[id]
			if !ok {
				continue
			}
			index[key] = EvidenceRef{FromID: m.FromID, Date: m.Date, Text: dialogue.CleanText(m)}
		}
	}

	addPersonEvidence(person, add)
	if dyn != nil {
		addDynamicsEvidence(dyn, add)
	}
	for _, w := range timeline {
		addPersonEvidence(w.Person, add)
	}

	return Portrait{
		Meta:          meta,
		Stats:         stats,
		Portrait:      map[string]PersonPortrait{target: person},
		Dynamics:      dyn,
		Timeline:      timeline,
		EvidenceIndex: index,
	}
}

func addPersonEvidence(p PersonPortrait, add func([]int)) {
	for _, t := range []TraitScore{
		p.BigFive.Openness, p.BigFive.Conscientiousness, p.BigFive.Extraversion,
		p.BigFive.Agreeableness, p.BigFive.Neuroticism,
	} {
		add(t.Evidence)
	}
	for _, a := range p.Accentuations {
		add(a.Evidence)
	}
	add(p.CommunicationStyle.Evidence)
	for _, c := range p.ValuesMotivations {
		add(c.Evidence)
	}
	for _, t := range p.RecurringTopics {
		add(t.Evidence)
	}
	for _, e := range p.EmotionalPatterns {
		add(e.Evidence)
	}
	for _, w := range p.WhatToExpect {
		add(w.Evidence)
	}
}

func addDynamicsEvidence(d *Dynamics, add func([]int)) {
	for _, roles := range d.KarpmanRoles {
		for _, r := range roles {
			add(r.Evidence)
		}
	}
	add(d.PowerBalance.Evidence)
	for _, w := range d.RelationshipArc {
		add(w.Evidence)
	}
}
