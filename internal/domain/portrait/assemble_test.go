package portrait

import (
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

func msg(id int, from, text string, date time.Time) dialogue.Message {
	return dialogue.Message{ID: id, FromID: from, ExtractedText: text, Date: date}
}

func TestAssemble_EvidenceIndexCoversCitedIDs(t *testing.T) {
	base := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		msg(1, "user1", "hello there", base),
		msg(2, "user2", "how are you", base.Add(time.Minute)),
		msg(3, "user1", "great, thanks", base.Add(2*time.Minute)),
		msg(4, "user2", "unused message", base.Add(3*time.Minute)),
	}
	person := PersonPortrait{
		BigFive: BigFive{
			Openness: TraitScore{Assessed: true, Score: new(0.6), Confidence: ConfHigh, Evidence: []int{1, 3}},
		},
		RecurringTopics: []Topic{
			{Topic: "greetings", Weight: 0.5, Confidence: ConfMedium, Evidence: []int{2}},
		},
	}
	dyn := &Dynamics{
		PowerBalance: Claim{Claim: "balanced", Confidence: ConfLow, Evidence: []int{3}},
	}
	meta := Meta{Model: "m", Disclaimer: Disclaimer}

	p := Assemble("user1", stubStats(), person, dyn, nil, msgs, meta)

	if p.Portrait["user1"].BigFive.Openness.Score == nil {
		t.Fatalf("person not stored under target")
	}
	if p.Dynamics != dyn {
		t.Fatalf("dynamics not carried through")
	}
	if p.Meta.Model != "m" {
		t.Fatalf("meta not carried through")
	}
	wantKeys := []string{"1", "2", "3"}
	for _, k := range wantKeys {
		ref, ok := p.EvidenceIndex[k]
		if !ok {
			t.Fatalf("evidence_index missing id %s", k)
		}
		if ref.Text == "" {
			t.Fatalf("evidence ref %s has empty text", k)
		}
	}
	if _, ok := p.EvidenceIndex["4"]; ok {
		t.Fatalf("id 4 was never cited; must not appear in evidence_index")
	}
	// Resolution correctness: id 1 is user1's "hello there".
	if got := p.EvidenceIndex["1"]; got.FromID != "user1" || got.Text != "hello there" || !got.Date.Equal(base) {
		t.Fatalf("evidence ref 1 wrong: %+v", got)
	}
}

func TestAssemble_UnresolvableIDsSkipped(t *testing.T) {
	person := PersonPortrait{
		BigFive: BigFive{
			Neuroticism: TraitScore{Assessed: true, Score: new(0.2), Confidence: ConfLow, Evidence: []int{99}},
		},
	}
	p := Assemble("user1", stubStats(), person, nil, nil, nil, Meta{})
	if len(p.EvidenceIndex) != 0 {
		t.Fatalf("unresolvable id 99 should be skipped, got %v", p.EvidenceIndex)
	}
}

func stubStats() Stats { return Stats{PerSender: map[string]SenderStats{}} }
