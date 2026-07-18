package portrait

import (
	"fmt"
	"testing"
)

func TestReduce_WeightedMeanBigFive(t *testing.T) {
	frags := []Fragment{
		{ChunkIndex: 0, Target: "userA", Person: PersonPortrait{
			BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: new(0.4), Confidence: ConfLow, Evidence: []int{1}}},
		}},
		{ChunkIndex: 1, Target: "userA", Person: PersonPortrait{
			BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: new(0.8), Confidence: ConfHigh, Evidence: []int{4, 3, 2}}},
		}},
	}
	p := Reduce("userA", frags)
	o := p.BigFive.Openness
	if !o.Assessed || o.Score == nil {
		t.Fatalf("openness should be assessed, got %+v", o)
	}
	if !approxEq(*o.Score, 0.7) {
		t.Errorf("openness score = %v, want 0.7", *o.Score)
	}
	if o.Confidence != ConfHigh {
		t.Errorf("openness confidence = %v, want high", o.Confidence)
	}
	want := []int{1, 2, 3, 4}
	if len(o.Evidence) != len(want) {
		t.Fatalf("evidence = %v, want %v", o.Evidence, want)
	}
	for i := range want {
		if o.Evidence[i] != want[i] {
			t.Fatalf("evidence = %v, want sorted %v", o.Evidence, want)
		}
	}
}

func TestReduce_UnassessedDimensionSuppressed(t *testing.T) {
	frags := []Fragment{
		{Target: "userA", Person: PersonPortrait{
			BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: new(0.5), Confidence: ConfMedium, Evidence: []int{1}}},
		}},
	}
	p := Reduce("userA", frags)
	n := p.BigFive.Neuroticism
	if n.Assessed || n.Score != nil {
		t.Fatalf("neuroticism = %+v, want not-assessed with nil score", n)
	}
}

func TestReduce_TopicDedupe(t *testing.T) {
	frags := []Fragment{
		{Target: "userA", Person: PersonPortrait{
			RecurringTopics: []Topic{{Topic: "работа", Weight: 0.4, Confidence: ConfMedium, Evidence: []int{1}}},
		}},
		{Target: "userA", Person: PersonPortrait{
			RecurringTopics: []Topic{{Topic: "Работа ", Weight: 0.6, Confidence: ConfLow, Evidence: []int{2}}},
		}},
	}
	p := Reduce("userA", frags)
	if len(p.RecurringTopics) != 1 {
		t.Fatalf("recurring_topics = %d, want 1 merged", len(p.RecurringTopics))
	}
	tp := p.RecurringTopics[0]
	if !approxEq(tp.Weight, 0.5) {
		t.Errorf("merged weight = %v, want 0.5", tp.Weight)
	}
	if len(tp.Evidence) != 2 || tp.Evidence[0] != 1 || tp.Evidence[1] != 2 {
		t.Errorf("merged evidence = %v, want [1 2]", tp.Evidence)
	}
	if tp.Confidence != ConfMedium {
		t.Errorf("merged confidence = %v, want medium", tp.Confidence)
	}
}

func TestReduce_TopKCap(t *testing.T) {
	var topics []Topic
	for i := range 15 {
		ev := make([]int, i+1)
		for j := range ev {
			ev[j] = i*100 + j
		}
		topics = append(topics, Topic{
			Topic:      fmt.Sprintf("topic%02d", i),
			Weight:     0.5,
			Confidence: ConfLow,
			Evidence:   ev,
		})
	}
	frags := []Fragment{{Target: "userA", Person: PersonPortrait{RecurringTopics: topics}}}
	p := Reduce("userA", frags)
	if len(p.RecurringTopics) != 10 {
		t.Fatalf("recurring_topics = %d, want top-10", len(p.RecurringTopics))
	}
	if p.RecurringTopics[0].Topic != "topic14" {
		t.Errorf("top entry = %q, want topic14 (most evidence)", p.RecurringTopics[0].Topic)
	}
}

func TestReduce_FiltersByTarget(t *testing.T) {
	frags := []Fragment{
		{Target: "userA", Person: PersonPortrait{
			BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: new(0.2), Confidence: ConfLow, Evidence: []int{1}}},
		}},
		{Target: "userB", Person: PersonPortrait{
			BigFive: BigFive{Openness: TraitScore{Assessed: true, Score: new(0.9), Confidence: ConfHigh, Evidence: []int{2}}},
		}},
	}
	p := Reduce("userA", frags)
	if !approxEq(*p.BigFive.Openness.Score, 0.2) {
		t.Errorf("openness = %v, want 0.2 (only userA fragment)", *p.BigFive.Openness.Score)
	}
}
