package portrait

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var updateGolden = flag.Bool("update", false, "rewrite testdata golden files")

func samplerPortrait() Portrait {
	date := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	return Portrait{
		Meta: Meta{
			Model:      "test-model",
			Disclaimer: Disclaimer,
		},
		Stats: Stats{
			PerSender: map[string]SenderStats{
				"userA": {DisplayName: "Alice", Messages: 3},
			},
		},
		Portrait: map[string]PersonPortrait{
			"userA": {
				BigFive: BigFive{
					Openness:     TraitScore{Assessed: true, Score: new(0.62), Confidence: ConfMedium, Evidence: []int{12, 88}},
					Extraversion: NotAssessed(),
				},
				Accentuations: []Accentuation{
					{Type: "hyperthymic", Intensity: 0.7, Confidence: ConfMedium, Evidence: []int{12}},
				},
				CommunicationStyle: CommunicationStyle{
					Tone: "warm", Formality: "casual", Directness: "direct",
					Humor: "dry", EmotionalExpressiveness: "high", ConflictStyle: "avoidant",
					Confidence: ConfMedium, Evidence: []int{12},
				},
				RecurringTopics: []Topic{
					{Topic: "работа", Weight: 0.4, Confidence: ConfMedium, Evidence: []int{12, 88}},
				},
				EmotionalPatterns: []EmotionalPattern{
					{Pattern: "frustration at work", Triggers: []string{"deadlines"}, Confidence: ConfLow, Evidence: []int{88}},
				},
				WhatToExpect: []Prediction{
					{Prediction: "will vent when overloaded", Confidence: ConfLow, Evidence: []int{88}},
				},
			},
		},
		Dynamics: &Dynamics{
			KarpmanRoles: map[string][]KarpmanRole{
				"userA": {{Role: "victim", Tendency: 0.6, Confidence: ConfMedium, Evidence: []int{88}}},
			},
			PowerBalance: Claim{Claim: "userB tends to lead", Confidence: ConfLow, Evidence: []int{12}},
			RelationshipArc: []ArcWindow{
				{Window: "2026-01", Summary: "2 messages, 1 initiations by userA", Confidence: ConfLow, Evidence: []int{12, 88}},
			},
		},
		EvidenceIndex: map[string]EvidenceRef{
			"12": {FromID: "userA", Date: date, Text: "снова работа"},
			"88": {FromID: "userA", Date: date, Text: "работа не волк"},
		},
	}
}

func TestRenderMarkdown_Golden(t *testing.T) {
	got := RenderMarkdown(samplerPortrait())
	golden := filepath.Join("testdata", "portrait_golden.md")

	if *updateGolden {
		if err := os.WriteFile(golden, []byte(got), 0o644); err != nil {
			t.Fatalf("update golden: %v", err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if got != string(want) {
		t.Errorf("render drifted from golden.\n--- got ---\n%s\n--- want ---\n%s", got, string(want))
	}
}

func TestRenderMarkdown_DisclaimerFirstAndSuppression(t *testing.T) {
	got := RenderMarkdown(samplerPortrait())
	first := strings.SplitN(got, "\n", 2)[0]
	if first != Disclaimer {
		t.Errorf("line 1 = %q, want disclaimer", first)
	}
	if strings.Contains(strings.ToLower(got), "extraversion") {
		t.Errorf("not-assessed extraversion must be suppressed, got:\n%s", got)
	}
	if !strings.Contains(got, "работа не волк") {
		t.Errorf("cited quote from evidence_index must be resolved, got:\n%s", got)
	}
}
