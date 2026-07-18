package portrait

import (
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

func TestReduceDynamics_EvidenceWeightedTendency(t *testing.T) {
	dyns := []Dynamics{
		{KarpmanRoles: map[string][]KarpmanRole{
			"user1": {{Role: "victim", Tendency: 0.2, Confidence: ConfLow, Evidence: []int{1}}},
		}},
		{KarpmanRoles: map[string][]KarpmanRole{
			"user1": {{Role: "victim", Tendency: 0.8, Confidence: ConfHigh, Evidence: []int{4, 3, 2}}},
		}},
	}
	out := ReduceDynamics("user1", "user2", dyns)
	roles := out.KarpmanRoles["user1"]
	if len(roles) != 1 {
		t.Fatalf("roles = %+v, want 1 victim", roles)
	}
	v := roles[0]
	if v.Role != "victim" {
		t.Fatalf("role = %q, want victim", v.Role)
	}
	// evidence-count-weighted mean: (0.2*1 + 0.8*3) / 4 = 0.65
	if !approxEq(v.Tendency, 0.65) {
		t.Errorf("tendency = %v, want 0.65", v.Tendency)
	}
	if v.Confidence != ConfHigh {
		t.Errorf("confidence = %v, want high", v.Confidence)
	}
	want := []int{1, 2, 3, 4}
	if len(v.Evidence) != len(want) {
		t.Fatalf("evidence = %v, want %v", v.Evidence, want)
	}
	for i := range want {
		if v.Evidence[i] != want[i] {
			t.Fatalf("evidence = %v, want sorted-unique %v", v.Evidence, want)
		}
	}
}

func TestReduceDynamics_DropsNonCatalogAndEmptyEvidenceRoles(t *testing.T) {
	dyns := []Dynamics{
		{KarpmanRoles: map[string][]KarpmanRole{
			"user1": {
				{Role: "saboteur", Tendency: 0.9, Confidence: ConfHigh, Evidence: []int{1}}, // not in catalog
				{Role: "victim", Tendency: 0.5, Confidence: ConfLow, Evidence: nil},          // empty evidence
			},
		}},
	}
	out := ReduceDynamics("user1", "user2", dyns)
	if len(out.KarpmanRoles) != 0 {
		t.Fatalf("expected all roles dropped, got %+v", out.KarpmanRoles)
	}
}

func TestReduceDynamics_PowerBalanceHighestEvidence(t *testing.T) {
	dyns := []Dynamics{
		{PowerBalance: Claim{Claim: "b leads", Confidence: ConfLow, Evidence: []int{1}}},
		{PowerBalance: Claim{Claim: "a leads", Confidence: ConfHigh, Evidence: []int{2, 3}}},
		{PowerBalance: Claim{Claim: "", Confidence: ConfHigh, Evidence: []int{9}}}, // empty claim skipped
	}
	out := ReduceDynamics("user1", "user2", dyns)
	if out.PowerBalance.Claim != "a leads" {
		t.Fatalf("power balance claim = %q, want %q", out.PowerBalance.Claim, "a leads")
	}
	if out.PowerBalance.Confidence != ConfHigh {
		t.Errorf("power balance confidence = %v, want high", out.PowerBalance.Confidence)
	}
	want := []int{1, 2, 3}
	if len(out.PowerBalance.Evidence) != len(want) {
		t.Fatalf("power balance evidence = %v, want unioned %v", out.PowerBalance.Evidence, want)
	}
	for i := range want {
		if out.PowerBalance.Evidence[i] != want[i] {
			t.Fatalf("power balance evidence = %v, want %v", out.PowerBalance.Evidence, want)
		}
	}
}

func TestReduceDynamics_NoDynamicsIsZero(t *testing.T) {
	out := ReduceDynamics("user1", "user2", nil)
	if out.KarpmanRoles != nil {
		t.Errorf("karpman roles = %+v, want nil", out.KarpmanRoles)
	}
	if out.PowerBalance.Claim != "" || len(out.PowerBalance.Evidence) != 0 {
		t.Errorf("power balance = %+v, want zero Claim", out.PowerBalance)
	}
}

func TestRelationshipArc_MonthlyBuckets(t *testing.T) {
	jan := time.Date(2026, 1, 5, 10, 0, 0, 0, time.UTC)
	feb := time.Date(2026, 2, 3, 10, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 3, FromID: "user1", Date: feb},                     // out of order on purpose
		{ID: 1, FromID: "user1", Date: jan},                     // session start by user1 -> init
		{ID: 2, FromID: "user2", Date: jan.Add(10 * time.Minute)}, // same session
	}
	arc := RelationshipArc(msgs, "user1", 30*time.Minute)
	if len(arc) != 2 {
		t.Fatalf("windows = %+v, want 2", arc)
	}
	if arc[0].Window != "2026-01" || arc[1].Window != "2026-02" {
		t.Fatalf("windows not sorted ascending: %q, %q", arc[0].Window, arc[1].Window)
	}
	if arc[0].Confidence != ConfLow {
		t.Errorf("confidence = %v, want low", arc[0].Confidence)
	}
	// 2026-01: 2 messages, 1 initiation by user1 (id 1 starts the first session).
	if arc[0].Summary != "2 messages, 1 initiations by user1" {
		t.Errorf("jan summary = %q", arc[0].Summary)
	}
	if len(arc[0].Evidence) != 2 || arc[0].Evidence[0] != 1 || arc[0].Evidence[1] != 2 {
		t.Errorf("jan evidence = %v, want [1 2]", arc[0].Evidence)
	}
	// 2026-02: 1 message, 1 initiation by user1 (gap > sessionGap).
	if arc[1].Summary != "1 messages, 1 initiations by user1" {
		t.Errorf("feb summary = %q", arc[1].Summary)
	}
	if len(arc[1].Evidence) != 1 || arc[1].Evidence[0] != 3 {
		t.Errorf("feb evidence = %v, want [3]", arc[1].Evidence)
	}
}

func TestRelationshipArc_EvidenceCap(t *testing.T) {
	base := time.Date(2026, 3, 1, 9, 0, 0, 0, time.UTC)
	var msgs []dialogue.Message
	for i := range 30 {
		msgs = append(msgs, dialogue.Message{ID: i + 1, FromID: "user1", Date: base.Add(time.Duration(i) * time.Minute)})
	}
	arc := RelationshipArc(msgs, "user1", time.Hour)
	if len(arc) != 1 {
		t.Fatalf("windows = %d, want 1", len(arc))
	}
	if len(arc[0].Evidence) != 20 {
		t.Errorf("evidence len = %d, want cap 20", len(arc[0].Evidence))
	}
}
