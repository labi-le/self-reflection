package analyzellm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
)

// chatResponse wraps content in the minimal OpenAI chat-completion envelope the
// llm.Client parses.
func chatResponse(content string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"content": content}}},
	})
	return string(b)
}

const validFragment = `{
  "big_five": {
    "openness": {"score": 0.8, "confidence": "high", "evidence": [999]},
    "conscientiousness": {"score": 0.6, "confidence": "medium", "evidence": [1, 999]},
    "extraversion": {"score": null, "confidence": "low", "evidence": []},
    "agreeableness": {"score": null, "confidence": "low", "evidence": []},
    "neuroticism": {"score": null, "confidence": "low", "evidence": []}
  },
  "communication_style": {
    "tone": "warm", "formality": "casual", "directness": "direct", "humor": "dry",
    "emotional_expressiveness": "high", "conflict_style": "avoidant",
    "confidence": "medium", "evidence": [1]
  },
  "recurring_topics": [
    {"topic": "work", "weight": 0.5, "confidence": "medium", "evidence": [999]},
    {"topic": "family", "weight": 0.3, "confidence": "low", "evidence": [1]}
  ],
  "values_motivations": [
    {"claim": "values honesty", "confidence": "medium", "evidence": [1]}
  ]
}`

// Test_Analyze_drops_non_member_evidence locks the evidence member-id guard: a
// cited id outside chunk.MemberIDs is dropped, and a trait/list-item left with
// no valid evidence becomes assessed:false / is removed — never a fabricated
// score.
func Test_Analyze_drops_non_member_evidence(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, chatResponse(validFragment))
	}))
	defer srv.Close()

	a := New(llm.New(srv.URL, "m", "", time.Second), "m", 0)
	chunk := portrait.Chunk{Index: 3, MemberIDs: map[int]bool{1: true}, Text: "[1] hi"}

	frag, err := a.Analyze(context.Background(), chunk, "user1")
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	if frag.ChunkIndex != 3 || frag.Target != "user1" {
		t.Fatalf("fragment identity wrong: %+v", frag)
	}

	// openness had only id 999 (not a member) -> not assessed, no score.
	o := frag.Person.BigFive.Openness
	if o.Assessed || o.Score != nil || len(o.Evidence) != 0 {
		t.Errorf("openness should be NotAssessed, got %+v", o)
	}
	// conscientiousness cited [1,999] -> 999 dropped, 1 kept, still assessed.
	c := frag.Person.BigFive.Conscientiousness
	if !c.Assessed || c.Score == nil || len(c.Evidence) != 1 || c.Evidence[0] != 1 {
		t.Errorf("conscientiousness guard wrong: %+v", c)
	}
	// recurring_topics: "work" cited only 999 -> dropped; "family" kept.
	if len(frag.Person.RecurringTopics) != 1 || frag.Person.RecurringTopics[0].Topic != "family" {
		t.Errorf("recurring_topics guard wrong: %+v", frag.Person.RecurringTopics)
	}
	// values_motivations survives with member id.
	if len(frag.Person.ValuesMotivations) != 1 {
		t.Errorf("values_motivations should survive: %+v", frag.Person.ValuesMotivations)
	}
	// communication_style evidence filtered to members only.
	if got := frag.Person.CommunicationStyle.Evidence; len(got) != 1 || got[0] != 1 {
		t.Errorf("communication_style evidence guard wrong: %v", got)
	}
	if frag.EvidenceDropped != 3 {
		t.Errorf("EvidenceDropped = %d, want 3 (openness 999, conscientiousness 999, topic work 999)", frag.EvidenceDropped)
	}
}

// Test_Analyze_repairs_malformed_json verifies the repair loop: a first
// unparseable reply triggers a re-prompt, and the valid retry succeeds.
func Test_Analyze_repairs_malformed_json(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			io.WriteString(w, chatResponse("this is not json {"))
			return
		}
		io.WriteString(w, chatResponse(validFragment))
	}))
	defer srv.Close()

	a := New(llm.New(srv.URL, "m", "", time.Second), "m", 2)
	chunk := portrait.Chunk{Index: 0, MemberIDs: map[int]bool{1: true}, Text: "[1] hi"}

	frag, err := a.Analyze(context.Background(), chunk, "user1")
	if err != nil {
		t.Fatalf("Analyze after repair: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected repair (2 calls), got %d", calls)
	}
	if !frag.Person.BigFive.Conscientiousness.Assessed {
		t.Errorf("repaired fragment not parsed: %+v", frag.Person.BigFive)
	}
}

// Test_Analyze_returns_error_when_retries_exhausted: persistent malformed JSON
// exhausts repairRetries and surfaces the parse error.
func Test_Analyze_returns_error_when_retries_exhausted(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		io.WriteString(w, chatResponse("still not json {"))
	}))
	defer srv.Close()

	a := New(llm.New(srv.URL, "m", "", time.Second), "m", 2)
	chunk := portrait.Chunk{Index: 1, MemberIDs: map[int]bool{1: true}, Text: "[1] hi"}

	if _, err := a.Analyze(context.Background(), chunk, "user1"); err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if calls != 3 { // initial + 2 retries
		t.Fatalf("expected 3 attempts, got %d", calls)
	}
}

// Test_buildPrompt_forbids_fabrication mirrors profile's
// Test_buildPrompt_forbids_fabrication: the built prompt must name the target,
// include the transcript, demand cited [<id>] evidence, and forbid scoring
// without it. A rewrite that drops the no-fabrication rule fails here.
func Test_buildPrompt_forbids_fabrication(t *testing.T) {
	p := strings.ToLower(buildPrompt("user1", "[7] the transcript body"))
	for _, want := range []string{
		"user1",
		"the transcript body",
		"[<id>]",
		"without at least one cited",
		"never fabricate",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt missing %q:\n%s", want, p)
		}
	}
}

// m2Fragment exercises the M2 portrait lists: an accentuation whose type is not
// in the catalog, and list items citing only an out-of-chunk id, must all be
// dropped by guard.
const m2Fragment = `{
  "big_five": {
    "openness": {"score": null, "confidence": "low", "evidence": []},
    "conscientiousness": {"score": null, "confidence": "low", "evidence": []},
    "extraversion": {"score": null, "confidence": "low", "evidence": []},
    "agreeableness": {"score": null, "confidence": "low", "evidence": []},
    "neuroticism": {"score": null, "confidence": "low", "evidence": []}
  },
  "communication_style": {
    "tone": "warm", "formality": "casual", "directness": "direct", "humor": "dry",
    "emotional_expressiveness": "high", "conflict_style": "avoidant",
    "confidence": "medium", "evidence": [1]
  },
  "recurring_topics": [],
  "values_motivations": [],
  "accentuations": [
    {"type": "demonstrative", "intensity": 0.7, "confidence": "high", "evidence": [1]},
    {"type": "not_a_real_type", "intensity": 0.9, "confidence": "high", "evidence": [1]},
    {"type": "anxious", "intensity": 0.5, "confidence": "low", "evidence": [999]}
  ],
  "emotional_patterns": [
    {"pattern": "withdraws under stress", "triggers": ["conflict"], "confidence": "medium", "evidence": [1]},
    {"pattern": "phantom", "triggers": [], "confidence": "low", "evidence": [999]}
  ],
  "what_to_expect": [
    {"prediction": "will avoid confrontation", "confidence": "medium", "evidence": [1]},
    {"prediction": "ghost", "confidence": "low", "evidence": [999]}
  ]
}`

// Test_Analyze_guards_m2_lists locks the M2 guard: an accentuation type outside
// portrait.AccentuationTypes is dropped, and any list item citing only an
// out-of-chunk id is dropped, while valid items survive.
func Test_Analyze_guards_m2_lists(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, chatResponse(m2Fragment))
	}))
	defer srv.Close()

	a := New(llm.New(srv.URL, "m", "", time.Second), "m", 0)
	chunk := portrait.Chunk{Index: 2, MemberIDs: map[int]bool{1: true}, Text: "[1] hi"}

	frag, err := a.Analyze(context.Background(), chunk, "user1")
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}

	// not_a_real_type dropped (catalog), anxious dropped (out-of-chunk evidence).
	acc := frag.Person.Accentuations
	if len(acc) != 1 || acc[0].Type != "demonstrative" {
		t.Errorf("accentuations guard wrong: %+v", acc)
	}
	// "phantom" cited only 999 -> dropped.
	ep := frag.Person.EmotionalPatterns
	if len(ep) != 1 || ep[0].Pattern != "withdraws under stress" {
		t.Errorf("emotional_patterns guard wrong: %+v", ep)
	}
	// "ghost" cited only 999 -> dropped.
	pr := frag.Person.WhatToExpect
	if len(pr) != 1 || pr[0].Prediction != "will avoid confrontation" {
		t.Errorf("what_to_expect guard wrong: %+v", pr)
	}
}

const dynamicsReply = `{
  "karpman_roles": {
    "userA": [
      {"role": "victim", "tendency": 0.6, "confidence": "medium", "evidence": [1]},
      {"role": "not_a_role", "tendency": 0.9, "confidence": "high", "evidence": [1]},
      {"role": "rescuer", "tendency": 0.4, "confidence": "low", "evidence": [999]}
    ],
    "userB": [
      {"role": "persecutor", "tendency": 0.7, "confidence": "high", "evidence": [1, 999]}
    ]
  },
  "power_balance": {"claim": "B holds more influence", "confidence": "medium", "evidence": [1, 999]}
}`

// Test_AnalyzeDynamics_guards_roles_and_evidence locks the dynamics guard: a role
// outside portrait.KarpmanRoles or with only out-of-chunk evidence is dropped,
// evidence is filtered to chunk members, and roles are keyed by participants a/b.
func Test_AnalyzeDynamics_guards_roles_and_evidence(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, chatResponse(dynamicsReply))
	}))
	defer srv.Close()

	da, ok := New(llm.New(srv.URL, "m", "", time.Second), "m", 0).(analyze.DynamicsAnalyzer)
	if !ok {
		t.Fatal("adapter does not implement analyze.DynamicsAnalyzer")
	}
	chunk := portrait.Chunk{Index: 4, MemberIDs: map[int]bool{1: true}, Text: "[1] hi"}

	frag, err := da.AnalyzeDynamics(context.Background(), chunk, "userA", "userB")
	if err != nil {
		t.Fatalf("AnalyzeDynamics: %v", err)
	}
	if frag.Dynamics == nil {
		t.Fatalf("dynamics fragment has nil Dynamics")
	}
	if frag.EvidenceDropped != 3 {
		t.Errorf("EvidenceDropped = %d, want 3 (rescuer 999, persecutor 999, power_balance 999)", frag.EvidenceDropped)
	}
	dyn := *frag.Dynamics

	if _, ok := dyn.KarpmanRoles["userA"]; !ok {
		t.Fatalf("roles not keyed by a: %+v", dyn.KarpmanRoles)
	}
	if _, ok := dyn.KarpmanRoles["userB"]; !ok {
		t.Fatalf("roles not keyed by b: %+v", dyn.KarpmanRoles)
	}
	// not_a_role dropped (catalog); rescuer dropped (out-of-chunk evidence).
	a := dyn.KarpmanRoles["userA"]
	if len(a) != 1 || a[0].Role != "victim" {
		t.Errorf("userA roles guard wrong: %+v", a)
	}
	// persecutor kept; evidence [1,999] filtered to [1].
	b := dyn.KarpmanRoles["userB"]
	if len(b) != 1 || b[0].Role != "persecutor" || len(b[0].Evidence) != 1 || b[0].Evidence[0] != 1 {
		t.Errorf("userB roles guard wrong: %+v", b)
	}
	// power_balance evidence filtered to members.
	if got := dyn.PowerBalance.Evidence; len(got) != 1 || got[0] != 1 {
		t.Errorf("power_balance evidence guard wrong: %v", got)
	}
	if dyn.RelationshipArc != nil {
		t.Errorf("relationship arc must be nil (built in service): %+v", dyn.RelationshipArc)
	}
}

const noEvidenceStyleFragment = `{
  "big_five": {
    "openness": {"score": null, "confidence": "low", "evidence": []},
    "conscientiousness": {"score": null, "confidence": "low", "evidence": []},
    "extraversion": {"score": null, "confidence": "low", "evidence": []},
    "agreeableness": {"score": null, "confidence": "low", "evidence": []},
    "neuroticism": {"score": null, "confidence": "low", "evidence": []}
  },
  "communication_style": {
    "tone": "warm", "formality": "casual", "directness": "direct", "humor": "dry",
    "emotional_expressiveness": "high", "conflict_style": "avoidant",
    "confidence": "medium", "evidence": [999]
  },
  "recurring_topics": [],
  "values_motivations": []
}`

// Test_Analyze_empties_communication_style_without_evidence locks fix 7: a
// communication style whose evidence is all out-of-chunk is emptied so it is
// suppressed like every other no-evidence field, and its dropped id is counted.
func Test_Analyze_empties_communication_style_without_evidence(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, chatResponse(noEvidenceStyleFragment))
	}))
	defer srv.Close()

	a := New(llm.New(srv.URL, "m", "", time.Second), "m", 0)
	chunk := portrait.Chunk{Index: 0, MemberIDs: map[int]bool{1: true}, Text: "[1] hi"}

	frag, err := a.Analyze(context.Background(), chunk, "user1")
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	cs := frag.Person.CommunicationStyle
	if cs.Tone != "" || cs.Formality != "" || cs.ConflictStyle != "" || len(cs.Evidence) != 0 {
		t.Errorf("communication_style with no valid evidence must be emptied, got %+v", cs)
	}
	if frag.EvidenceDropped != 1 {
		t.Errorf("EvidenceDropped = %d, want 1 (the out-of-chunk id 999)", frag.EvidenceDropped)
	}
}
