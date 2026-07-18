// Package analyzellm is the LLM-backed adapter for the analyze.Analyzer port. It
// maps one conversation chunk to a portrait Fragment for a single target: it
// builds an evidence-citing prompt from the domain catalogs, calls the chat
// model at temperature 0 with a json_schema response format, parses the flat M1
// reply, and applies the member-id guard so no claim can survive on fabricated
// or out-of-chunk evidence.
package analyzellm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
)

// maxTokens bounds the fragment reply. It is generous because reasoning models
// (e.g. DeepSeek V4) spend part of the output budget on hidden reasoning before
// the JSON, so a tight cap would truncate the fragment mid-object.
const maxTokens = 8192

// adapter implements analyze.Analyzer over an llm.Client.
type adapter struct {
	client        *llm.Client
	model         string
	repairRetries int
}

// New wires an analyze.Analyzer over an llm.Client. repairRetries is the number
// of extra attempts made when the model returns unparseable JSON.
func New(c *llm.Client, model string, repairRetries int) analyze.Analyzer {
	return &adapter{client: c, model: model, repairRetries: repairRetries}
}

// wireTrait is the on-wire big-five dimension: the model reports a score,
// confidence, and cited ids. Whether the trait is "assessed" is derived here
// (post-guard), never trusted from the model.
type wireTrait struct {
	Score      *float64            `json:"score"`
	Confidence portrait.Confidence `json:"confidence"`
	Evidence   []int               `json:"evidence"`
}

type wireBigFive struct {
	Openness          wireTrait `json:"openness"`
	Conscientiousness wireTrait `json:"conscientiousness"`
	Extraversion      wireTrait `json:"extraversion"`
	Agreeableness     wireTrait `json:"agreeableness"`
	Neuroticism       wireTrait `json:"neuroticism"`
}

// wireFragment is the flat single-subject shape the model returns for the M1
// fields, before the member-id guard maps it onto portrait.PersonPortrait.
type wireFragment struct {
	BigFive            wireBigFive                 `json:"big_five"`
	Accentuations      []portrait.Accentuation     `json:"accentuations"`
	CommunicationStyle portrait.CommunicationStyle `json:"communication_style"`
	RecurringTopics    []portrait.Topic            `json:"recurring_topics"`
	ValuesMotivations  []portrait.Claim            `json:"values_motivations"`
	EmotionalPatterns  []portrait.EmotionalPattern `json:"emotional_patterns"`
	WhatToExpect       []portrait.Prediction       `json:"what_to_expect"`
}

// wireDynamics is the on-wire shape for the 1:1 relationship reply: Karpman role
// tendencies keyed by participant and an overall power-balance claim, before the
// member-id guard maps them onto portrait.Dynamics.
type wireDynamics struct {
	KarpmanRoles map[string][]portrait.KarpmanRole `json:"karpman_roles"`
	PowerBalance portrait.Claim                    `json:"power_balance"`
}

// Analyze maps chunk to a guarded Fragment for target. It re-prompts with the
// parse error up to repairRetries times, then returns the error.
func (a *adapter) Analyze(ctx context.Context, chunk portrait.Chunk, target string) (portrait.Fragment, error) {
	base := buildPrompt(target, chunk.Text)
	schema := fragmentSchema()

	var parseErr error
	for attempt := 0; attempt <= a.repairRetries; attempt++ {
		prompt := base
		if parseErr != nil {
			prompt = base + "\n\nYour previous reply was not valid JSON (" + parseErr.Error() +
				"). Return ONLY the corrected JSON object, nothing else."
		}
		raw, err := a.client.Chat(ctx, prompt, llm.ChatOptions{
			Temperature:    0,
			MaxTokens:      maxTokens,
			ResponseFormat: schema,
		})
		if err != nil {
			return portrait.Fragment{}, fmt.Errorf("analyze chunk %d: chat: %w", chunk.Index, err)
		}

		var wf wireFragment
		if err := json.Unmarshal([]byte(raw), &wf); err != nil {
			parseErr = err
			continue
		}
		person, dropped := guard(wf, chunk.MemberIDs)
		return portrait.Fragment{
			ChunkIndex:      chunk.Index,
			Target:          target,
			Person:          person,
			EvidenceDropped: dropped,
		}, nil
	}
	return portrait.Fragment{}, fmt.Errorf("analyze chunk %d: parse reply: %w", chunk.Index, parseErr)
}

// guard enforces the evidence member-id rule: every cited id must belong to the
// chunk. A big-five dimension left without valid evidence (or a score) becomes
// NotAssessed; a topic, value, or M2 list item left without valid evidence is
// dropped, as is an accentuation whose type is not in the domain catalog; a
// communication style left without valid evidence is emptied so it is suppressed
// like every other no-evidence field. It returns the number of cited ids removed
// by the member-id filter across all fields (for meta.evidence_ids_dropped).
func guard(wf wireFragment, members map[int]bool) (portrait.PersonPortrait, int) {
	dropped := 0
	keep := func(ids []int) []int {
		out := filterIDs(ids, members)
		dropped += len(ids) - len(out)
		return out
	}
	p := portrait.PersonPortrait{
		BigFive: portrait.BigFive{
			Openness:          guardTrait(wf.BigFive.Openness, keep),
			Conscientiousness: guardTrait(wf.BigFive.Conscientiousness, keep),
			Extraversion:      guardTrait(wf.BigFive.Extraversion, keep),
			Agreeableness:     guardTrait(wf.BigFive.Agreeableness, keep),
			Neuroticism:       guardTrait(wf.BigFive.Neuroticism, keep),
		},
	}

	cs := wf.CommunicationStyle
	if cs.Evidence = keep(cs.Evidence); len(cs.Evidence) == 0 {
		cs = portrait.CommunicationStyle{}
	}
	p.CommunicationStyle = cs

	for _, t := range wf.RecurringTopics {
		if t.Evidence = keep(t.Evidence); len(t.Evidence) == 0 {
			continue
		}
		p.RecurringTopics = append(p.RecurringTopics, t)
	}
	for _, c := range wf.ValuesMotivations {
		if c.Evidence = keep(c.Evidence); len(c.Evidence) == 0 {
			continue
		}
		p.ValuesMotivations = append(p.ValuesMotivations, c)
	}
	for _, ac := range wf.Accentuations {
		if !inCatalog(ac.Type, portrait.AccentuationTypes) {
			continue
		}
		if ac.Evidence = keep(ac.Evidence); len(ac.Evidence) == 0 {
			continue
		}
		p.Accentuations = append(p.Accentuations, ac)
	}
	for _, ep := range wf.EmotionalPatterns {
		if ep.Evidence = keep(ep.Evidence); len(ep.Evidence) == 0 {
			continue
		}
		p.EmotionalPatterns = append(p.EmotionalPatterns, ep)
	}
	for _, pr := range wf.WhatToExpect {
		if pr.Evidence = keep(pr.Evidence); len(pr.Evidence) == 0 {
			continue
		}
		p.WhatToExpect = append(p.WhatToExpect, pr)
	}
	return p, dropped
}

func guardTrait(wt wireTrait, keep func([]int) []int) portrait.TraitScore {
	ev := keep(wt.Evidence)
	if len(ev) == 0 || wt.Score == nil {
		return portrait.NotAssessed()
	}
	conf := wt.Confidence
	if conf == "" {
		conf = portrait.ConfLow
	}
	return portrait.TraitScore{Assessed: true, Score: wt.Score, Confidence: conf, Evidence: ev}
}

func filterIDs(ids []int, members map[int]bool) []int {
	var out []int
	for _, id := range ids {
		if members[id] {
			out = append(out, id)
		}
	}
	return out
}

func inCatalog(v string, catalog []string) bool {
	for _, c := range catalog {
		if v == c {
			return true
		}
	}
	return false
}

// buildPrompt states the target, embeds the transcript, and forbids scoring or
// claiming without a cited [<id>] message id — the no-fabrication rule modeled
// on internal/bot/profile/profile.go's buildPrompt. The JSON shape is described
// in words because the server does not inject the schema into the prompt.
func buildPrompt(target, transcript string) string {
	var b strings.Builder
	b.WriteString("You are a careful psychological analyst. Analyze the participant identified as ")
	b.WriteString(target)
	b.WriteString(" using ONLY the conversation transcript below.\n\n")
	b.WriteString("Each transcript line is prefixed with its message id in the form [<id>]. ")
	b.WriteString("For every score you assign and every claim you make, you MUST cite in that field's ")
	b.WriteString("\"evidence\" array the [<id>] message ids it is based on, as integers.\n\n")
	b.WriteString("Strict rules:\n")
	b.WriteString("1. Do not assign a score, report an item, or make a claim without at least one cited [<id>] message id as evidence. ")
	b.WriteString("If you have no evidence for a field, return an empty evidence array and a null score.\n")
	b.WriteString("2. Never fabricate evidence ids: cite only ids that actually appear in the transcript, and only ids belonging to the analyzed participant.\n")
	b.WriteString("3. Base every inference strictly on the transcript; do not guess beyond it.\n\n")
	b.WriteString("Transcript:\n")
	b.WriteString(transcript)
	b.WriteString("\n\nReturn a single JSON object with exactly these fields:\n")
	b.WriteString("- \"big_five\": an object with keys ")
	b.WriteString(strings.Join(portrait.OCEAN, ", "))
	b.WriteString("; each value is an object {\"score\": number in 0..1 or null, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	b.WriteString("- \"communication_style\": an object {\"tone\", \"formality\", \"directness\", \"humor\", ")
	b.WriteString("\"emotional_expressiveness\", \"conflict_style\": short strings, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	b.WriteString("- \"recurring_topics\": array of {\"topic\": string, \"weight\": number in 0..1, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	b.WriteString("- \"values_motivations\": array of {\"claim\": string, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	b.WriteString("- \"accentuations\": array of {\"type\": one of ")
	b.WriteString(strings.Join(portrait.AccentuationTypes, ", "))
	b.WriteString(", \"intensity\": number in 0..1, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	b.WriteString("- \"emotional_patterns\": array of {\"pattern\": string, \"triggers\": array of strings, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	b.WriteString("- \"what_to_expect\": array of {\"prediction\": string, \"confidence\": one of ")
	b.WriteString(confidencePhrase())
	b.WriteString(", \"evidence\": array of integer message ids}.\n")
	return b.String()
}

func confidencePhrase() string {
	return strings.Join(confidenceValues(), ", ")
}

// AnalyzeDynamics infers the 1:1 relationship dynamics between participants a and
// b for chunk: their Karpman drama-triangle role tendencies and an overall
// power-balance claim. It shares the repair-retry loop and member-id guard with
// Analyze; the relationship arc is assembled in the service, not here.
func (ad *adapter) AnalyzeDynamics(ctx context.Context, chunk portrait.Chunk, a, b string) (portrait.Fragment, error) {
	base := buildDynamicsPrompt(a, b, chunk.Text)
	schema := dynamicsSchema(a, b)

	var parseErr error
	for attempt := 0; attempt <= ad.repairRetries; attempt++ {
		prompt := base
		if parseErr != nil {
			prompt = base + "\n\nYour previous reply was not valid JSON (" + parseErr.Error() +
				"). Return ONLY the corrected JSON object, nothing else."
		}
		raw, err := ad.client.Chat(ctx, prompt, llm.ChatOptions{
			Temperature:    0,
			MaxTokens:      maxTokens,
			ResponseFormat: schema,
		})
		if err != nil {
			return portrait.Fragment{}, fmt.Errorf("analyze dynamics chunk %d: chat: %w", chunk.Index, err)
		}

		var wd wireDynamics
		if err := json.Unmarshal([]byte(raw), &wd); err != nil {
			parseErr = err
			continue
		}
		d, dropped := guardDynamics(wd, a, b, chunk.MemberIDs)
		return portrait.Fragment{
			ChunkIndex:      chunk.Index,
			Target:          a,
			Dynamics:        &d,
			EvidenceDropped: dropped,
		}, nil
	}
	return portrait.Fragment{}, fmt.Errorf("analyze dynamics chunk %d: parse reply: %w", chunk.Index, parseErr)
}

// guardDynamics keys the Karpman roles by participants a and b, dropping any role
// not in portrait.KarpmanRoles or left without valid evidence, and filters the
// power-balance claim's evidence to chunk members. The relationship arc is nil.
// It returns the guarded dynamics and the number of cited ids removed by the
// member-id filter (for meta.evidence_ids_dropped).
func guardDynamics(wd wireDynamics, a, b string, members map[int]bool) (portrait.Dynamics, int) {
	dropped := 0
	keep := func(ids []int) []int {
		out := filterIDs(ids, members)
		dropped += len(ids) - len(out)
		return out
	}
	roles := map[string][]portrait.KarpmanRole{
		a: guardKarpmanRoles(wd.KarpmanRoles[a], keep),
		b: guardKarpmanRoles(wd.KarpmanRoles[b], keep),
	}
	pb := wd.PowerBalance
	pb.Evidence = keep(pb.Evidence)
	return portrait.Dynamics{KarpmanRoles: roles, PowerBalance: pb, RelationshipArc: nil}, dropped
}

func guardKarpmanRoles(in []portrait.KarpmanRole, keep func([]int) []int) []portrait.KarpmanRole {
	var out []portrait.KarpmanRole
	for _, r := range in {
		if !inCatalog(r.Role, portrait.KarpmanRoles) {
			continue
		}
		if r.Evidence = keep(r.Evidence); len(r.Evidence) == 0 {
			continue
		}
		out = append(out, r)
	}
	return out
}

// buildDynamicsPrompt asks for the Karpman role tendencies of a and b and an
// overall power-balance claim, each field citing [<id>] evidence, with roles
// restricted to the domain catalog.
func buildDynamicsPrompt(a, b, transcript string) string {
	var sb strings.Builder
	sb.WriteString("You are a careful psychological analyst. Analyze the 1:1 relationship dynamics between two participants using ONLY the conversation transcript below.\n\n")
	sb.WriteString("Participant A is identified as ")
	sb.WriteString(a)
	sb.WriteString("; participant B is identified as ")
	sb.WriteString(b)
	sb.WriteString(".\n\n")
	sb.WriteString("Each transcript line is prefixed with its message id in the form [<id>]. ")
	sb.WriteString("For every role tendency and every claim, you MUST cite in that field's ")
	sb.WriteString("\"evidence\" array the [<id>] message ids it is based on, as integers.\n\n")
	sb.WriteString("Strict rules:\n")
	sb.WriteString("1. Do not report a role tendency or make a claim without at least one cited [<id>] message id as evidence.\n")
	sb.WriteString("2. Never fabricate evidence ids: cite only ids that actually appear in the transcript.\n")
	sb.WriteString("3. Karpman drama-triangle roles are restricted to: ")
	sb.WriteString(strings.Join(portrait.KarpmanRoles, ", "))
	sb.WriteString(".\n")
	sb.WriteString("4. Base every inference strictly on the transcript; do not guess beyond it.\n\n")
	sb.WriteString("Transcript:\n")
	sb.WriteString(transcript)
	sb.WriteString("\n\nReturn a single JSON object with exactly these fields:\n")
	sb.WriteString("- \"karpman_roles\": an object with keys \"")
	sb.WriteString(a)
	sb.WriteString("\" and \"")
	sb.WriteString(b)
	sb.WriteString("\"; each value is an array of {\"role\": one of ")
	sb.WriteString(strings.Join(portrait.KarpmanRoles, ", "))
	sb.WriteString(", \"tendency\": number in 0..1, \"confidence\": one of ")
	sb.WriteString(confidencePhrase())
	sb.WriteString(", \"evidence\": array of integer message ids}.\n")
	sb.WriteString("- \"power_balance\": an object {\"claim\": string, \"confidence\": one of ")
	sb.WriteString(confidencePhrase())
	sb.WriteString(", \"evidence\": array of integer message ids} describing the overall balance of power between them.\n")
	return sb.String()
}
