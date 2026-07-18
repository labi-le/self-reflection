// Package portrait holds the pure, I/O-free domain model of a psychological
// analysis: deterministic conversation statistics, the LLM-inferred portrait of
// a participant, 1:1 relationship dynamics, and the self-contained evidence
// index that lets any renderer resolve a cited message id to its text. It also
// owns the framework catalogs (OCEAN, Leonhard-Shmishek accentuations, Karpman
// roles) as the single source of truth shared by prompt builders and the JSON
// schema. Nothing here performs I/O or talks to a model.
package portrait

import (
	"time"

	"tg2llm/internal/domain/dialogue"
)

// Confidence is the qualitative trust an inferred claim carries.
type Confidence string

const (
	ConfLow    Confidence = "low"
	ConfMedium Confidence = "medium"
	ConfHigh   Confidence = "high"
)

// Framework catalogs — the single source of truth for prompts and the JSON
// schema/grammar. Keep these in sync with the spec's section 4.5.
var (
	OCEAN             = []string{"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
	AccentuationTypes = []string{"demonstrative", "pedantic", "stuck", "excitable", "hyperthymic", "dysthymic", "anxious", "cyclothymic", "affective_exalted", "emotive"}
	KarpmanRoles      = []string{"victim", "rescuer", "persecutor"}
)

// Disclaimer ships inside portrait.json (the dashboard's contract), not only in
// the Markdown report, so no downstream renderer can present the inferences as
// clinical fact.
const Disclaimer = "LLM-inferred hypotheses from chat text, not a clinical diagnosis. Scores are a relative lean within this conversation, not population-normed."

// TraitScore is one evidence-gated 0..1 score. Score is nil when !Assessed: an
// absence of signal is never rendered as a real 0.0 measurement.
type TraitScore struct {
	Assessed   bool       `json:"assessed"`
	Score      *float64   `json:"score"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// NotAssessed is the zero-evidence marker: a claim with no supporting messages.
func NotAssessed() TraitScore {
	return TraitScore{Assessed: false, Score: nil, Confidence: ConfLow, Evidence: nil}
}

// BigFive is the OCEAN model, each dimension evidence-gated.
type BigFive struct {
	Openness          TraitScore `json:"openness"`
	Conscientiousness TraitScore `json:"conscientiousness"`
	Extraversion      TraitScore `json:"extraversion"`
	Agreeableness     TraitScore `json:"agreeableness"`
	Neuroticism       TraitScore `json:"neuroticism"`
}

// Accentuation is one Leonhard-Shmishek type observed with an intensity.
type Accentuation struct {
	Type       string     `json:"type"`
	Intensity  float64    `json:"intensity"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// CommunicationStyle describes how the participant converses.
type CommunicationStyle struct {
	Tone                    string     `json:"tone"`
	Formality               string     `json:"formality"`
	Directness              string     `json:"directness"`
	Humor                   string     `json:"humor"`
	EmotionalExpressiveness string     `json:"emotional_expressiveness"`
	ConflictStyle           string     `json:"conflict_style"`
	Confidence              Confidence `json:"confidence"`
	Evidence                []int      `json:"evidence"`
}

// Claim is a free-text inferred assertion with its support.
type Claim struct {
	Claim      string     `json:"claim"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// Topic is a recurring subject weighted by how much of the conversation it spans.
type Topic struct {
	Topic      string     `json:"topic"`
	Weight     float64    `json:"weight"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// EmotionalPattern is a recurring emotional response and what tends to trigger it.
type EmotionalPattern struct {
	Pattern    string     `json:"pattern"`
	Triggers   []string   `json:"triggers"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// Prediction is a "what to expect" behavioral forecast.
type Prediction struct {
	Prediction string     `json:"prediction"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// PersonPortrait is one participant's inferred portrait. It is ALSO the shape a
// MAP fragment carries for a single chunk (Evidence holds the ids the model
// cited, post the member-id guard).
type PersonPortrait struct {
	BigFive            BigFive            `json:"big_five"`
	Accentuations      []Accentuation     `json:"accentuations"`
	CommunicationStyle CommunicationStyle `json:"communication_style"`
	ValuesMotivations  []Claim            `json:"values_motivations"`
	RecurringTopics    []Topic            `json:"recurring_topics"`
	EmotionalPatterns  []EmotionalPattern `json:"emotional_patterns"`
	WhatToExpect       []Prediction       `json:"what_to_expect"`
}

// KarpmanRole is a drama-triangle role tendency for a participant.
type KarpmanRole struct {
	Role       string     `json:"role"`
	Tendency   float64    `json:"tendency"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// ArcWindow summarizes the relationship during one calendar window.
type ArcWindow struct {
	Window     string     `json:"window"`
	Summary    string     `json:"summary"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// Dynamics is the 1:1 relationship analysis (nil for non-pair scopes).
type Dynamics struct {
	KarpmanRoles    map[string][]KarpmanRole `json:"karpman_roles"`
	PowerBalance    Claim                    `json:"power_balance"`
	RelationshipArc []ArcWindow              `json:"relationship_arc"`
}

// EmojiCount / TermCount are deterministic frequency rows.
type EmojiCount struct {
	Emoji string `json:"emoji"`
	Count int    `json:"count"`
}
type TermCount struct {
	Term  string `json:"term"`
	Count int    `json:"count"`
}

// SenderStats are the exact, LLM-free statistics for one participant.
type SenderStats struct {
	DisplayName            string       `json:"display_name"`
	Messages               int          `json:"messages"`
	Chars                  int          `json:"chars"`
	AvgMsgLen              float64      `json:"avg_msg_len"`
	MedianMsgLen           int          `json:"median_msg_len"`
	QuestionsRatio         float64      `json:"questions_ratio"`
	ResponseLatencyMedianS int          `json:"response_latency_median_s"`
	Initiations            int          `json:"initiations"`
	ConversationEnders     int          `json:"conversation_enders"`
	ActiveHoursHistogram   [24]int      `json:"active_hours_histogram"`
	WeekdayHistogram       [7]int       `json:"weekday_histogram"`
	EmojiTop               []EmojiCount `json:"emoji_top"`
	StickerCount           int          `json:"sticker_count"`
	TopTerms               []TermCount  `json:"top_terms"`
	MonologueRatio         float64      `json:"monologue_ratio"`
}

// SilenceGap is a stretch of no messages between two timestamps.
type SilenceGap struct {
	From    time.Time `json:"from"`
	To      time.Time `json:"to"`
	Seconds int       `json:"seconds"`
}

// Pairwise are the two-party interaction statistics (1:1 only).
type Pairwise struct {
	InitiativeBalance float64      `json:"initiative_balance"`
	Reciprocity       float64      `json:"reciprocity"`
	AvgReplyGapS      int          `json:"avg_reply_gap_s"`
	SilenceGaps       []SilenceGap `json:"silence_gaps"`
}

// Stats is the full deterministic block. Pairwise is nil unless the scope is 1:1.
type Stats struct {
	PerSender map[string]SenderStats `json:"per_sender"`
	Pairwise  *Pairwise              `json:"pairwise,omitempty"`
}

// Meta carries run provenance and the machine-readable disclaimer.
type Meta struct {
	Model              string    `json:"model"`
	EmbedModel         string    `json:"embed_model,omitempty"`
	DateFrom           string    `json:"date_from"`
	DateTo             string    `json:"date_to"`
	MessageCount       int       `json:"message_count"`
	ChunksTotal        int       `json:"chunks_total"`
	ChunksAnalyzed     int       `json:"chunks_analyzed"`
	Coverage           float64   `json:"coverage"`
	EvidenceIDsDropped int       `json:"evidence_ids_dropped"`
	GeneratedAt        time.Time `json:"generated_at"`
	TokenCostEstimate  int       `json:"token_cost_estimate"`
	Disclaimer         string    `json:"disclaimer"`
}

// EvidenceRef resolves a cited message id to its content, so portrait.json is
// self-contained (no re-read of the export needed to render a quote).
type EvidenceRef struct {
	FromID string    `json:"from_id"`
	Date   time.Time `json:"date"`
	Text   string    `json:"text"`
}

// Portrait is the complete output contract written to portrait.json.
type Portrait struct {
	Meta          Meta                      `json:"meta"`
	Stats         Stats                     `json:"stats"`
	Portrait      map[string]PersonPortrait `json:"portrait"`
	Dynamics      *Dynamics                 `json:"dynamics,omitempty"`
	Timeline      []WindowPortrait          `json:"timeline,omitempty"`
	EvidenceIndex map[string]EvidenceRef    `json:"evidence_index"`
}

// Chunk is one segmentation unit fed to the MAP phase and the Q&A index.
type Chunk struct {
	Index     int
	Messages  []dialogue.Message
	MemberIDs map[int]bool
	Text      string
}

// Fragment is one chunk's MAP output for a single target, AFTER the evidence
// member-id guard has been applied by the analyzer adapter.
type Fragment struct {
	ChunkIndex     int
	Target         string
	Person         PersonPortrait
	Dynamics       *Dynamics
	EvidenceDropped int
}
