# Psychological Analysis Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `analyze` engine to `tg2llm` that turns a Telegram export into a structured, evidence-grounded `portrait.json` (deterministic stats + LLM-inferred psychological portrait + 1:1 dynamics) plus a grounded interactive Q&A — no manual copy-paste into an LLM chat.

**Architecture:** Hybrid pipeline — deterministic Go stats + per-chunk LLM MAP (schema-constrained, temperature 0, evidence validated against the chunk's member-ID set) + **pure-Go REDUCE** (evidence-weighted aggregation) + a per-analysis embedding index for Q&A. Hexagonal: pure logic in `internal/domain/portrait`, orchestration in `internal/app/analyze`, adapters in `internal/infra/*`, CLI in `cmd/analyze`. New neutral infra (`llm`, `vector`, `kvcache`) is shared with the existing bot/media code via contract-preserving refactors.

**Tech Stack:** Go 1.26 (module `tg2llm`), `modernc.org/sqlite` (pure-Go, no CGO), `rs/zerolog`. External HTTP services: an OpenAI-compatible LLM endpoint (llama.cpp / cloud) and a Chroma vector server. No new Go module dependencies.

**Spec:** `docs/superpowers/specs/2026-07-18-psych-analysis-engine-design.md` — read it before starting any task.

## Global Constraints

- **Go via Nix only:** every build/test runs `nix-shell shell.nix --run 'GOTOOLCHAIN=local go ...'`. Set `GOTOOLCHAIN=local` explicitly in one-shot `--run` invocations.
- **Two direct module deps only** (`go.mod`): `github.com/rs/zerolog`, `modernc.org/sqlite`. Adding any `require` is a plan violation.
- **Hexagonal split:** pure, I/O-free logic lives in `internal/domain/*` with **no port/interface**; `internal/infra/*` holds adapters with external I/O; `internal/app/*` orchestrates via ports. Do NOT put a pure function behind an interface.
- **CLI gotcha:** stdlib `flag` stops at the first non-flag arg. `cmd/analyze` dispatches on `args[0]` (verb) then parses `args[1:]` with a per-verb `flag.FlagSet`; flags precede positionals within each subcommand.
- **Contract-preserving refactors:** `internal/infra/media` and `internal/bot/memory` may be refactored to consume new neutral packages, but their **public APIs and existing tests MUST stay green**. If a refactor threatens the bot, stop and fall back to analysis-local infra; escalate via `irc`.
- **Evidence integrity is the product's core guarantee:** the LLM MAY cite only message IDs; every cited ID is validated against the chunk's `MemberIDs`; non-members are dropped; a claim left with no evidence is emitted `assessed:false` / `score:null` — NEVER a fabricated `0.0`.
- **Log hygiene:** logs carry only IDs, counts, and error strings — never message content or raw model replies. Content dumps are gated behind `--verbose` at trace level.
- **Commits:** plain imperative subject, no semantic prefix, subject-only (no footer, no `Co-authored-by`). Commit only when the human explicitly asks — during autonomous execution, leave commits to the orchestrator's checkpoints.
- **Comments:** self-documenting code; a hook flags new comments — keep only load-bearing ones (public-API docs, non-obvious concurrency/algorithm rationale).
- **Tests:** fakes only, no live LLM/HTTP (mirror `internal/bot/profile/profile_test.go` fakes and the black-box `internal/app/service_test.go`). Deterministic, isolated, full-suite safe.

## File structure

```
cmd/analyze/main.go                      verb dispatch + per-verb FlagSet + wiring + egress warning
cmd/analyze/portrait_cmd.go              `portrait` subcommand
cmd/analyze/ask_cmd.go                   `ask` subcommand (REPL + --question)
cmd/analyze/participants_cmd.go          `participants` subcommand
internal/domain/portrait/portrait.go     Portrait/PersonPortrait/Meta/Fragment types + catalog consts
internal/domain/portrait/stats.go        ComputeStats([]Message, target) Stats   (pure)
internal/domain/portrait/segment.go      Segment([]Message, SegmentOptions) []Chunk  (pure)
internal/domain/portrait/reduce.go       Reduce(target, []Fragment) PersonPortrait  (pure)
internal/domain/portrait/assemble.go     Assemble(...) Portrait + BuildEvidenceIndex (pure)
internal/domain/portrait/render.go       RenderMarkdown(Portrait) string  (pure, golden)
internal/app/analyze/ports.go            Analyzer, Embedder, Index interfaces + AnalyzeOptions
internal/app/analyze/service.go          AnalyzeService.Run (stats→segment→map(pool)→reduce→assemble→index→write)
internal/app/analyze/qa.go               QAService.Ask (retrieve→validate cites→answer/refuse)
internal/infra/kvcache/kvcache.go        hoisted generic content-hash SQLite KV + Signature
internal/infra/llm/llm.go                OpenAI-compatible client (chat + embeddings): host+model+key, response_format, max_tokens
internal/infra/vector/vector.go          hoisted Chroma v2 REST client + DeleteCollection
internal/infra/analyzellm/analyzellm.go  Analyzer adapter: prompt build + schema + parse + evidence guard + repair
internal/infra/analyzellm/schema.go      JSON schema / GBNF for the flat single-subject fragment
```

## Dependency waves (orchestration)

- **Wave 0 (contract, prerequisite):** Task 1 (domain types + catalogs) → Task 2 (app ports). Everything depends on these; build first.
- **Wave 1 (parallel, independent):** Task 3 kvcache+media, Task 4 vector+bot, Task 5 llm client, Task 6 stats, Task 7 segment, Task 8 reduce, Task 9 render. (3/4/5 need nothing from Wave 0 except being in the tree; 6/7/8/9 need Task 1 types.)
- **Wave 2 (parallel):** Task 10 analyzellm (needs 1,2,5), Task 11 service (needs 1,2,3,6,7,8), Task 12 qa (needs 1,2,4,5), Task 13 assemble+evidence_index (needs 1; can also fold into Task 11).
- **Wave 3 (integration, orchestrator):** Task 14 cmd/analyze wiring + Task 15 end-to-end smoke + `.gitignore`.
- **M2 extension:** Tasks 16–19 add the remaining framework fields, prompts, aggregation, and the dynamics pass.

Interface signatures are pinned in Tasks 1–2 so Wave 1/2 tasks integrate without coordination.

---

### Task 1: Domain types + framework catalogs (`internal/domain/portrait`)

**Files:**
- Create: `internal/domain/portrait/portrait.go`
- Test: `internal/domain/portrait/portrait_test.go`

**Interfaces:**
- Consumes: `tg2llm/internal/domain/dialogue` (`dialogue.Message`).
- Produces (every later task depends on these EXACT types):

```go
package portrait

import (
	"time"

	"tg2llm/internal/domain/dialogue"
)

type Confidence string

const (
	ConfLow    Confidence = "low"
	ConfMedium Confidence = "medium"
	ConfHigh   Confidence = "high"
)

// Catalogs — single source of truth for prompts and the JSON schema.
var (
	OCEAN             = []string{"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
	AccentuationTypes = []string{"demonstrative", "pedantic", "stuck", "excitable", "hyperthymic", "dysthymic", "anxious", "cyclothymic", "affective_exalted", "emotive"}
	KarpmanRoles      = []string{"victim", "rescuer", "persecutor"}
)

const Disclaimer = "LLM-inferred hypotheses from chat text, not a clinical diagnosis. Scores are a relative lean within this conversation, not population-normed."

// TraitScore is one evidence-gated 0..1 score. Score is nil when !Assessed.
type TraitScore struct {
	Assessed   bool       `json:"assessed"`
	Score      *float64   `json:"score"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

type BigFive struct {
	Openness          TraitScore `json:"openness"`
	Conscientiousness TraitScore `json:"conscientiousness"`
	Extraversion      TraitScore `json:"extraversion"`
	Agreeableness     TraitScore `json:"agreeableness"`
	Neuroticism       TraitScore `json:"neuroticism"`
}

type Accentuation struct {
	Type       string     `json:"type"`
	Intensity  float64    `json:"intensity"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

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

type Claim struct {
	Claim      string     `json:"claim"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

type Topic struct {
	Topic      string     `json:"topic"`
	Weight     float64    `json:"weight"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

type EmotionalPattern struct {
	Pattern    string     `json:"pattern"`
	Triggers   []string   `json:"triggers"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

type Prediction struct {
	Prediction string     `json:"prediction"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

// PersonPortrait is one participant's inferred portrait. It is ALSO the shape a
// MAP fragment carries for a single chunk (evidence = the IDs the model cited).
type PersonPortrait struct {
	BigFive            BigFive            `json:"big_five"`
	Accentuations      []Accentuation     `json:"accentuations"`
	CommunicationStyle CommunicationStyle `json:"communication_style"`
	ValuesMotivations  []Claim            `json:"values_motivations"`
	RecurringTopics    []Topic            `json:"recurring_topics"`
	EmotionalPatterns  []EmotionalPattern `json:"emotional_patterns"`
	WhatToExpect       []Prediction       `json:"what_to_expect"`
}

type KarpmanRole struct {
	Role       string     `json:"role"`
	Tendency   float64    `json:"tendency"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

type ArcWindow struct {
	Window     string     `json:"window"`
	Summary    string     `json:"summary"`
	Confidence Confidence `json:"confidence"`
	Evidence   []int      `json:"evidence"`
}

type Dynamics struct {
	KarpmanRoles    map[string][]KarpmanRole `json:"karpman_roles"`
	PowerBalance    Claim                    `json:"power_balance"`
	RelationshipArc []ArcWindow              `json:"relationship_arc"`
}

type EmojiCount struct {
	Emoji string `json:"emoji"`
	Count int    `json:"count"`
}
type TermCount struct {
	Term  string `json:"term"`
	Count int    `json:"count"`
}

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

type SilenceGap struct {
	From    time.Time `json:"from"`
	To      time.Time `json:"to"`
	Seconds int       `json:"seconds"`
}

type Pairwise struct {
	InitiativeBalance float64      `json:"initiative_balance"`
	Reciprocity       float64      `json:"reciprocity"`
	AvgReplyGapS      int          `json:"avg_reply_gap_s"`
	SilenceGaps       []SilenceGap `json:"silence_gaps"`
}

type Stats struct {
	PerSender map[string]SenderStats `json:"per_sender"`
	Pairwise  *Pairwise              `json:"pairwise,omitempty"`
}

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

type EvidenceRef struct {
	FromID string    `json:"from_id"`
	Date   time.Time `json:"date"`
	Text   string    `json:"text"`
}

type Portrait struct {
	Meta          Meta                      `json:"meta"`
	Stats         Stats                     `json:"stats"`
	Portrait      map[string]PersonPortrait `json:"portrait"`
	Dynamics      *Dynamics                 `json:"dynamics,omitempty"`
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
// member-ID guard has been applied.
type Fragment struct {
	ChunkIndex int
	Target     string
	Person     PersonPortrait
	Dynamics   *Dynamics // nil in M1
}
```

- [ ] **Step 1: Write the failing test** — `portrait_test.go` asserting the catalog invariants and the not-assessed marker helper.

```go
package portrait

import "testing"

func TestCatalogs(t *testing.T) {
	if len(OCEAN) != 5 {
		t.Fatalf("OCEAN = %d, want 5", len(OCEAN))
	}
	if len(AccentuationTypes) != 10 {
		t.Fatalf("accentuations = %d, want 10", len(AccentuationTypes))
	}
	if len(KarpmanRoles) != 3 {
		t.Fatalf("karpman = %d, want 3", len(KarpmanRoles))
	}
}

func TestNotAssessedTraitHasNilScore(t *testing.T) {
	ts := NotAssessed()
	if ts.Assessed || ts.Score != nil {
		t.Fatalf("NotAssessed() = %+v, want assessed=false score=nil", ts)
	}
}
```

- [ ] **Step 2: Run test to verify it fails** — `nix-shell shell.nix --run 'GOTOOLCHAIN=local go test ./internal/domain/portrait/'`. Expected: FAIL (`NotAssessed` undefined).
- [ ] **Step 3: Write minimal implementation** — the types above plus the helper:

```go
// NotAssessed is the zero-evidence marker: a claim with no supporting messages.
func NotAssessed() TraitScore { return TraitScore{Assessed: false, Score: nil, Confidence: ConfLow, Evidence: nil} }
```

- [ ] **Step 4: Run test to verify it passes** — same command. Expected: PASS.
- [ ] **Step 5: Vet** — `nix-shell shell.nix --run 'GOTOOLCHAIN=local go vet ./internal/domain/portrait/'`.

---

### Task 2: App ports (`internal/app/analyze/ports.go`)

**Files:**
- Create: `internal/app/analyze/ports.go`
- Test: `internal/app/analyze/ports_test.go` (compile-time interface-satisfaction guard, mirrors `internal/app/ports_test.go`).

**Interfaces:**
- Consumes: `internal/domain/portrait` (Task 1).
- Produces:

```go
package analyze

import (
	"context"
	"time"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
)

// Analyzer maps one chunk to a fragment via an LLM. Implementations MUST restrict
// every evidence ID to chunk.MemberIDs before returning (the member-ID guard).
type Analyzer interface {
	Analyze(ctx context.Context, chunk portrait.Chunk, target string) (portrait.Fragment, error)
}

// Embedder turns texts into vectors (OpenAI-compatible /v1/embeddings).
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Index is a per-analysis vector collection for Q&A retrieval.
type Index interface {
	Add(ctx context.Context, collection string, chunks []portrait.Chunk, vectors [][]float32) error
	Query(ctx context.Context, collection string, vector []float32, k int, where map[string]any) ([]portrait.Chunk, error)
	Delete(ctx context.Context, collection string) error
}

type AnalyzeOptions struct {
	Model         string
	EmbedModel    string
	Target        string // prefixed from_id
	With          string // second party for dynamics ("" = infer)
	Relationship  bool
	DateRange     dialogue.DateRange
	SessionGap    time.Duration
	ChunkChars    int
	Jobs          int
	RepairRetries int
}
```

- [ ] **Step 1: Write the failing test** — `ports_test.go` with `var _ Analyzer = (*fakeAnalyzer)(nil)` etc. and a trivial fake.
- [ ] **Step 2: Run** — `go test ./internal/app/analyze/`. Expected: FAIL (types undefined).
- [ ] **Step 3: Implement** — the file above.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Vet.**

---

### Task 3: Hoist `kvcache` + re-point `media` (`internal/infra/kvcache`)

**Files:**
- Create: `internal/infra/kvcache/kvcache.go`, `internal/infra/kvcache/kvcache_test.go`
- Modify: `internal/infra/media/cache.go`, `internal/infra/media/signature.go` (re-export from kvcache), and any media callers.

**Interfaces:**
- Produces: `kvcache.New(path, table string) (*Cache, error)`, `(*Cache).Get(key string) (string, bool)`, `(*Cache).Put(key, value string) error`, `(*Cache).Close() error` (nil-safe), and `kvcache.Signature(parts ...string) string` (SHA-1[:10] of parts joined by `|`).

- [ ] **Step 1:** Read `internal/infra/media/cache.go` and `signature.go` in full.
- [ ] **Step 2: Write failing test** `kvcache_test.go`: open at a temp path with a custom table, Put/Get round-trips, missing key returns `("", false)`, nil `*Cache` Get/Put/Close are safe, empty value is caller's choice (store round-trips).
- [ ] **Step 3: Run** — FAIL.
- [ ] **Step 4: Implement** — move the cache verbatim, parameterizing the table name (validate the identifier against `^[A-Za-z_][A-Za-z0-9_]*$` before interpolating, since SQLite can't bind a table name); move `Signature`.
- [ ] **Step 5: Re-point media** — `media` calls `kvcache.New(path, "media_cache")` and `kvcache.Signature`; delete the duplicated code.
- [ ] **Step 6: Run regression gate** — `nix-shell shell.nix --run 'GOTOOLCHAIN=local go test ./internal/infra/media/ ./internal/infra/kvcache/'`. Expected: PASS (media tests unchanged).

---

### Task 4: Hoist `vector` + re-point `bot/memory` (`internal/infra/vector`)

**Files:**
- Create: `internal/infra/vector/vector.go`, `internal/infra/vector/vector_test.go`
- Modify: `internal/bot/memory/chroma.go` (wrap `vector`), keeping `memory.RAG`/`Embedder` public API unchanged.

**Interfaces:**
- Produces: `vector.New(host string, opts ...Option) *Client`; `(*Client).GetOrCreate(ctx, name) (id string, err error)`; `(*Client).Add(ctx, collID string, recs []Record) error`; `(*Client).Query(ctx, collID string, vec []float32, k int, where map[string]any) ([]Hit, error)`; `(*Client).DeleteCollection(ctx, name string) error`. `Record{ID string; Embedding []float32; Document string; Metadata map[string]any}`, `Hit{Document string; Metadata map[string]any; Distance float32}`.

- [ ] **Step 1:** Read `internal/bot/memory/chroma.go` and `memory.go` in full; note `chromaStore` verbs and the `record`/`queryHit` shapes.
- [ ] **Step 2: Write failing test** `vector_test.go` using an `httptest.Server` that mimics the Chroma v2 REST contract (get-or-create → uuid; add → 200; query → nested arrays; **delete → 200**), asserting each verb builds the right URL/body and `DeleteCollection` issues the delete.
- [ ] **Step 3: Run** — FAIL.
- [ ] **Step 4: Implement** — hoist `chromaStore` verbatim as exported `Client` with `Record`/`Hit`; add `DeleteCollection` (Chroma v2 `DELETE .../collections/{name}`).
- [ ] **Step 5: Re-point bot/memory** — `chroma.go` wraps `vector.Client` (map `record`↔`Record`, `queryHit`↔`Hit`); public `memory` API unchanged.
- [ ] **Step 6: Regression gate** — `go test ./internal/bot/memory/ ./internal/infra/vector/`. Expected: PASS. If bot tests break in a way that needs a public-API change, STOP and escalate; fall back to an analysis-local `vector` client and leave `bot/memory` untouched.

---

### Task 5: OpenAI-compatible LLM client (`internal/infra/llm`)

**Files:**
- Create: `internal/infra/llm/llm.go`, `internal/infra/llm/llm_test.go`

**Interfaces:**
- Produces:

```go
package llm

type Client struct { /* host, model, key, http */ }
func New(host, model, key string, timeout time.Duration) *Client

type ChatOptions struct {
	Temperature    float64
	MaxTokens      int             // 0 => omit (server default)
	ResponseFormat json.RawMessage // nil => omit; e.g. {"type":"json_schema","json_schema":{...}}
}
// Chat POSTs one non-streaming completion and returns the assistant content.
func (c *Client) Chat(ctx context.Context, prompt string, opts ChatOptions) (string, error)
// Embed POSTs /v1/embeddings and returns one vector per input.
func (c *Client) Embed(ctx context.Context, model string, texts []string) ([][]float32, error)
```

- [ ] **Step 1: Write failing test** `llm_test.go` with an `httptest.Server`: asserts `Chat` sends bearer `Authorization` when key set (and none when empty), includes `response_format`/`max_tokens`/`temperature` when set (and omits when zero/nil), parses `choices[0].message.content`; `Embed` parses `data[].embedding`; non-200 is an error; empty choices is an error.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** — modeled on `internal/bot/profile/extractor.go` but with bearer auth, `response_format`, configurable `max_tokens`, and an embeddings method. Timeout via `http.Client{Timeout: timeout}`.
- [ ] **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 6: Deterministic stats (`internal/domain/portrait/stats.go`)

**Files:**
- Create: `internal/domain/portrait/stats.go`, `internal/domain/portrait/stats_test.go`

**Interfaces:**
- Produces: `func ComputeStats(msgs []dialogue.Message, target string, sessionGap time.Duration) Stats`. Pairwise is populated only when exactly two distinct `FromID`s are present; otherwise `Stats.Pairwise == nil`.

- [ ] **Step 1: Write failing tests** over a hand-built `[]dialogue.Message` fixture with two senders, known timestamps, one `?` message, a burst, and a >sessionGap gap. Assert exact: `messages`, `avg_msg_len`, `questions_ratio`, `initiations` (first msg after a gap), `conversation_enders`, `active_hours_histogram` bucket, `monologue_ratio`, and `pairwise.initiative_balance` sign. Add a 3-sender fixture asserting `Pairwise == nil`.

```go
func TestComputeStats_TwoSenders(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 1, FromID: "userA", Sender: "A", Date: base, ExtractedText: "привет"},
		{ID: 2, FromID: "userB", Sender: "B", Date: base.Add(2 * time.Minute), ExtractedText: "как дела?"},
		{ID: 3, FromID: "userA", Sender: "A", Date: base.Add(3 * time.Minute), ExtractedText: "норм"},
		{ID: 4, FromID: "userA", Sender: "A", Date: base.Add(90 * time.Minute), ExtractedText: "ещё тут?"},
	}
	s := ComputeStats(msgs, "userA", 30*time.Minute)
	if s.PerSender["userA"].Messages != 3 {
		t.Fatalf("A messages = %d, want 3", s.PerSender["userA"].Messages)
	}
	if s.PerSender["userB"].QuestionsRatio != 1.0 {
		t.Fatalf("B questions_ratio = %v, want 1.0", s.PerSender["userB"].QuestionsRatio)
	}
	if s.Pairwise == nil || s.Pairwise.InitiativeBalance <= 0 {
		t.Fatalf("initiative_balance should favor target A, got %+v", s.Pairwise)
	}
}
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `ComputeStats` (single pass building per-sender counters + a second pass for latency/initiation over time-sorted messages; emoji via a rune-range scan; top terms via whitespace tokenization + a small RU/EN stopword set defined in the file). **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 7: Segmentation (`internal/domain/portrait/segment.go`)

**Files:** Create `segment.go`, `segment_test.go`.

**Interfaces:**
- Produces: `func Segment(msgs []dialogue.Message, sessionGap time.Duration, chunkChars int) []Chunk`. Session boundary = gap > sessionGap. A single session whose rendered text exceeds `chunkChars` is hard-split at message boundaries. Each `Chunk` has `Index` (0-based, ascending), `Messages`, `MemberIDs`, and `Text` (rendered `[<id>] <sender>: <text>` lines).

- [ ] **Step 1: Write failing tests:** (a) two sessions separated by a >gap boundary → ≥2 chunks; (b) a single oversized session → split into multiple chunks, none exceeding `chunkChars`, union of `MemberIDs` == all input IDs; (c) `Index` values are `0..n-1` ascending; (d) empty input → empty slice.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 8: Pure-Go Reduce (`internal/domain/portrait/reduce.go`)

**Files:** Create `reduce.go`, `reduce_test.go`.

**Interfaces:**
- Produces: `func Reduce(target string, frags []Fragment) PersonPortrait`. Rules: numeric scores (`big_five`, `accentuation.intensity`, `topic.weight`) = mean weighted by `len(evidence)` across fragments that assessed them; evidence = sorted unique union; list claims (`values_motivations`, `recurring_topics`, `emotional_patterns`, `what_to_expect`, `accentuations`) = concat → group by normalized key (lowercased trimmed `claim`/`topic`/`type`/`pattern`/`prediction`) → keep top-K (K=10) by evidence count → evidence unioned. A `big_five` dimension asserted by NO fragment → `NotAssessed()`. Confidence of a merged item = max across merged sources.

- [ ] **Step 1: Write failing tests:** (a) two fragments each assessing `openness` with scores 0.4 (1 evidence) and 0.8 (3 evidence) → weighted mean `0.7`, evidence union sorted; (b) a `big_five` dimension no fragment assessed → `Assessed==false && Score==nil`; (c) duplicate `recurring_topics` "работа" from two chunks → one entry, evidence unioned, weight averaged; (d) top-K caps a list at 10.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 9: Markdown render (`internal/domain/portrait/render.go`)

**Files:** Create `render.go`, `render_test.go`, `testdata/portrait_golden.md`.

**Interfaces:**
- Produces: `func RenderMarkdown(p Portrait) string`. MUST open with `p.Meta.Disclaimer`; MUST skip `TraitScore` entries where `!Assessed`; MUST render each shown claim with its cited quotes resolved via `p.EvidenceIndex`.

- [ ] **Step 1: Write a golden test** (mirror `cmd/parse/golden_test.go`): build a small `Portrait` with one assessed + one not-assessed OCEAN dim and a topic with two evidence IDs present in `EvidenceIndex`; render; compare to `testdata/portrait_golden.md`; assert the not-assessed dim string is absent and the disclaimer is line 1.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement + write the golden file.** **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 10: Analyzer adapter (`internal/infra/analyzellm`)

**Files:** Create `analyzellm.go`, `schema.go`, `analyzellm_test.go`.

**Interfaces:**
- Consumes: `llm.Client` (Task 5), `portrait` types (Task 1), `analyze.Analyzer` (Task 2).
- Produces: `func New(c *llm.Client, model string, repairRetries int) analyze.Analyzer`. `Analyze` builds the prompt from the domain catalogs (M1 fields only: `big_five`, `communication_style`, `recurring_topics`, `values_motivations`), the transcript from `chunk.Text`, and the M1 fragment JSON schema (`schema.go`); calls `llm.Chat` at temperature 0 with `response_format` + a large `max_tokens`; parses; **validates every evidence ID ∈ `chunk.MemberIDs`, dropping non-members**; a claim emptied by the guard is set `assessed:false` / dropped from lists; retries parse up to `repairRetries` by re-prompting with the error.

- [ ] **Step 1: Write failing tests** with a **fake `llm`-like transport** (inject an `httptest.Server` via `llm.New`): (a) a fragment citing an out-of-chunk ID → that ID is dropped and, if it was the only evidence, the claim is `assessed:false`/removed; (b) malformed JSON on the first call, valid on the retry → success; (c) the prompt text forbids scoring without a cited message (assert the built prompt contains the no-fabrication instruction — mirror `Test_buildPrompt_forbids_fabrication…`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (prompt builder pulls type lists from `portrait.OCEAN` etc.; schema in `schema.go` is a Go-built `json.RawMessage`). **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 11: AnalyzeService (`internal/app/analyze/service.go` + `assemble`)

**Files:** Create `service.go`, `internal/domain/portrait/assemble.go`, `service_test.go`, `assemble_test.go`.

**Interfaces:**
- Consumes: `Analyzer`, `Embedder`, `Index` ports; `portrait.ComputeStats/Segment/Reduce`; `kvcache`.
- Produces: `func NewAnalyzeService(a Analyzer, e Embedder, idx Index, cache *kvcache.Cache, opts AnalyzeOptions) *AnalyzeService`; `(*AnalyzeService).Run(ctx, msgs []dialogue.Message) (portrait.Portrait, error)`. And `portrait.Assemble(target string, stats Stats, person PersonPortrait, dyn *Dynamics, msgs []dialogue.Message, meta Meta) Portrait` which builds `EvidenceIndex` from every ID cited anywhere (looked up in `msgs`).

- [ ] **Step 1: Write failing tests (black-box, `package analyze_test`)** with fake `Analyzer`/`Embedder`/`Index`: (a) `Run` over a 3-chunk fixture calls the analyzer per chunk on the `--jobs` pool and Reduce assembles a portrait whose `evidence_index` contains exactly the cited IDs; (b) **determinism** — running with jobs=1 and jobs=4 yields byte-identical `json.Marshal(portrait)` (the collector writes a pre-sized `[]Fragment` by `ChunkIndex`); (c) a chunk whose analyzer errors is skipped and `meta.coverage` reflects it; (d) cache hit path skips the analyzer call.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** the worker pool (feeder → `opts.Jobs` workers → collector into `make([]portrait.Fragment, len(chunks))` indexed by `ChunkIndex`), cache lookup keyed `kvcache.Signature("map", opts.Model)` + `sha256(chunk.Text)`, then `Reduce` + `Assemble`. **Step 4: Run** — PASS. **Step 5: Vet + `go test -race ./internal/app/analyze/`.**

---

### Task 12: Q&A (`internal/app/analyze/qa.go`)

**Files:** Create `qa.go`, `qa_test.go`.

**Interfaces:**
- Produces: `func NewQAService(a *llm.Client, e Embedder, idx Index, collection string) *QAService`; `(*QAService).Ask(ctx, question string, k int) (answer string, cited []int, err error)`. Retrieves top-k chunks (embed question → `Index.Query` with a `where` filter scoping the analyzed chat/target), builds a grounded prompt (excerpts carry inline `[<id>]` per message), instructs answer-only-from-excerpts-or-refuse, then **validates cited IDs ⊆ union of retrieved chunks' member IDs**, stripping/refusing on violation.

- [ ] **Step 1: Write failing tests** with fakes: (a) answer citing a retrieved ID passes and returns that ID; (b) answer citing an ID absent from retrieved chunks → stripped, and if that leaves no support → refusal string; (c) empty retrieval → refusal without calling the LLM for a fabricated answer.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Vet.**

---

### Task 13: (folded into Task 11 `assemble.go`) — evidence index + meta

Covered by Task 11's `Assemble`. Acceptance: `evidence_index` keys are string IDs; each maps to `{from_id, date, text}` from the exact `msgs`; `meta.disclaimer == portrait.Disclaimer`; `meta.coverage == chunks_analyzed/chunks_total`.

---

### Task 14: CLI wiring (`cmd/analyze`)

**Files:** Create `cmd/analyze/main.go`, `portrait_cmd.go`, `ask_cmd.go`, `participants_cmd.go`, `flags_test.go`.

**Interfaces:**
- Consumes: everything above; `tgexport` reader; `dialogue`; `media` decoder (optional); `initLogger` (own copy, house style).

- [ ] **Step 1:** `main()` dispatches on `os.Args[1]` ∈ {`portrait`,`ask`,`participants`}; unknown/absent → usage to stderr, exit 2.
- [ ] **Step 2: Write `flags_test.go`** asserting per-verb `FlagSet` parses `--target`/`--llm-host`/... before the positional, and that a bare numeric `--target 123` is normalized to `user123` (reuse `dialogue.NormalizeOwnerID`, hoisted in this task from `cmd/parse` into `internal/domain/dialogue` — update `cmd/parse` to call it; run `go test ./cmd/parse/` as a regression gate).
- [ ] **Step 3: Implement** `portrait` (read→decode?→`ComputeStats` unused-target guard→`NewAnalyzeService.Run`→write `portrait.json` (fold JSON marshal+`os.WriteFile` here) + optional `--markdown`), `ask` (require export or `--index`; build index if needed; REPL loop over `QAService.Ask`; purge collection unless `--keep-index`), `participants` (read→`ComputeStats`→print `from_id  display_name  messages`).
- [ ] **Step 4:** Egress warning — if `--llm-host`/`--embed-host` is not localhost, `log.Warn().Msg("remote host: raw messages and inferences leave this device")`.
- [ ] **Step 5:** `go build ./... && go vet ./...`.

---

### Task 15: End-to-end smoke + `.gitignore`

- [ ] **Step 1:** Add to `.gitignore`: `.tg2llm_analysis_cache.db*`, `portrait.json`, `*.portrait.md`.
- [ ] **Step 2:** Build a tiny fixture export (reuse `internal/infra/tgexport/testdata/full_account_mini.json` shape) and run `analyze participants` — verify sender listing (no LLM needed).
- [ ] **Step 3:** Run `analyze portrait --llm-host <local> ...` against a running llama.cpp; verify `portrait.json` validates (stats populated, M1 inferred fields present, `evidence_index` covers cited IDs, no `assessed:false` field carries a non-null score). If no server is available, run the black-box service test with fakes as the smoke proof and note the live run as the remaining manual check.
- [ ] **Step 4:** `nix-shell shell.nix --run 'GOTOOLCHAIN=local go build ./... && go vet ./... && go test ./...'` — full suite green.

---

## M2 — full catalog (extends M1; contract already complete)

### Task 16: MAP prompt + schema add M2 fields
Extend `analyzellm` prompt/schema (Task 10) to also emit `accentuations` (enum = `portrait.AccentuationTypes`), `emotional_patterns`, `what_to_expect`. Tests: fragment with an accentuation whose `type` ∉ catalog is dropped; evidence guard still applies. Reduce (Task 8) already aggregates these lists — add a reduce test for accentuations top-K.

### Task 17: Dynamics pass (`karpman_roles`, `power_balance`)
Add `Analyzer.AnalyzeDynamics(ctx, chunk, a, b string) (Dynamics, error)` (new port method + adapter method + schema for the 3-role catalog), run it only when `opts.Relationship` and the chat is 1:1. Reduce dynamics with the same evidence-weighted rules. Tests: role `∉ KarpmanRoles` dropped; dynamics nil for non-1:1.

### Task 18: `relationship_arc`
Bucket chunks by `YYYY-MM`; per bucket, one summary claim aggregated from that window's fragments (Go groups; the per-chunk map already emits dated observations). Test: two months → two arc windows, each carrying only its window's evidence IDs.

### Task 19: Report + participants polish
Extend `RenderMarkdown` (Task 9) to render accentuation bars (text), Karpman tendencies, and the arc; extend the golden. Confirm `assessed:false` fields still suppressed.

---

## Self-Review

**Spec coverage:** stats §4.1→T6; portrait fields §4.2→T1/T10/T16-17; dynamics §4.3→T17-18; evidence_index §4.4→T11/T13; catalogs §4.5→T1; pipeline §5→T7/T10/T11/T12; structured output §5.2→T10; CLI §6→T14; packages §7→T1-14; concurrency/determinism §8→T11; errors §9→T10/T11; privacy/ethics §10→T5(auth)/T14(egress)/T4(delete)/T15(gitignore)/T9(suppression); testing §11→each task's tests + T11 -race; milestones §12→M1 T1-15 / M2 T16-19. No uncovered section.

**Placeholder scan:** no TBD/TODO; every type referenced (Portrait, Fragment, Chunk, ports, llm/vector/kvcache signatures) is defined in Tasks 1-5. Task 13 explicitly folded into 11 (not a placeholder).

**Type consistency:** `ComputeStats(msgs, target, sessionGap)`, `Segment(msgs, sessionGap, chunkChars)`, `Reduce(target, frags)`, `Assemble(...)`, `Analyzer.Analyze(ctx, chunk, target)`, `Index.{Add,Query,Delete}`, `llm.{Chat,Embed}` — names/params identical across all referencing tasks. `TraitScore.Score` is `*float64` everywhere (nil = not assessed).
