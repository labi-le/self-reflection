# Psychological Analysis Engine — Design

- **Date:** 2026-07-18
- **Status:** Approved (design), revised after independent review; pending implementation plan
- **Scope of this doc:** the **analysis engine** only. The visual dashboard is a
  separate sub-project with its own spec, designed later on top of this engine's
  output contract.

## 0. Revisions after independent review

Three independent reviewers (architecture, LLM-pipeline, product/ethics) vetted
the first draft. Accepted, load-bearing changes folded into this revision:

- **Evidence integrity is the core guarantee.** The LLM is asked to cite real
  message IDs; LLMs fabricate integer IDs, so every cited ID is **validated
  against the chunk's known member-ID set** and non-members are dropped. A claim
  left with no evidence becomes `not_assessed` (never a fabricated score).
- **Reduce is pure Go, not an LLM pass.** The LLM runs only in the per-chunk MAP
  phase. Merging fragments into the final portrait is deterministic Go
  (evidence-weighted aggregation, dedupe, top-K). This removes multi-level
  evidence hallucination and makes the output reproducible and testable.
- **New neutral infra, no cross-subsystem coupling.** `openaiExtractor` cannot
  carry a key/`response_format`/schema and caps output at 400 tokens, and the
  Chroma client is unexported inside the bot subsystem. So we hoist neutral,
  exported packages (`internal/infra/llm`, `internal/infra/vector`,
  `internal/infra/kvcache`) used by both the bot and the analysis engine —
  refactors that **preserve the bot's public API** (gated on its tests).
- **portrait.json is self-contained.** A Go-built `evidence_index`
  (`id → {from_id, date, text}`) lets the report/dashboard render every cited
  quote without re-reading the export.
- **Privacy beyond git.** A non-localhost host ships intimate content
  off-device; the CLI warns, and the vector store gets a delete verb +
  `--purge-index`.
- **Milestones, full contract.** The JSON contract is complete from day one;
  work is sequenced M1 → M2 (below). Unpopulated fields emit as `not_assessed`.

## 1. Goal & context

`tg2llm` today cleans a Telegram export into LLM-friendly text. The user then
**manually** pastes that text into an LLM chat to reason about a person and to
ask reflective questions. This project removes the manual step: the tool talks
to an LLM itself and produces a **structured psychological portrait** plus an
**interactive Q&A** grounded in the real messages.

Two capabilities, one engine:

1. **One-shot portrait** — run a command, get a structured `portrait.json`
   (+ optional Markdown report) describing a participant and the pair dynamics.
2. **Interactive Q&A** — ask questions about the person/conversation and get
   answers that cite real messages, or refuse when unsupported.

The subject is configurable: a **specific participant** (`from_id`, owner or
interlocutor) **and** the **relationship dynamics** of a 1:1 pair.

## 2. Non-goals (explicitly out of scope)

- The visual dashboard / TUI / web frontend (separate spec, built after the
  engine; this engine's JSON is its input contract).
- Anything under `finetune/` (model training) — untouched.
- The persona bot's **public behavior/contracts** (`internal/bot` runner /
  brain / persona / `memory.RAG` API). We may refactor bot internals to consume
  the new neutral infra packages, but only in a way that keeps the bot's public
  API and its tests green; if a refactor risks the bot, we fall back to
  analysis-local infra instead.
- No new Go **module** dependencies. `go.mod` has exactly two direct deps
  (`rs/zerolog`, `modernc.org/sqlite`). Chroma and the embedding server are
  external HTTP services reached by hand-rolled REST clients already in the repo.

## 3. Reused / hoisted infrastructure (verified on current tree)

- **Export reader:** `internal/infra/tgexport` — single-chat *and* full-account
  exports (`shape.go`: `isFullAccount`, `fullAccountChats`, owner detection
  `detectOwnerFullAccount` / `infer1on1Owner`).
- **Domain message:** `internal/domain/dialogue/message.go` — `Message` carries
  `ID, Date, FromID, Sender, ExtractedText, ReplyToMsgID, ForwardedFrom(ID),
  HasPhoto, MediaType, StickerEmoji, ChatID, ChatType`.
- **Cleaning:** `internal/domain/dialogue/text.go` (`CleanText`, `Caption`,
  `IsUseless`); optional media decode via `internal/infra/media`.
- **LLM seam — modeled on, not reused:** `internal/bot/profile/extractor.go`'s
  `openaiExtractor` shows the request shape, but it has **no key/Authorization**,
  **no `response_format`**, caps output at `defaultMaxTokens=400`, and its
  `Extract(ctx, prompt) (string, error)` signature cannot carry a schema. So the
  analyzer is a **new client** in a neutral package (below), modeled on it.
- **Structured-output precedent:** none in-repo (no `response_format` / grammar /
  JSON-repair today). This engine introduces it; the only retry precedent is the
  llamacpp PII regeneration (different purpose).
- **Hoisted neutral packages** (new homes for existing, currently-trapped code):
  - `internal/infra/kvcache` — the generic content-hash SQLite KV store
    currently in `internal/infra/media/cache.go` (`NewCache/Get/Put/Close`, WAL,
    `MaxOpenConns(1)`, nil-safe, resume-safe) + `Signature` from
    `media/signature.go`. `media` re-points to it; nothing media-specific but the
    table name (made a parameter).
  - `internal/infra/vector` — the Chroma v2 REST client currently unexported as
    `chromaStore` in `internal/bot/memory/chroma.go`, hoisted and exported, with
    an added `DeleteCollection` verb. `internal/bot/memory` refactors to wrap it,
    keeping `memory.RAG` / `Embedder` public API unchanged.
  - `internal/infra/llm` — new OpenAI-compatible client: `host+model+key`
    (bearer auth), `response_format` (json_schema / json_object), configurable
    `max_tokens`, `temperature`, plus an embeddings call. Used by the analyzer
    and the embedder; `profile`/`bot` may adopt it later (out of scope now).
- **Worker-pool shape:** `internal/app/service.go` `decodeAll` (feeder →
  `--jobs` workers → single lock-free collector). Note: its determinism comes
  from the *caller* re-ordering by message ID, so the analysis collector writes
  into a **pre-sized `[]Fragment` indexed by chunk position** (§8).

## 4. Domain model — the Portrait (the contract)

The engine emits **strictly structured JSON**. Every **LLM-inferred** field
carries `confidence` (`low|medium|high`) and `evidence` (validated message IDs).
A field with no surviving evidence is emitted as `assessed:false` /
`score:null` — never a fabricated number. Deterministic stats carry no
confidence — they are exact.

The **model never emits the dynamic-keyed top-level map** (unexpressible under a
strict schema with `additionalProperties:false`): the LLM returns a **flat,
single-subject** fragment; Go assembles the keyed maps and the `evidence_index`.

```jsonc
{
  "meta": {
    "model": "…", "embed_model": "…",
    "date_range": { "from": "…", "to": "…" },
    "message_count": 12345,
    "chunks_total": 210, "chunks_analyzed": 208, "coverage": 0.990,
    "evidence_ids_dropped": 4,                 // failed the member-ID guard
    "generated_at": "RFC3339", "token_cost_estimate": 1830000,
    "disclaimer": "LLM-inferred hypotheses from chat text, not a clinical diagnosis. Scores are a relative lean within this conversation, not population-normed."
  },

  // 4.1 DETERMINISTIC (Go, no LLM, exact). Per participant + pairwise.
  "stats": {
    "per_sender": {
      "<from_id>": {
        "display_name": "…",
        "messages": 0, "chars": 0, "avg_msg_len": 0.0, "median_msg_len": 0,
        "questions_ratio": 0.0,
        "response_latency_median_s": 0,
        "initiations": 0, "conversation_enders": 0,
        "active_hours_histogram": [/* len 24 */],
        "weekday_histogram": [/* len 7 */],
        "emoji_top": [ { "emoji": "…", "count": 0 } ],
        "sticker_count": 0,
        "top_terms": [ { "term": "…", "count": 0 } ],
        "monologue_ratio": 0.0
      }
    },
    "pairwise": {                                // 1:1 only; null/omitted for groups
      "initiative_balance": 0.0,                 // -1 other .. +1 target
      "reciprocity": 0.0, "avg_reply_gap_s": 0,
      "silence_gaps": [ { "from": "…", "to": "…", "seconds": 0 } ]
    }
  },

  // 4.2 LLM-INFERRED portrait, per participant. Assembled from validated fragments.
  "portrait": {
    "<from_id>": {
      "big_five": {                              // OCEAN
        "openness":          { "assessed": true,  "score": 0.62, "confidence": "medium", "evidence": [12,88] },
        "conscientiousness": { "assessed": false, "score": null, "confidence": "low",    "evidence": [] },
        "extraversion":      { "assessed": true,  "score": 0.71, "confidence": "high",   "evidence": [3,9,40] },
        "agreeableness":     { "assessed": true,  "score": 0.55, "confidence": "low",    "evidence": [21] },
        "neuroticism":       { "assessed": false, "score": null, "confidence": "low",    "evidence": [] }
      },
      "accentuations": [ { "type": "hyperthymic", "intensity": 0.6, "confidence": "medium", "evidence": [5,7] } ],
      "communication_style": {
        "tone": "…", "formality": "…", "directness": "…", "humor": "…",
        "emotional_expressiveness": "…", "conflict_style": "…",
        "confidence": "medium", "evidence": [2,15]
      },
      "values_motivations": [ { "claim": "…", "confidence": "low",    "evidence": [] } ],
      "recurring_topics":   [ { "topic": "…", "weight": 0.0, "confidence": "medium", "evidence": [] } ],
      "emotional_patterns": [ { "pattern": "…", "triggers": ["…"], "confidence": "low", "evidence": [] } ],
      "what_to_expect":     [ { "prediction": "…", "confidence": "low", "evidence": [] } ]
    }
  },

  // 4.3 PAIR DYNAMICS (1:1 only)
  "dynamics": {
    "karpman_roles": { "<from_id>": [ { "role": "victim", "tendency": 0.0, "confidence": "low", "evidence": [] } ] },
    "power_balance":   { "claim": "…", "confidence": "low", "evidence": [] },
    "relationship_arc": [ { "window": "YYYY-MM", "summary": "…", "confidence": "low", "evidence": [] } ]
  },

  // 4.4 EVIDENCE INDEX — Go-built, self-contained. Every ID cited anywhere above.
  "evidence_index": {
    "12": { "from_id": "user123", "date": "RFC3339", "text": "…" }
  }
}
```

### 4.5 Framework catalogs (typed constants in `internal/domain/portrait`)

Single source of truth for prompt builders and the JSON schema/grammar.

- **Big Five / OCEAN** — openness, conscientiousness, extraversion,
  agreeableness, neuroticism.
  (https://en.wikipedia.org/wiki/Big_Five_personality_traits)
- **Accentuations — Leonhard–Shmishek, 10 types.**
  Character: demonstrative, pedantic, stuck, excitable.
  Temperament: hyperthymic, dysthymic, anxious, cyclothymic, affective_exalted,
  emotive. (https://psycho-tests.com/test/leonhard-shmishek)
- **Karpman drama triangle — 3 roles:** victim, rescuer, persecutor; roles
  shift. (https://karpmandramatriangle.com/ ,
  https://en.wikipedia.org/wiki/Karpman_drama_triangle)

Validity note baked into the product: text-based trait inference is
**screening-grade, not clinical**; Karpman is descriptive, not a psychometric
instrument. Hence `assessed`/`evidence`/`confidence` gating, the JSON
`meta.disclaimer`, and qualitative labelling in the report.

## 5. Pipeline

Uniform path; the "fast path" is just the single-chunk case.

```
read (tgexport) ─► clean (+ optional media decode)  [message IDs preserved]
     │
     ├─► deterministic stats  (Go, pure, exact)                      ─┐
     │                                                                │
     └─► segment → chunks (session-gap; split oversize; pack to budget)│
              │                                                        │
              ├─► MAP  (LLM per chunk, temp 0, schema-constrained)     │
              │      → parse → VALIDATE evidence ⊆ chunk member-IDs    │
              │      → Fragment (pre-sized slice by chunk index)       │
              │                                                        │
              ├─► REDUCE (pure Go: aggregate scores evidence-weighted, │
              │           dedupe/top-K claims, suppress no-evidence)   │
              │                                                        ▼
              ├─► embedding index (vector, per-analysis collection)  assemble Portrait
              │        (chunks carry inline per-message IDs)           + evidence_index (Go)
              │                                                        │
              │                                            ┌───────────┴───────────┐
              │                                       portrait.json         Markdown report
              └────────────────────────────────────────► Q&A (retrieve → validate cites → answer/refuse)
```

### 5.1 Stages

1. **Read** (single-chat or full-account); resolve owner; select target/chat.
2. **Clean** (+ optional `--decode-media`); message IDs preserved (evidence).
3. **Deterministic stats** — pure `portrait.ComputeStats([]Message, target)`.
   Pairwise/dynamics only for 1:1 (guarded; skipped-with-log for groups).
4. **Segment** — group into sessions by `--session-gap` (default 30m); **hard-
   split any session that alone exceeds the budget** at message boundaries;
   pack sessions into chunks up to `chunk_chars = (model_ctx − prompt −
   max_output) × chars_per_token`, with `chars_per_token ≈ 2` for Russian
   (the existing `chatmlsink` `runes/4` heuristic under-counts Cyrillic ~2–3×).
   Each chunk records its member message IDs.
5. **Map** — `Analyzer.Analyze(ctx, chunk)` → one flat fragment as
   schema-constrained JSON at **temperature 0** with a large, configurable
   `max_tokens`. Runs on the `--jobs` worker pool. The adapter then
   **validates every `evidence` ID against the chunk's member set**, drops
   non-members, and marks any claim thus emptied `assessed:false`.
6. **Reduce** — `portrait.Reduce([]Fragment)` in **pure Go**: numeric fields =
   evidence-weighted mean; list fields = concat → dedupe → rank by evidence
   count → top-K; a field with zero surviving evidence → `not_assessed`.
   Deterministic and unit-testable. (No LLM here.)
7. **Assemble** — Go wraps fragments under `from_id` keys and builds
   `evidence_index` from the exact messages fed to the model.
8. **Embedding index** — embed chunks into a per-analysis `vector` collection
   (namespaced by export hash); each chunk **document embeds its per-message
   IDs inline**. Optional: without an embedder, Q&A falls back to keyword search
   over chunks in the cache DB.
9. **Emit** — service writes `portrait.json`; `--markdown` renders a report via
   `portrait.RenderMarkdown` (pure, golden-tested).
10. **Q&A** — retrieve top-k chunks (semantic, else keyword), scoped by the
    vector `where` filter to the analyzed chat/target; inject portrait summary +
    stats; the model answers **only** from excerpts and cites message IDs. Cited
    IDs are **validated ⊆ retrieved chunks' member sets**; unsupported citations
    force the refusal path.

### 5.2 Structured-output strategy

- Prefer server-enforced structure via `response_format` (`json_schema`, else
  `json_object` + a GBNF grammar) on llama.cpp; the prompt also **describes the
  shape in words** (the server does not inject the schema) and forbids scoring
  without a cited message (rule modeled on `profile.go` `buildPrompt`).
  (https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- Always **Go-validate** the parsed JSON independent of server support (generic
  OpenAI endpoints may ignore or reject `response_format`). On parse failure,
  re-prompt with the error up to `--repair-retries` (default 1). If a **single-
  chunk fast path** still fails, **fall back to splitting into smaller chunks
  and map/reduce** rather than emitting nothing.

## 6. CLI surface

New binary `cmd/analyze`. **Dispatch on `args[0]`** (the verb), then parse
`args[1:]` with a **per-verb `flag.FlagSet`** — the flags-before-positionals
gotcha applies *within* each subcommand (a single flat `FlagSet` over `os.Args`
would treat the verb as a positional and parse zero flags).

- `analyze portrait [flags] <export.json>` → `portrait.json` (+ `--markdown`).
- `analyze ask [flags] <export.json>` → REPL / `--question`. Requires the export
  (or `--index`/`--cache`) to ground answers; given only a `portrait.json` it
  answers solely from the embedded `evidence_index` and says so.
- `analyze participants <export.json>` → prints `from_id → display_name,
  messages` (from stats) so users can discover valid `--target`/`--with` ids.

| flag | default | meaning |
|---|---|---|
| `--target <from_id>` | inferred interlocutor | subject; bare numeric id `user`-prefixed (`normalizeOwnerID` semantics) |
| `--with <from_id>` | inferred | the second party for dynamics (disambiguates groups) |
| `--chat <chat_id>` | inferred | pin a chat in a full-account export |
| `--relationship` | off | also produce 1:1 pair dynamics |
| `--llm-host` | `http://localhost:8080` | OpenAI-compatible base URL |
| `--llm-model` | `default` | model name (salts the cache) |
| `--llm-key` | `""` | bearer key (cloud); empty for local |
| `--embed-host` | `""` | embedding server; empty ⇒ Q&A keyword fallback |
| `--jobs` | `3` | map-phase workers |
| `--cache` | `<export-dir>/.tg2llm_analysis_cache.db` | fragment cache |
| `--session-gap` | `30m` | silence gap starting a new session |
| `--repair-retries` | `1` | structured-output repair attempts |
| `--decode-media` | `false` | transcribe voice / describe photos first (extra egress) |
| `--date-from` / `--date-to` | `""` | analysis window |
| `--purge-index` / `--keep-index` | purge | drop / retain the vector collection after `ask` |
| `--verbose` | `false` | trace logging |

## 7. Package layout & ports

```
cmd/analyze/main.go                verb dispatch (portrait|ask|participants),
                                   per-verb FlagSet, wiring, initLogger (own copy,
                                   house style), --llm-host egress warning.
internal/domain/portrait/          PURE, no I/O, no ports:
  portrait.go                        Portrait/Stats/Fragment types + catalog consts.
  stats.go                           ComputeStats([]Message, target) Stats.
  reduce.go                          Reduce([]Fragment) Portrait  (deterministic).
  render.go                          RenderMarkdown(Portrait) string (golden-tested).
internal/app/analyze/
  ports.go                           Analyzer (Analyze/MapChunk), Embedder, Index;
                                     AnalyzeOptions. (No StatsComputer/Reduce port —
                                     those are pure domain funcs.)
  service.go                         AnalyzeService.Run: stats → segment → map
                                     (pool, ordered slice) → reduce → assemble →
                                     index → write JSON.
  qa.go                              QAService.Ask: retrieve → validate cites → answer.
internal/infra/llm/                  NEW OpenAI-compatible client (host+model+key,
                                     response_format, max_tokens, temperature,
                                     chat + embeddings). Wrapped by the analyzer
                                     and the embedder adapters.
internal/infra/vector/               Chroma v2 REST client hoisted from bot/memory
                                     (getOrCreate/add/query + DeleteCollection).
internal/infra/kvcache/              content-hash SQLite KV + Signature hoisted
                                     from media (table name parameterized).
internal/infra/analyzellm/           Analyzer adapter: prompt build (domain catalog),
                                     llm client call, parse + Go-validate + evidence
                                     member-ID guard + repair retry.
```

Contract-preserving refactors (gated on their tests): `internal/infra/media`
re-points its cache/signature to `kvcache`; `internal/bot/memory` wraps
`internal/infra/vector` internally (public `RAG`/`Embedder` API unchanged).

## 8. Concurrency, caching, determinism

- Map reuses the `decodeAll` pool shape, but the collector writes into a
  **pre-sized `[]Fragment` indexed by chunk position** (Go map iteration is
  randomized; `decodeAll`'s order comes from the caller, which Reduce lacks), so
  Reduce consumes a deterministic order.
- **Reduce is pure Go**, so the assembled `portrait.json` is byte-identical
  given the same set of fragments — the Markdown golden test is stable.
- LLM map calls use **temperature 0**; still, llama.cpp parallel slots and cloud
  endpoints are not bit-reproducible. Honest claim: **reproducibility comes from
  the fragment cache**, not from the model; the assemble/reduce pass is
  deterministic.
- Cache: `kvcache` (SQLite, WAL, `MaxOpenConns(1)`, nil-safe, resume-safe); key
  = `sha256(chunk-text) | model | Signature(prompt-kind, typology)`; empty/failed
  results are not cached (reruns retry only those); each success `Put`
  immediately.

## 9. Error handling

Philosophy matches the media decoder: **backend failures are logged, never
fatal** (`.Warn().Err(...)`).

- A chunk whose map call fails after retries → skipped; `meta.coverage` records
  the shortfall; stats always succeed.
- Malformed JSON → Go-validate + `--repair-retries`; else drop fragment + log;
  single-chunk fast-path failure → split-and-retry, never empty output.
- Evidence IDs failing the member guard → dropped, counted in
  `meta.evidence_ids_dropped`.
- Embedder unavailable → Q&A keyword fallback (no crash).
- **Log hygiene:** logs carry only IDs, counts, and error strings — **never
  message content or raw model replies**. Any content dump is gated behind
  `--verbose` at trace level (matches the media components' content-free logs).
- Hard errors: reading the export, writing the output file.

## 10. Privacy & ethics guardrails

- **Local git:** add global globs to `.gitignore` —
  `.tg2llm_analysis_cache.db*`, `portrait.json`, `*.portrait.md`; write outputs
  to predictable, pattern-matchable paths.
- **Network egress (bigger than git):** any non-localhost `--llm-host` /
  `--embed-host` transmits raw messages, quotes, and inferences about a real,
  non-consenting third party off-device (and may be retained). The CLI **prints
  a stderr warning** when a remote host is configured; `--decode-media` also
  routes voice/photo content through whisper/vision hosts (documented).
- **Vector retention:** the store keeps verbatim `documents`; gitignore cannot
  reach a networked server. `vector` gets `DeleteCollection`; `analyze ask`
  purges its per-analysis collection unless `--keep-index`.
- **Validity:** every inferred claim carries `assessed`/`confidence`/`evidence`;
  no-evidence claims are **suppressed** (`assessed:false`, `score:null`), not
  plotted as zero. `meta.disclaimer` ships **inside** `portrait.json` (the
  dashboard's contract), and the report opens with it. The map prompt **forbids
  scoring without a cited message** (rule modeled on `profile.go`). Scores are
  labelled a relative lean within this conversation, not population-normed.

## 11. Testing strategy

Fakes only, no live LLM (mirrors `profile_test`'s fake `Extractor`, black-box
`service_test`).

- **stats** — exact unit tests over fixtures (counts, latency, histograms,
  initiative balance); highest value, fully deterministic.
- **reduce** — evidence-weighted aggregation, dedupe/top-K, and that an
  empty-evidence trait is emitted `assessed:false` / `score:null`, never 0.0.
- **evidence guard** — a fragment citing an out-of-chunk ID has it dropped and
  the emptied claim downgraded; `meta.evidence_ids_dropped` counted.
- **prompt lock** — mirrors `Test_buildPrompt_forbids_fabrication…`: the map
  prompt forbids rating without a cited message.
- **assemble** — `evidence_index` covers every cited ID; flat-fragment → keyed
  map wrapping.
- **structured output** — parse + repair path; single-chunk fast-path failure
  falls back to split map/reduce.
- **Q&A** — fake retriever + LLM: answer cites only retrieved chunk member IDs
  and refuses when excerpts don't support it.
- **segmentation** — session-gap boundaries, oversize-session split, RU token
  budget.
- **Markdown render** — golden test asserting the disclaimer + suppression of
  not_assessed claims.
- **regression gates** — `internal/infra/media` and `internal/bot/memory` tests
  stay green after the kvcache/vector hoists.

## 12. Milestones (full contract from day one)

The JSON contract in §4 is complete immediately; unpopulated fields emit
`assessed:false`. Work sequences so the least-validatable frameworks come last:

- **M1 — end-to-end vertical slice:** neutral infra (llm/vector/kvcache) +
  reader/clean reuse → stats → segment → map/reduce → `evidence_index` →
  `portrait.json` + Markdown → Q&A. Populated fields: `stats`, `big_five`,
  `communication_style`, `recurring_topics`. `participants` subcommand.
- **M2 — full catalog:** `accentuations`, `karpman_roles`, `power_balance`,
  `what_to_expect`, `emotional_patterns`, `relationship_arc`, and the
  `--relationship` dynamics pass.

## 13. Sources

- Big Five / OCEAN — https://en.wikipedia.org/wiki/Big_Five_personality_traits
- Leonhard–Shmishek accentuations (10 types) — https://psycho-tests.com/test/leonhard-shmishek
- Lichko accentuations (11 types; considered, not chosen) — https://prismatest.com/en/tests/lichko
- Karpman drama triangle — https://karpmandramatriangle.com/ , https://en.wikipedia.org/wiki/Karpman_drama_triangle
- llama.cpp structured output (GBNF / json_schema / response_format) —
  https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md ,
  https://deepwiki.com/ggml-org/llama.cpp/7.3-grammar-and-structured-output

## 14. Future work (separate specs)

- **Visual dashboard** (local web, SVG/canvas) rendering `portrait.json`: OCEAN
  radar (skipping `assessed:false`), accentuation bars, Karpman-triangle view,
  activity heatmaps, `evidence_index`-backed drill-down.
- **Temporal monitoring**: re-run over rolling windows (the `relationship_arc`
  field is the seed).
