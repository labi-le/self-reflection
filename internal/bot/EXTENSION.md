# Persona-bot extension points

Design note for two planned increments. **Nothing here is implemented or wired
into the live bot** — this file, plus the `Memory` port and the `future:`
schema comment in `memory.go`, only fix the seams so later work slots in without
reshaping the core. The core (`internal/bot`) stays stdlib-only; every concrete
adapter lives in a sub-package and is wired in `cmd/bot`.

## RAG / Memory — IMPLEMENTED (`internal/bot/memory`)

The persona recalls relevant older turns of *the current chat* without stuffing
the whole history into the context window. Shipped as the `bot.Memory` adapter in
`internal/bot/memory`, wired in `cmd/bot` behind `--embed-host` + `--chroma-host`
(both required; either missing disables memory and the bot falls back to its
recent window alone).

Pipeline (all on the LAN, no data leaves the host network):

1. **Embed** — each turn is POSTed to a **bge-m3 embeddings server** (llama.cpp
   `--embedding --pooling cls` on CPU; the `embed` service in `docker/`, port
   8092), which returns a 1024-dim vector.
2. **Remember** — on every inbound and outbound turn the runner stores
   `(document=text, metadata={chat_id, role, ts})` in the **ChromaDB** collection
   `persona-memory` (cosine space), keyed by a content hash so identical turns
   dedup.
3. **Recall** — on an inbound turn the runner embeds the message and queries
   Chroma for the top-`--memory-k` nearest vectors **filtered to `chat_id`**, then
   re-checks each hit's `chat_id` in Go. That per-chat scope, enforced in both the
   Chroma `where` and the `Chunk.ChatID` re-check, is the privacy contract:
   memory from one conversation is never surfaced in another.
4. **Inject** — the recalled fragments are prepended as one scoped `system` block
   ahead of the persona reply.

Memory is **best-effort**: an embed or Chroma failure is logged and the bot still
replies from its recent window, so the backends being down never blocks a turn.
`nopMemory` is the "disabled" default when the flags are unset.

## MTProto (user-account transport)

Goal: run the persona over a real Telegram **user** account (MTProto) instead of
(or alongside) the Bot API.

Shape: it implements the **same `bot.ChatTransport` interface** — `Updates` +
`Send` — so the core bot loop is transport-agnostic and needs no change. It is
gated by the existing **`allowed_ids`** allowlist (owner always allowed), exactly
like the Bot API path.

Because automating a user account carries **Telegram ToS / ban risk**, this
transport ships behind hard guardrails, all off by default:

- **Kill-switch env var** — a single env flag disables all sending instantly; if
  unset/false the transport refuses to send.
- **DRAFT mode** — default: generate the reply and log/stage it but **do not
  send**, so behavior can be reviewed before going live.
- **Explicit ToS-risk acknowledgment** — sending requires an explicit opt-in
  (env/config), never implicit.
- **Global rate-limit** — cap outbound messages/minute across all peers.
- **Per-peer cooldown** — minimum interval between messages to the same peer.
- **Human-like typing delay** — pause proportional to reply length (with a
  typing action) instead of instant replies.
- **Full audit log** — every inbound trigger and every outbound (or would-be
  outbound, in DRAFT) message is recorded for review.

No MTProto library is a dependency today: **`go.mod`/`go.sum` are unchanged** and
`gotd/td` (or any alternative) is added only when this adapter is actually built.
