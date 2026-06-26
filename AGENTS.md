# AGENTS.md — tg2llm

Onboarding for AI agents. Read this first — it captures the verified project facts and the
environment quirks that otherwise cost a lot of trial-and-error. Every claim here was checked
against the source on the current tree (`go build/vet/test ./...` is green; `go version` =
`go1.26.3`).

## What this is

`tg2llm` converts a **Telegram / AyuGram JSON export** (`result.json`) into a clean,
LLM-friendly **plain-text dialogue**. Optionally it decodes media locally: **voice** →
transcript (whisper.cpp via `whisper-cli`), **photos** → description/OCR via an
**OpenAI-compatible vision server** (typically llama.cpp on a GPU).

- Language: **Go 1.26** (`go.mod` → `go 1.26`), module `tg2llm`, **hexagonal** (ports & adapters).
- Direct deps (only two): `modernc.org/sqlite` v1.53.0 (pure-Go, **no CGO**) + `github.com/rs/zerolog` v1.35.1.
- The base parser (no `--decode-media`) is runtime-dependency-free; media decoding shells out to
  `ffmpeg` + `whisper-cli` and talks HTTP to the vision server.
- **No Python** (the original reference was removed — do not re-introduce it) and **no ollama
  backend** (removed). The *only* vision backend is the OpenAI `/v1/chat/completions` one.
- Repo: `https://github.com/labi-le/self-reflection.git` (branch `master`).

## Environment quirks (READ — these save the most time)

- **Go runs only via Nix.** Always:
  `nix-shell shell.nix --run 'GOTOOLCHAIN=local go ...'` (Go is `go1.26.3`).
  Always set `GOTOOLCHAIN=local` or it tries to fetch a toolchain. (The `shellHook` exports it,
  but set it explicitly in one-shot `--run` invocations.)
- **CLI gotcha (document everywhere):** Go's stdlib `flag` parser stops at the first non-flag
  arg, so **all `--flags` must come BEFORE the positional `<input.json> [output.txt]`.**
- **chroma-gate plugin** blocks `grep`/`glob` until you first call `chroma_query_documents`
  (collection `code-tg2llm`). `Read`/`Glob`-after-chroma are fine; **`Read` is not gated** — just
  read files directly when you know the path. Don't bypass chroma with `rg`/`find` in Bash.
- **`no-tail` rule** blocks piping through `tail`/`head` (data loss). Use `docker logs --since`,
  or capture to a file and `Read` it.
- **`tmux capture-pane` is blocked** inside the interactive_bash tool — use the Bash tool for it.
- **gopls is flaky/stale** on cross-file new symbols — **trust `go build`/`go vet`**, not the
  inline LSP, after refactors.
- **Dev-shell aliases** (banner on `nix-shell` entry): `parse`, `parse-media`, `gotest`,
  `fetch-models`, `vision-up` / `vision-logs` / `vision-down`, `tg-build` / `tg-run`.

## Build / test / vet / race (exact)

```sh
# build + vet + test (green on current tree)
nix-shell shell.nix --run 'GOTOOLCHAIN=local go build ./... && go vet ./... && go test ./...'

# race detector (works — the nix shell provides a C compiler; verified on these two pkgs)
nix-shell shell.nix --run 'GOTOOLCHAIN=local go test -race ./internal/app ./internal/infra/media'

# inside the dev shell, `gotest` == `go test ./...`
```

Only `internal/app`, `internal/domain/dialogue`, `internal/infra/media` have tests; the other
packages report `[no test files]`.

## Architecture map (real paths + responsibility)

```
cmd/parse/main.go                 CLI entrypoint: flag parsing, ~ expansion of --whisper-model,
                                  cache-path defaulting, initLogger (zerolog console; --verbose
                                  => trace + caller basename:line), wires source/sink/decoder.
internal/domain/dialogue/         core, no I/O:
  message.go                      Message entity (ID, Date, Sender, ExtractedText, ReplyToMsgID,
                                  ForwardedFrom[ID], HasPhoto, MediaType, StickerEmoji,
                                  PhotoPath, FilePath).
  text.go                         CleanText / Caption / IsUseless (+ useless placeholders,
                                  sticker & bare-URL regexes).
  daterange.go                    DateRange{From,To}, Contains, ParseBound.
  message_log.go                  Message.MarshalZerologObject (structured logging).
internal/app/
  ports.go                        MessageSource / DialogueSink / MediaDecoder interfaces;
                                  ParseOptions (+ its MarshalZerologObject).
  service.go                      ParseService.Run: read -> decodeAll (worker pool) ->
                                  ordered single-pass write (clean/decode/forward/drop/headers).
internal/infra/tgexport/reader.go read the Telegram/AyuGram JSON export (object-with-"messages"
                                  or bare array; polymorphic text field; referenced-IDs map).
internal/infra/textsink/writer.go write the dialogue text (day headers + message lines).
internal/infra/media/
  cache.go                        SQLite cache (WAL, single conn, INSERT-or-replace, nil-safe).
  decoder.go                      Decoder (port impl), content-hash keying, singleflight
                                  (decodeOnce), BuildDefaultDecoder, DEFAULT_VISION_PROMPT (RU),
                                  NOT_INCLUDED guard.
  whisper.go                      Signature (SHA-1[:10] of parts joined by "|") + OneLine.
  whisper_transcriber.go          Transcriber: ffmpeg -> 16kHz mono WAV -> whisper-cli.
  openai_describer.go             OpenAIDescriber: the ONLY vision backend (/v1/chat/completions).
pkg/ctxlog/context.go             ctxlog.Op(logger, "Type.Method") = operation-scoped zerolog.
docker/                           GPU vision (llama.cpp:server-vulkan) + tg2llm utility image.
```

Tests live beside the code: `dialogue_test.go` and `media_test.go` are **white-box** (same
package); `service_test.go` is **black-box** (`package app_test`). Run with `gotest`.

## CLI flags

`parse [flags] <input.json> [output.txt]` — `output.txt` defaults to **`parsed_dialogue.txt`**.
**All `--flags` must precede the positional paths.** Verified against `cmd/parse/main.go`.

| flag | default | help / notes |
|------|---------|--------------|
| `--decode-media` | `false` | transcribe voice + describe photos using local models (enables both) |
| `--whisper-model` | `""` | path to a whisper.cpp GGUF model; **required for voice** (else voice stays a placeholder + a warning). Leading `~` is expanded to `$HOME`. |
| `--whisper-bin` | `whisper-cli` | whisper.cpp binary (`pkgs.whisper-cpp` ships **`whisper-cli`**, not `whisper-cpp`) |
| `--whisper-lang` | `ru` | spoken language for transcription |
| `--vision-model` | `qwen2.5vl:7b` | model name sent in the request (llama.cpp serves whatever it loaded, but the name still salts the cache) |
| `--vision-host` | `http://localhost:8080` | vision server base URL; POSTs to `<host>/v1/chat/completions` |
| `--vision-prompt` | `""` → built-in RU prompt | prompt sent to the vision model (describe + OCR + explain memes, one paragraph) |
| `--media-cache` | `""` → `<export-dir>/.tg2llm_media_cache.db` | SQLite cache path (`<export-dir>` = dir of the abs input path) |
| `--jobs` | `3` | parallel media-decode workers (`1` = sequential) |
| `--date-from` | `""` | skip messages before this date (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`) |
| `--date-to` | `""` | skip messages after this date; **date-only** bound = end of that day (23:59:59 UTC) |
| `--verbose` | `false` | trace-level logging + caller (`basename:line`) |

There is **no** `--vision-api`, **no** `--vision-num-ctx`, **no** `--ollama-host` (all removed).
The vision context window is a **server-side** setting (`-c` in the llama.cpp command), not a
per-request flag.

## Media decoding

`Decoder.Decode` dispatches by message shape (`internal/infra/media/decoder.go`):

- **Photo** — only if `HasPhoto && describeFn != nil` → `decodePath(PhotoPath, …, "photo")`.
  `OpenAIDescriber.Describe`: read file → base64 → POST `<host>/v1/chat/completions` with
  `messages:[{role:user, content:[{type:text,text:prompt},{type:image_url,image_url:{url:"data:image/jpeg;base64,…"}}]}]`,
  `temperature:0`, `stream:false`; returns `OneLine(choices[0].message.content)`. Client timeout 600s.
- **Voice** — only if `MediaType == "voice_message" && transcribeFn != nil` →
  `decodePath(FilePath, …, "voice")`. `Transcriber.Transcribe`:
  `ffmpeg -nostdin -y -i <in> -ar 16000 -ac 1 -f wav <wav>` then
  `whisper-cli -m <model> -f <wav> -l <lang> -nt -otxt -of <out>`, read `<out>.txt` → `OneLine`.
  Each step has a 600s timeout context. Non-speech clips correctly yield whisper annotations
  like `*звук*` / `[музыка]`.
- **Skipped** (no decode branch): stickers, videos, audio files, generic files.
- A `rel` path that is empty or contains `File not included` (`NOT_INCLUDED`) is skipped before
  hashing.

Backend errors are **logged, never fatal** (`.Warn().Err(...)`, visible with `--verbose`); the
message is left undecoded and the run continues. The vision server's context size is set
server-side (`-c 16384` in compose), so there is no per-request `num_ctx`.

## SQLite cache (`internal/infra/media/cache.go` + `decoder.go`)

- DB at `<export-dir>/.tg2llm_media_cache.db`; schema `media_cache(key TEXT PRIMARY KEY, value TEXT)`.
  Opened with `journal_mode=WAL`, `busy_timeout=5000`, `MaxOpenConns(1)`; `Close()` does a
  `wal_checkpoint(TRUNCATE)`. `NewCache("")` and all methods are **nil-safe** (decode just runs uncached).
- **Key** = `kind:sha256(file-content)[:signature]`, where `kind` ∈ {`photo`,`voice`} and
  `signature = Signature(...)` = first 10 hex of SHA-1 of the parts joined by `|`:
  photo → `Signature(visionModel, visionPrompt)`, voice → `Signature(basename(whisperModel), whisperLang)`.
- **Content-hash dedup**: identical media (even under a different path) is decoded once; changing
  the model/prompt/lang changes the signature → re-decode (no stale hits).
- **Failed/empty results are NOT cached** (the trimmed result must be non-empty) → a rerun retries
  only those. Each success is `Put` immediately (durable), so a killed run keeps its progress.

## Parallelism + ordering

`service.decodeAll` is a worker pool sized by `--jobs` (`<=1` ⇒ sequential; default 3): a feeder
goroutine pushes in-range messages onto `jobCh`, workers decode, results land on `resCh`, and a
single collecting goroutine is the sole writer of the `map[int]string` (no locking needed).
`decoder.decodeOnce` is a hand-rolled **singleflight** (`inflight map` + per-key `WaitGroup`) that
collapses concurrent identical-key decodes onto one backend call. **Decode is parallel; the
assemble/write pass is separate and sequential**, so output is **byte-identical regardless of
`--jobs`** (locked in by `TestServiceParallelMatchesSequential`, run under `-race`).

## Docker / GPU vision (`docker/`)

- **`vision`** = `ghcr.io/ggml-org/llama.cpp:server-vulkan` on an AMD GPU via Vulkan
  (`/dev/kfd` + `/dev/dri`, `seccomp=unconfined`, `--device Vulkan0 -ngl 99`), port 8080,
  `mem_limit`/`memswap_limit: 20g` (defensive host-RAM cap — host has no swap).
  Command: `-m …/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj …/mmproj-…-f16.gguf
  --host 0.0.0.0 --port 8080 --device Vulkan0 -ngl 99 -c 16384 -np 1 --flash-attn on`.
- **`-np 1` is load-bearing.** Without it llama.cpp auto-picks `n_parallel=4`; concurrent
  multi-megapixel images then exhaust the unified KV cache (VRAM KV-slot exhaustion — a
  llama.cpp "failed to find a memory slot for batch of size N" crash). `-c 16384` gives one
  large image headroom.
- **`tg2llm`** service = utility image (Go binary + `whisper-cli` + `ffmpeg`, 3-stage Dockerfile,
  `CGO_ENABLED=0`) that runs the parser with `--vision-host http://vision:8080` and a bind-mounted
  whisper model. `mem_limit: 4g`.
- **Makefile** (`make up | logs | build | run | down | status`) uses `-f docker-compose.yml`
  **relative**, so run it from `docker/`. The dev-shell `vision-*` / `tg-*` aliases wrap compose
  by absolute path and work from anywhere. Overridables: `MODELS_DIR`
  (`/mnt/ssd2tb/llm/models`), `EXPORT_DIR`, `WHISPER_MODEL`.
- **Models** (into `${MODELS_DIR:-/mnt/ssd2tb/llm/models}`): `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
  (~4.7 GB) + `mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf` (~1.4 GB) from
  `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`. Download with **`fetch-models`** (uses the `hf`
  HuggingFace CLI + `hf-xet`). Do **NOT** use `aria2c` (its multi-range splitting 403s on HF's
  Xet per-range signed URLs) or plain `curl` (slow, single-stream).

## Output format (`textsink/writer.go` + `app/service.go`)

```
\n=== YYYY-MM-DD ===          (day header — note the leading blank line)
[<id>] [HH:MM:SS] <sender>: <text>
[<id>] [HH:MM:SS] <sender> (reply to <id>): ...
[<id>] [HH:MM:SS] <sender>: [Forwarded from <name>]: ...
[<id>] [HH:MM:SS] <sender>: <caption> [photo: <description>]
[<id>] [HH:MM:SS] <sender>: [voice_message: <transcript>]
```

- Time is `HH:MM:SS`; reply suffix is ` (reply to N)`; forwards prefix the text with
  `[Forwarded from <name>]: ` (`<name>` = `forwarded_from` or `Unknown`).
- Decoded-media tag is `[<label>: <decoded>]` where `label` = `MediaType`, else `photo`
  (when `HasPhoto`), else `media`; a caption is kept and combined as `caption + " " + tag`.
- **Drop rule:** a message whose final text `IsUseless` (empty, bare placeholder like `[photo]`,
  `[… Sticker]`, or a bare URL) is dropped **unless** its ID is referenced by some
  `reply_to_message_id`.

## Conventions

- **Logging:** `rs/zerolog`; operation scope via `ctxlog.Op(logger, "Type.Method")`; component
  sub-loggers (`media-decoder`, `media-cache`, `media-describer`, `media-transcriber`);
  `MarshalZerologObject` on domain types (`Message`, `ParseOptions`); `--verbose` ⇒ trace + caller.
- **Comments:** a hook flags *new* comments — write self-documenting code, keep only necessary
  ones (public-API docs, non-obvious concurrency/algorithm rationale) and be ready to justify them.
- **Git:** plain imperative commit subjects (no semantic prefix); **subject-only — no footer, no
  `Co-authored-by`** (standing owner instruction). **Commit/push only when explicitly asked.**

<mcp_instructions>
  <server name="context7">
    Use this server to fetch current documentation whenever the user asks about a library,
    framework, SDK, API, CLI tool, or cloud service — even well-known ones. Prefer it over web
    search for library docs. Do not use it for refactoring, writing scripts from scratch,
    debugging business logic, code review, or general programming concepts.
  </server>
</mcp_instructions>
