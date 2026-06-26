# tg2llm

Convert a Telegram / AyuGram JSON export into a clean, LLM-friendly plain-text dialogue — with optional **local** media decoding: voice messages are transcribed (whisper.cpp) and photos are described (a vision model), so the whole conversation becomes readable text.

Written in Go (hexagonal architecture), stdlib-first; the only third-party dependencies are a pure-Go SQLite driver and zerolog.

## Features

- **Plain-text dialogue** from `result.json` with per-day headers, timestamps, replies and forwards.
- **Voice transcription** via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`whisper-cli`, runs locally).
- **Photo description / OCR** via a vision model on any **OpenAI-compatible** server (e.g. [llama.cpp](https://github.com/ggml-org/llama.cpp) `server`, which can run on an AMD/Intel GPU via Vulkan).
- **SQLite cache** keyed by file **content hash** (sha256): identical media is decoded once; reruns skip already-decoded media; failed/empty results are *not* cached, so a rerun retries only them.
- **Parallel decoding** (`--jobs`) with a worker pool + in-flight de-duplication; output order is always preserved.
- **Date filtering** (`--date-from` / `--date-to`) to skip the beginning (or end) of a long chat.
- **Docker deployment** for GPU-accelerated photo description (llama.cpp + Vulkan).
- Everything runs **offline / locally** — no data leaves your machine.

## Output format

```
=== 2026-06-19 ===
[101] [09:15:04] Alice: are you free this afternoon?
[102] [09:16:10] Bob: [Forwarded from Carol]: meeting moved to 4pm
[103] [09:20:29] Bob: here's the new layout [photo: A wireframe with a header, a sidebar on the left and a content area on the right ...]
[104] [09:22:39] Alice (reply to 103): [voice_message: Looks good, let's ship it ...]
```

- Day separators: `=== YYYY-MM-DD ===`
- Messages: `[id] [HH:MM:SS] sender: text`
- Replies: `sender (reply to <id>)`
- Forwards: `[Forwarded from <name>]: ...`
- Decoded media: `[photo: <description>]`, `[voice_message: <transcript>]` (a caption is kept and combined with the tag)

## Requirements

Everything is provided by the Nix dev shell (`shell.nix`): Go 1.26, `ffmpeg`, `whisper-cpp` (the `whisper-cli` binary), the HuggingFace CLI (`hf` + `hf-xet`) for model downloads, and Docker is used from the host for the GPU vision server.

```sh
nix-shell        # drops you into the dev shell with all aliases below
```

Without Nix you need: Go 1.26+, and (for `--decode-media`) `ffmpeg` + `whisper-cli` on `PATH`, a whisper GGUF model, and a reachable vision server.

## Quick start

```sh
nix-shell

# 1) plain text, no media decoding (fast)
parse result.json dialogue.txt

# 2) with media decoding (voice + photos)
parse-media result.json dialogue.txt
```

> **Note:** with Go's flag parser, all `--flags` must come **before** the input/output paths.

The dev shell exposes these aliases (see the banner printed on entry):

| alias | what it does |
|-------|--------------|
| `parse <in.json> [out.txt]` | convert an export to text |
| `parse-media <in.json> [out.txt]` | + transcribe voice & describe photos |
| `gotest` | run the Go test suite |
| `fetch-models` | download the Qwen2.5-VL GGUF + mmproj (for the Docker vision server) |
| `vision-up` / `vision-logs` / `vision-down` | manage the llama.cpp vision server in Docker (GPU, port 8080) |
| `tg-build` / `tg-run` | build & run the parser fully in Docker |

## Usage

```
Usage: parse [flags] <input.json> [output.txt]
```

`output.txt` defaults to `parsed_dialogue.txt`.

| flag | default | description |
|------|---------|-------------|
| `--decode-media` | `false` | transcribe voice messages and describe photos |
| `--whisper-model` | `""` | path to a whisper.cpp GGUF model (**required** for voice) |
| `--whisper-bin` | `whisper-cli` | whisper.cpp binary |
| `--whisper-lang` | `ru` | spoken language for transcription |
| `--vision-model` | `qwen2.5vl:7b` | vision model name sent to the server |
| `--vision-host` | `http://localhost:8080` | vision server base URL (OpenAI-compatible `/v1/chat/completions`, e.g. llama.cpp) |
| `--vision-prompt` | *(built-in RU prompt)* | prompt sent to the vision model |
| `--media-cache` | `<export-dir>/.tg2llm_media_cache.db` | decoded-media cache file |
| `--jobs` | `3` | parallel media-decode workers (`1` = sequential) |
| `--date-from` | `""` | skip messages before this date (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`) |
| `--date-to` | `""` | skip messages after this date (date-only includes the whole day) |
| `--verbose` | `false` | trace-level logging with caller info |

### Examples

```sh
# skip everything before a date (and optionally cap the end)
parse --date-from 2026-06-20 result.json dialogue.txt
parse --date-from 2026-06-20 --date-to 2026-06-23 result.json dialogue.txt

# media decode (parse-media is pre-wired to the llama.cpp server on :8080)
parse-media result.json dialogue.txt

# or explicitly, against any OpenAI-compatible vision server
parse --decode-media --vision-host http://localhost:8080 \
      --whisper-model ~/.cache/whisper-cpp/ggml-small.bin \
      result.json dialogue.txt
```

## Media decoding

`--decode-media` turns each supported media message into text:

- **Voice** (`voice_message`) → `ffmpeg` converts the `.ogg` to 16 kHz WAV, then `whisper-cli` transcribes it. Requires `--whisper-model` (a whisper GGUF, e.g. `ggml-small.bin`).
- **Photos** → sent (base64) to the vision server and replaced with its description. The server is any OpenAI-compatible vision endpoint (POST `/v1/chat/completions`: llama.cpp `server`, vLLM, …), set via `--vision-host`. This is what the Docker setup uses for **GPU** acceleration.

Any backend error is logged (`--verbose` for detail) and the message is left undecoded rather than crashing the run.

### Caching

Decoded results are stored in a SQLite database (default `<export-dir>/.tg2llm_media_cache.db`) keyed by `kind:sha256(file):model-signature`:

- identical-content media is decoded **once** (content-hash dedup),
- reruns skip already-decoded media,
- changing the model/prompt invalidates the relevant entries,
- empty/failed results are **not** cached, so a rerun retries only those.

## Docker (GPU photo description)

The [`docker/`](docker/) directory runs the photo-description model on a GPU via llama.cpp (Vulkan) and, optionally, the parser itself. See [`docker/README.md`](docker/README.md) for details.

```sh
# download the vision model into your models dir (default /mnt/ssd2tb/llm/models)
fetch-models

cd docker
make up                                   # start the llama.cpp vision server (GPU, :8080)
make logs                                 # wait for the model to load
make run EXPORT_DIR=/path/to/ChatExport   # parse the export inside Docker
make down                                 # stop
```

You can also run the parser on the host against the dockerized server: `parse-media` is pre-wired to `--vision-host http://localhost:8080`.

## Architecture

Hexagonal (ports & adapters):

```
cmd/parse/                 CLI entrypoint, flag parsing, logger setup
internal/domain/dialogue/  core types: Message, CleanText, DateRange, ParseBound
internal/app/              ParseService orchestration, ports, parallel decode pool
internal/infra/
  tgexport/                read the Telegram/AyuGram JSON export
  textsink/                write the dialogue text
  media/                   SQLite cache, decoder, whisper + OpenAI vision backend
pkg/ctxlog/                operation-scoped zerolog helper
```

The base parser (no `--decode-media`) is dependency-free at runtime; media decoding shells out to `ffmpeg`/`whisper-cli` and talks HTTP to the vision server.

## Development

```sh
gotest                                              # via the dev shell, or:
nix-shell shell.nix --run 'GOTOOLCHAIN=local go test ./...'
nix-shell shell.nix --run 'GOTOOLCHAIN=local go build ./... && go vet ./...'
```
