# tg2llm in Docker (GPU vision via llama.cpp)

Runs image description (photos and video-note frames) on your AMD GPU (Vulkan)
through a `llama.cpp` server, and voice / video-note transcription via a bundled
`whisper-cli`. Mirrors the existing `/mnt/ssd2tb/llm/models` llama.cpp compose
pattern.

## 1. Download the vision model (once)

Into your models dir (default `/mnt/ssd2tb/llm/models`):

```sh
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
  Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
  --local-dir /mnt/ssd2tb/llm/models
```

- `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (~4.7 GB) — the vision model.
- `mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf` (~1.4 GB) — the multimodal projector
  (required for images). A smaller `...-Q8_0.gguf` projector (~850 MB) also works.

(From the dev shell, `fetch-models` runs this for you.)

## 2. Start the GPU vision server

```sh
make up      # starts the llama.cpp:server-vulkan vision service on :8080
make logs    # watch it load the model on the GPU
```

## 3. Run the parser

```sh
make build                                   # build the tg2llm utility image
make run                                      # uses the default export path
make run EXPORT_DIR=/path/to/ChatExport       # or point it at your export
```

Output lands at `<export>/parsed_dialogue.txt`. The decoded-media cache lives in the
checkout at `cache/media.db`: compose bind-mounts `../cache` to `/cache` and sets
`TG2LLM_CACHE_DIR=/cache` so the binary finds it (there is no `go.mod` in the image to
anchor on). Container and host runs therefore share one database and re-runs only decode
new or previously-failed media. `make run` creates `../cache` first and passes your
uid/gid, so the cache and the output file stay yours — keep that directory in place, or
Docker recreates it root-owned and decoding falls back to uncached with a `WRN cache open
failed` line. A `parsed_dialogue.txt` left behind by an older (root) run must be deleted once
(`sudo rm <export>/parsed_dialogue.txt`), otherwise the first run after upgrading fails
with a permission error before it decodes anything.

## Overridable variables

| Var             | Default                                          | Purpose                              |
|-----------------|--------------------------------------------------|--------------------------------------|
| `MODELS_DIR`    | `/mnt/ssd2tb/llm/models`                         | host dir with the GGUF + mmproj      |
| `EXPORT_DIR`    | `/home/labile/Downloads/ChatExport_2026-06-25`   | Telegram export folder               |
| `WHISPER_MODEL` | `/home/labile/.cache/whisper-cpp/ggml-small.bin` | whisper model for voice              |
| `TG2LLM_UID`    | `$(id -u)` via `make run`, else `1000`           | uid the container writes as          |
| `TG2LLM_GID`    | `$(id -g)` via `make run`, else `1000`           | gid the container writes as          |

## How it maps to the code

- Photos and video-note frames → `--vision-host http://vision:8080`: the OpenAI-compatible
  describer POSTs to the llama.cpp server's `/v1/chat/completions` with the image as a
  base64 `image_url`. Runs on the GPU (Vulkan).
- Voice and video-note audio → `whisper-cli` (bundled in the image) + `ffmpeg`, no server
  needed. A video note yields `[video_message: <frame description> Речь: <transcript>]`.
- Context size is set server-side via `-c 4096` in the compose command, with
  `--image-max-tokens 1024` capping one image's token cost (the OpenAI endpoint has no
  per-request context field). Raise both if large images error out.
