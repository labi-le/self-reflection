{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.go # Go 1.26 toolchain
    pkgs.ffmpeg # decode .ogg voice -> 16k wav for whisper.cpp
    pkgs.whisper-cpp # local speech-to-text (binary: whisper-cli)
    pkgs.curl # general HTTP utility
    (pkgs.python3.withPackages (
      ps: with ps; [
        huggingface-hub
        hf-xet
      ]
    )) # fetch-models: hf CLI + Xet download
  ];

  shellHook = ''
    export GOTOOLCHAIN=local
    root="${toString ./.}"

    # Anchor the shared cache on the checkout unless the caller already chose a
    # location: the aliases below run the binary in YOUR directory (usually the
    # export dir, outside the module), so without this every directory would grow
    # its own cache/. It must exist before Docker binds it, or the daemon creates
    # it root-owned and the container cannot write.
    export TG2LLM_CACHE_DIR="''${TG2LLM_CACHE_DIR:-$root/cache}"
    mkdir -p -m 700 "$TG2LLM_CACHE_DIR"
    # The Docker services write into that same dir; without a uid they write as
    # root and the host can no longer open the database. (UID is readonly in bash,
    # hence the prefixed names.)
    export TG2LLM_UID="$(id -u)"
    export TG2LLM_GID="$(id -g)"
    bindir="$HOME/.cache/tg2llm"
    mkdir -p "$bindir"

    # Go is the primary implementation. Each alias rebuilds from the module root (Go caches,
    # so it is instant when nothing changed), then runs the binary in YOUR current directory,
    # so relative input/output paths resolve normally and the command works from any directory.
    # NOTE: with Go's stdlib flag parser, --flags must come BEFORE the input/output paths.
    alias parse="go -C $root build -o $bindir/parse ./cmd/parse && $bindir/parse"
    alias parse-media="go -C $root build -o $bindir/parse ./cmd/parse && $bindir/parse --decode-media --vision-host http://localhost:8080 --whisper-model $HOME/.cache/whisper-cpp/ggml-small.bin"
    alias gotest="go -C $root test ./..."

    # Download the Qwen2.5-VL vision model (GGUF + mmproj) for the Docker llama.cpp vision server.
    # Target dir overridable via MODELS_DIR; defaults to the local llama.cpp models dir.
    # hf (HuggingFace CLI) + hf-xet speak HF's Xet protocol natively: fast parallel chunked
    # download with resume. aria2c's multi-range splitting 403s against Xet's per-range signed URLs.
    models_dir="''${MODELS_DIR:-/mnt/ssd2tb/llm/models}"
    alias fetch-models="mkdir -p $models_dir && hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf --local-dir $models_dir"

    # Docker: llama.cpp vision server (GPU) + tg2llm utility. Wraps docker/docker-compose.yml by
    # absolute path so the aliases work from any directory (compose resolves the build context and
    # relative paths against the compose file's dir). Needs host docker + the compose plugin.
    compose="docker compose -f $root/docker/docker-compose.yml"
    alias vision-up="$compose up -d vision"
    alias vision-logs="$compose logs -f vision"
    alias vision-down="$compose down"
    alias tg-build="$compose build tg2llm"
    alias tg-run="$compose run --rm tg2llm"

    echo "tg2llm dev shell (Go)"
    echo "  parse <export.json> [out.txt]          convert a Telegram export"
    echo "  parse-media <export.json> [out.txt]    + transcribe voice & video notes, describe photos"
    echo "  gotest                                 run Go tests"
    echo "  fetch-models                           download Qwen2.5-VL GGUF + mmproj (Docker vision)"
    echo "  vision-up / vision-logs / vision-down  llama.cpp vision server in Docker (GPU, port 8080)"
    echo "  tg-build / tg-run                      build & run the parser in Docker"
    echo "  date filter: parse --date-from 2026-06-20 [--date-to 2026-06-23] result.json out.txt"
    echo "  shared cache: $TG2LLM_CACHE_DIR (parse, analyze and the Docker services)"
    echo "  (Go: put --flags BEFORE the input/output paths)"
  '';
}
