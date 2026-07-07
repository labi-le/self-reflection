#!/usr/bin/env bash
# Wait for training to finish (trainmon sentinel), then build the persona Q4 GGUF.
# Detached, single-instance. Writes /tmp/tg2llm_autostart.DONE (BUILD_OK|FAILED ...).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SENTINEL=/tmp/tg2llm_trainmon.DONE
OUT_SENTINEL=/tmp/tg2llm_autostart.DONE
LOCK=/tmp/tg2llm_autostart.lock
LOG=/tmp/tg2llm_autostart.log
OUTPUT_ROOT=/home/labile/GolandProjects/tg2llm/finetune/output/qwen2.5-7b-instruct-persona-qlora
BUILD="$SCRIPT_DIR/build_gguf.sh"
Q4=/mnt/ssd2tb/llm/models/persona-build/persona/gguf/persona.Q4_K_M.gguf
POLL=60

exec 9>"$LOCK" || exit 0
flock -n 9 || { echo "[$(date '+%F %T')] autostart already running -> exit" >>"$LOG"; exit 0; }

log(){ echo "[$(date '+%F %T')] $*" >>"$LOG" ; }
finish_fail(){ log "FAILED: $*"; echo "FAILED $*" > "$OUT_SENTINEL"; exit 1; }

log "=== autostart START; waiting for $TRAIN_SENTINEL ==="

# --- 1) wait for training completion sentinel ---
while :; do
  if [ -f "$TRAIN_SENTINEL" ]; then
    W="$(awk 'NR==1{print $1; exit}' "$TRAIN_SENTINEL" 2>/dev/null)"
    if [ "$W" = "DONE" ]; then
      log "training DONE: $(cat "$TRAIN_SENTINEL")"
      break
    elif [ "$W" = "PROBLEM" ]; then
      finish_fail "training-problem $(cat "$TRAIN_SENTINEL")"
    fi
  fi
  sleep "$POLL"
done

# --- 2) resolve final adapter (wait out the save-flush race, else newest checkpoint) ---
ADAPTER=""
for i in $(seq 1 10); do
  if [ -f "$OUTPUT_ROOT/adapter_model.safetensors" ]; then ADAPTER="$OUTPUT_ROOT"; break; fi
  log "waiting for final adapter flush at root ($i/10) ..."
  sleep 30
done
if [ -z "$ADAPTER" ]; then
  CK="$(for d in "$OUTPUT_ROOT"/checkpoint-*; do [ -d "$d" ] && echo "${d##*checkpoint-} $d"; done | sort -n | awk 'END{print $2}')"
  [ -n "$CK" ] && [ -f "$CK/adapter_model.safetensors" ] && ADAPTER="$CK"
fi
[ -n "$ADAPTER" ] || finish_fail "no-adapter under $OUTPUT_ROOT"
log "building from adapter: $ADAPTER"

# --- 3) build (merge -> gguf f16 -> Q4_K_M -> verify) ---
if bash "$BUILD" "$ADAPTER" persona >>"$LOG" 2>&1; then
  log "=== autostart DONE: $Q4 ==="
  echo "BUILD_OK $Q4" > "$OUT_SENTINEL"
else
  finish_fail "build-error (see $LOG)"
fi
