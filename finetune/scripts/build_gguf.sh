#!/usr/bin/env bash
# Build a quantized GGUF from a QLoRA adapter: merge -> convert -> quantize -> verify.
#   usage: tg2llm_build_gguf.sh <adapter_dir_host> <out_basename>
# CPU-ONLY (no GPU: never conflicts with training/vision), OFFLINE, memory-capped.
# Never deletes the adapter or checkpoints. All intermediates land on /mnt/ssd2tb.
set -uo pipefail

ADAPTER_HOST="${1:?usage: build_gguf <adapter_dir_host> <out_basename>}"
OUT="${2:?usage: build_gguf <adapter_dir_host> <out_basename>}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
TEMPLATE="${TEMPLATE:-qwen3_nothink}"
HF_DIRNAME="models--${BASE_MODEL//\//--}"

IMG_FT="tg2llm-finetune:latest"
IMG_LC="ghcr.io/ggml-org/llama.cpp:full"
MODELS_HOST="/mnt/ssd2tb/llm/models"
WORK_HOST="$MODELS_HOST/persona-build/$OUT"
MERGED_HOST="$WORK_HOST/merged"          # -> /merged/model (HF fp16)
GGUF_HOST="$WORK_HOST/gguf"              # -> gguf outputs
MERGE_MEM="${MERGE_MEM:-18g}"
CONV_MEM="${CONV_MEM:-12g}"
LOG="/tmp/tg2llm_build_${OUT}.log"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG" ; }
die(){ log "FATAL: $*"; exit 1; }
DK(){ if command -v rtk >/dev/null 2>&1; then rtk docker "$@"; else docker "$@"; fi ; }

log "=== build_gguf START adapter=$ADAPTER_HOST out=$OUT ==="

# ---- validate inputs (fail loud, touch nothing) ----
[ -d "$ADAPTER_HOST" ]                              || die "adapter dir missing: $ADAPTER_HOST"
[ -f "$ADAPTER_HOST/adapter_config.json" ]          || die "no adapter_config.json in $ADAPTER_HOST"
[ -f "$ADAPTER_HOST/adapter_model.safetensors" ]    || die "no adapter_model.safetensors in $ADAPTER_HOST"
[ -d "$MODELS_HOST/hf" ]                            || die "HF cache missing: $MODELS_HOST/hf"
DK image inspect "$IMG_FT" >/dev/null 2>&1          || die "image missing: $IMG_FT"
DK image inspect "$IMG_LC" >/dev/null 2>&1          || die "image missing: $IMG_LC"
mkdir -p "$MERGED_HOST" "$GGUF_HOST"                || die "mkdir failed under $WORK_HOST"

F16="$GGUF_HOST/${OUT}-f16.gguf"
Q4="$GGUF_HOST/${OUT}.Q4_K_M.gguf"

# ---- 1) MERGE (LLaMA-Factory export, CPU, offline) ----
cat > "$MERGED_HOST/merge.yaml" <<YAML
model_name_or_path: $BASE_MODEL
adapter_name_or_path: /adapter
template: $TEMPLATE
trust_remote_code: true
finetuning_type: lora
export_dir: /merged/model
export_size: 5
export_legacy_format: false
YAML

if [ -f "$MERGED_HOST/model/config.json" ]; then
  log "1/4 merge: already present, skipping"
else
  log "1/4 merge: fp16 merge on CPU (cap $MERGE_MEM) ..."
  DK run --rm --memory "$MERGE_MEM" \
    -e HF_HOME=/models/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
    -v "$MODELS_HOST":/models -v "$ADAPTER_HOST":/adapter:ro -v "$MERGED_HOST":/merged \
    --entrypoint bash "$IMG_FT" \
    -c 'cd /app && exec /opt/venv/bin/llamafactory-cli export /merged/merge.yaml' 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ]                      || die "merge failed"
  [ -f "$MERGED_HOST/model/config.json" ]          || die "merge produced no config.json"
fi

# ---- 2) CONVERT HF -> GGUF f16 (with tokenizer-compat fix) ----
# LLaMA-Factory (transformers 5.x) re-saves tokenizer_config.json with `extra_special_tokens`
# as a LIST, which the llama.cpp image's older transformers can't parse (expects a dict). The
# LoRA never touched the tokenizer, so restore the base model's original tokenizer files first.
if [ -s "$F16" ]; then
  log "2/4 convert: $F16 present, skipping"
else
  log "2/4 convert: restore base tokenizer + HF -> GGUF f16 ..."
  DK run --rm --memory "$CONV_MEM" \
    -e HF_DIRNAME="$HF_DIRNAME" \
    -v "$MODELS_HOST":/models -v "$MERGED_HOST":/merged -v "$GGUF_HOST":/gguf \
    --entrypoint bash "$IMG_LC" -c '
      set -e
      SNAP=$(ls -d /models/hf/hub/"$HF_DIRNAME"/snapshots/*/ 2>/dev/null | awk "NR==1")
      [ -n "$SNAP" ] || { echo "base snapshot not found under /models"; exit 3; }
      for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
        [ -f "$SNAP/$f" ] && cp -Lf "$SNAP/$f" /merged/model/"$f"
      done
      exec python3 /app/convert_hf_to_gguf.py /merged/model --outfile /gguf/'"${OUT}"'-f16.gguf --outtype f16
    ' 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ]                      || die "convert failed"
  [ -s "$F16" ]                                     || die "convert produced no f16 gguf"
fi

# ---- 3) QUANTIZE f16 -> Q4_K_M ----
if [ -s "$Q4" ]; then
  log "3/4 quantize: $Q4 present, skipping"
else
  log "3/4 quantize: f16 -> Q4_K_M ..."
  DK run --rm \
    -v "$GGUF_HOST":/gguf \
    --entrypoint /app/llama-quantize "$IMG_LC" \
    "/gguf/${OUT}-f16.gguf" "/gguf/${OUT}.Q4_K_M.gguf" Q4_K_M 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ]                      || die "quantize failed"
  [ -s "$Q4" ]                                      || die "quantize produced no Q4 gguf"
fi

# ---- 4) VERIFY: structural check only, NO inference. An unattended generate can loop
#      and flood/OOM (it already OOM-killed training once); llama-quantize above already
#      validated every tensor. Confirm the GGUF magic + a sane size. The functional chat
#      test is a MORNING, attended task — never in this unattended daemon path. ----
log "4/4 verify: GGUF structural check (no inference) ..."
MAGIC="$(dd if="$Q4" bs=4 count=1 2>/dev/null)"
[ "$MAGIC" = "GGUF" ]                               || die "verify: $Q4 missing GGUF magic (got '$MAGIC')"
QBYTES="$(stat -c %s "$Q4")"
MINQ=$(( "$(stat -c %s "$F16")" / 5 ))
[ "$QBYTES" -ge "$MINQ" ]                           || die "verify: $Q4 too small ($QBYTES < $MINQ bytes)"
log "verify: GGUF magic OK, size $QBYTES bytes"

SZ="$(du -h "$Q4" | cut -f1)"
log "=== build_gguf DONE: $Q4 ($SZ) ==="
echo "BUILD_OK $Q4"
