#!/usr/bin/env bash
# Task 10 config gate — LLaMA-Factory QLoRA persona fine-tune.
#
# Verifies the training YAML + dataset registry against the locked contract.
# No Python on host: pure bash + awk (parsing) + jq (JSON validation only).
# Designed to FAIL before the configs exist and PASS once they are authored.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="$SCRIPT_DIR/configs/qlora_qwen25_7b_instruct.yaml"
DATASET_INFO="$SCRIPT_DIR/data/dataset_info.json"

# jq: prefer PATH, fall back to the NixOS system path.
JQ_BIN="${JQ_BIN:-jq}"
if ! command -v "$JQ_BIN" >/dev/null 2>&1; then
  JQ_BIN="/run/current-system/sw/bin/jq"
fi

errors=0
ok()  { printf 'ok:   %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; errors=$((errors + 1)); }

# Extract a top-level YAML scalar value by key (first match only), trimming the
# leading "key:", surrounding whitespace, a trailing inline comment, and quotes.
# Matching only the "key:" line means comments that MENTION a forbidden token
# (e.g. an explanatory note about lm_head) never taint the checked value.
yaml_scalar() {
  awk -v k="$1" '
    index($0, k ":") == 1 {
      v = substr($0, length(k) + 2)     # drop "key:"
      sub(/^[[:space:]]+/, "", v)       # leading space
      sub(/[[:space:]]+#.*$/, "", v)    # trailing inline comment
      sub(/[[:space:]]+$/, "", v)       # trailing space
      gsub(/^["'\'']|["'\'']$/, "", v)   # surrounding quotes
      print v
      exit
    }
  ' "$YAML"
}

# --- existence (RED before the configs are authored) ------------------------
[[ -f "$YAML" ]]         || bad "training YAML not found: $YAML"
[[ -f "$DATASET_INFO" ]] || bad "dataset registry not found: $DATASET_INFO"
if (( errors > 0 )); then
  printf '\nRESULT: FAIL (%d error(s)) — configs missing\n' "$errors" >&2
  exit 1
fi

# --- (1) dataset_info.json is valid JSON ------------------------------------
if "$JQ_BIN" -e . "$DATASET_INFO" >/dev/null 2>&1; then
  ok "(1) dataset_info.json is valid JSON (jq -e)"
else
  bad "(1) dataset_info.json is not valid JSON (jq -e failed)"
fi

# --- (2) quantization_bit must be 4 -----------------------------------------
qbit="$(yaml_scalar quantization_bit)"
if [[ "$qbit" == "4" ]]; then
  ok "(2) quantization_bit: 4"
else
  bad "(2) quantization_bit must be 4 (got: '${qbit:-<absent>}')"
fi

# --- (3) cutoff_len present and <= 1024 -------------------------------------
cutoff="$(yaml_scalar cutoff_len)"
if [[ -z "$cutoff" ]]; then
  bad "(3) cutoff_len is absent"
elif [[ ! "$cutoff" =~ ^[0-9]+$ ]]; then
  bad "(3) cutoff_len is not an integer (got: '$cutoff')"
elif (( cutoff > 1024 )); then
  bad "(3) cutoff_len must be <= 1024 (got: $cutoff)"
else
  ok "(3) cutoff_len present and <= 1024 (= $cutoff)"
fi

# --- (4) model_name_or_path must be an Instruct model -----------------------
model="$(yaml_scalar model_name_or_path)"
case "$model" in
  *Instruct*) ok "(4) model_name_or_path is an Instruct model (= $model)" ;;
  "")         bad "(4) model_name_or_path is absent" ;;
  *)          bad "(4) model_name_or_path must contain 'Instruct' (got: '$model')" ;;
esac

# --- (5) lora_target must exclude lm_head AND embed_tokens -------------------
lora_target="$(yaml_scalar lora_target)"
if [[ -z "$lora_target" ]]; then
  bad "(5) lora_target is absent"
elif [[ "$lora_target" == *lm_head* || "$lora_target" == *embed_tokens* ]]; then
  bad "(5) lora_target must exclude lm_head/embed_tokens (got: '$lora_target')"
else
  ok "(5) lora_target excludes lm_head and embed_tokens (= $lora_target)"
fi

# --- verdict ----------------------------------------------------------------
if (( errors > 0 )); then
  printf '\nRESULT: FAIL (%d error(s))\n' "$errors" >&2
  exit 1
fi
printf '\nRESULT: PASS — all 5 checks green\n'
exit 0
