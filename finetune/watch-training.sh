#!/usr/bin/env bash
# Live training monitor: step, %, loss, ETA.
#
# Reads LLaMA-Factory's trainer_log.jsonl (appended every logging_steps with
# current/total steps, loss, percentage, elapsed_time and remaining_time). For a
# single clean run (no checkpoint-resume) those timings are accurate, so the ETA is
# taken straight from remaining_time — no supervisor.log needed.
#
# Usage:  bash finetune/watch-training.sh                          # live, refresh 10s
#         ONCE=1 bash finetune/watch-training.sh                   # one frame, exit
#         RUN=qwen3-8b-persona-qlora-premium bash finetune/watch-training.sh
#         TLOG=/abs/path/trainer_log.jsonl bash finetune/watch-training.sh
set -uo pipefail

OUT="${OUT:-/home/labile/GolandProjects/tg2llm/finetune/output}"
RUN="${RUN:-qwen3-8b-persona-qlora-premium}"
TLOG="${TLOG:-$OUT/$RUN/trainer_log.jsonl}"

hms2s() { awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print $1+0 }'; }

while :; do
  last=$(awk 'END{print}' "$TLOG" 2>/dev/null || true)
  if [ -z "$last" ]; then
    echo "waiting for trainer_log.jsonl ($TLOG) ..."
    [ -n "${ONCE:-}" ] && break
    sleep 5; continue
  fi

  IFS=$'\t' read -r step total pct loss epoch elapsed remain < <(
    printf '%s' "$last" | jq -r '[.current_steps, .total_steps, .percentage, (.loss // .eval_loss // "?"), .epoch, .elapsed_time, .remaining_time] | @tsv' 2>/dev/null
  )
  if [ -z "${step:-}" ]; then
    [ -n "${ONCE:-}" ] && break
    sleep 3; continue
  fi

  [ "$loss" != "?" ] && loss=$(awk "BEGIN{printf \"%.3f\", $loss}")
  ep=$(awk "BEGIN{printf \"%.2f\", ${epoch:-0}}")
  el_s=$(printf '%s' "$elapsed" | hms2s 2>/dev/null)
  if [ "${step:-0}" -gt 0 ] 2>/dev/null && [ -n "${el_s:-}" ]; then
    sps=$(awk "BEGIN{printf \"%.2f\", $el_s/$step}")
  else
    sps="?"
  fi
  rm_s=$(printf '%s' "$remain" | hms2s 2>/dev/null)
  if [ "${rm_s:-0}" -gt 0 ] 2>/dev/null; then
    finish=$(date -d "+$rm_s seconds" '+%a %H:%M' 2>/dev/null || echo '?')
  else
    finish="?"
  fi

  [ -z "${ONCE:-}" ] && clear
  printf 'TRAINING  %s\n' "$RUN"
  printf '  step    %s / %s   (%s%%,  epoch %s)\n' "$step" "$total" "$pct" "$ep"
  printf '  loss    %s\n' "$loss"
  printf '  rate    %s s/step  (avg)\n' "$sps"
  printf '  elapsed %s\n' "$elapsed"
  printf '  ETA     %s   ~finish %s\n' "$remain" "$finish"
  printf '  (updated %s  — Ctrl-C to quit)\n' "$(date '+%H:%M:%S')"

  [ -n "${ONCE:-}" ] && break
  sleep 10
done
