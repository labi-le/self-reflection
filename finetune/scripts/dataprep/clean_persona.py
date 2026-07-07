#!/usr/bin/env python3
"""v2 recipe — clean persona data.

- Cap consecutive emoji runs to <=3 (kills the 848-emoji-spam that fed the
  high-temp degeneration observed on the v1 model).
- Drop conversations that contain an emoji-only or >1500-char assistant turn
  (~0.7% pasted-article / spam outliers).

Usage: clean_persona.py [IN] [OUT]
  IN  default finetune/data/train.scrubbed.spaced.jsonl
  OUT default finetune/data/train.persona.clean.jsonl
"""
import json
import re
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else "finetune/data/train.scrubbed.spaced.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "finetune/data/train.persona.clean.jsonl"

EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF"
    "\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF\uFE0F\u200d]"
)


def cap_emoji_runs(s: str, limit: int = 3) -> str:
    out: list[str] = []
    run = 0
    for ch in s:
        if EMOJI.match(ch):
            run += 1
            if run <= limit:
                out.append(ch)
        else:
            run = 0
            out.append(ch)
    return "".join(out)


def bad_turn(c: str) -> bool:
    if len(c) > 1500:
        return True
    return bool(c.strip()) and not EMOJI.sub("", c).strip()  # emoji-only


kept = dropped = capped = 0
with open(OUT, "w", encoding="utf-8") as w:
    for line in open(IN, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        msgs = o.get("messages", [])
        if any(m.get("role") == "assistant" and bad_turn(m.get("content", "")) for m in msgs):
            dropped += 1
            continue
        for m in msgs:
            c = m.get("content", "")
            nc = cap_emoji_runs(c)
            if nc != c:
                capped += 1
            m["content"] = nc
        json.dump(o, w, ensure_ascii=False)
        w.write("\n")
        kept += 1

print(f"kept={kept} dropped={dropped} turns_emoji_capped={capped}")
