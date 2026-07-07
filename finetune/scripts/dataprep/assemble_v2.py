#!/usr/bin/env python3
"""v2 recipe — assemble train.premium2.jsonl.

~77% cleaned persona + ~20% stripped instruct + ~3% identity, with the SAME
Денис system prompt forced on EVERY row (Oracle: unify the prompt so identity
is unconditional). Seed 42. Also strips stray unicode line-separators that break
naive JSONL readers.

Usage: assemble_v2.py
"""
import json
import random
import re

PERSONA = "finetune/data/train.persona.clean.jsonl"
INSTRUCT = "finetune/data/premium_raw/instruct.plain.jsonl"
IDENTITY = "finetune/data/premium_raw/identity.jsonl"
OUT = "finetune/data/train.premium2.jsonl"
SYS = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."
N_PERSONA = 11500
N_INSTRUCT = 3000
random.seed(42)
SEP = re.compile("[\u2028\u2029\u0085\x1c\x1d\x1e\x0b\x0c]")


def load(p: str) -> list:
    out = []
    for line in open(p, encoding="utf-8").read().split("\n"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def normalize(o: dict) -> dict:
    msgs = o.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = SYS
    else:
        msgs = [{"role": "system", "content": SYS}] + msgs
    for m in msgs:
        m["content"] = SEP.sub(" ", m.get("content", ""))
    return {"messages": msgs}


persona = [normalize(o) for o in load(PERSONA)]
instruct = [normalize(o) for o in load(INSTRUCT)]
identity = [normalize(o) for o in load(IDENTITY)]
random.shuffle(persona)
random.shuffle(instruct)
sel = persona[:N_PERSONA] + instruct[:N_INSTRUCT] + identity
random.shuffle(sel)

with open(OUT, "w", encoding="utf-8") as w:
    for r in sel:
        json.dump(r, w, ensure_ascii=False)
        w.write("\n")

np, ni, nid = min(N_PERSONA, len(persona)), min(N_INSTRUCT, len(instruct)), len(identity)
tot = len(sel)
print(f"persona {np} ({100*np/tot:.0f}%) + instruct {ni} ({100*ni/tot:.0f}%) + "
      f"identity {nid} ({100*nid/tot:.0f}%) = {tot} rows -> {OUT}")
