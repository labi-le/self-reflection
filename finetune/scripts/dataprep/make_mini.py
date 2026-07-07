#!/usr/bin/env python3
"""Fast-iteration mini dataset: small balanced subset for ~10min recipe tests.

~900 persona + ~200 identity + ~100 instruct = ~1200 rows, unified Денис prompt.
Small enough to train in ~9 min on the 4B, big enough to reveal whether the
persona dominates at low temp + identity is correct + stopping works.

Usage: make_mini.py [OUT] [N_PERSONA] [N_IDENTITY] [N_INSTRUCT]
"""
import json
import random
import re
import sys

PERSONA = "finetune/data/train.persona.clean.jsonl"
INSTRUCT = "finetune/data/premium_raw/instruct.plain.jsonl"
IDENTITY = "finetune/data/premium_raw/identity.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "finetune/data/train.mini.jsonl"
N_PERSONA = int(sys.argv[2]) if len(sys.argv) > 2 else 900
N_IDENTITY = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_INSTRUCT = int(sys.argv[4]) if len(sys.argv) > 4 else 100
SYS = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."
random.seed(42)
SEP = re.compile("[\u2028\u2029\u0085\x1c\x1d\x1e\x0b\x0c]")


def load(p):
    return [json.loads(x) for x in open(p, encoding="utf-8").read().split("\n") if x.strip()]


def norm(o):
    msgs = o.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = SYS
    else:
        msgs = [{"role": "system", "content": SYS}] + msgs
    for m in msgs:
        m["content"] = SEP.sub(" ", m.get("content", ""))
    return {"messages": msgs}


persona = [norm(o) for o in load(PERSONA)]
identity = [norm(o) for o in load(IDENTITY)]
instruct = [norm(o) for o in load(INSTRUCT)]
random.shuffle(persona)
random.shuffle(identity)
random.shuffle(instruct)
sel = persona[:N_PERSONA] + identity[:N_IDENTITY] + instruct[:N_INSTRUCT]
random.shuffle(sel)
with open(OUT, "w", encoding="utf-8") as w:
    for r in sel:
        json.dump(r, w, ensure_ascii=False)
        w.write("\n")
print(f"mini: {min(N_PERSONA,len(persona))} persona + {min(N_IDENTITY,len(identity))} identity "
      f"+ {min(N_INSTRUCT,len(instruct))} instruct = {len(sel)} rows -> {OUT}")
