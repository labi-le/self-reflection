#!/usr/bin/env python3
"""v2 recipe — strip instruct formatting to plain casual text.

The v1 model leaked ```code fences```, LaTeX (\\( 17 \\times 23 \\)) and bullet/
numbered lists from the instruct blend. This flattens the restyled instruct
answers to plain conversational text so nothing structural survives to leak.

Usage: strip_instruct.py [IN] [OUT]
  IN  default finetune/data/premium_raw/instruct_restyled.jsonl
  OUT default finetune/data/premium_raw/instruct.plain.jsonl
"""
import json
import re
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else "finetune/data/premium_raw/instruct_restyled.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "finetune/data/premium_raw/instruct.plain.jsonl"


def strip_fmt(t: str) -> str:
    t = re.sub(r"```[a-zA-Z0-9_+\-]*\n?", "", t)  # code-fence openers
    t = t.replace("```", "").replace("`", "")      # stray fences + inline code
    # LaTeX delimiters -> keep inner
    t = re.sub(r"\\\[(.+?)\\\]", r"\1", t, flags=re.S)
    t = re.sub(r"\\\((.+?)\\\)", r"\1", t, flags=re.S)
    t = re.sub(r"\$\$(.+?)\$\$", r"\1", t, flags=re.S)
    t = re.sub(r"\$(.+?)\$", r"\1", t, flags=re.S)
    # common LaTeX commands -> plain
    t = t.replace("\\times", " на ").replace("\\cdot", "*").replace("\\div", "/")
    t = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"\1/\2", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)  # drop remaining \commands
    t = t.replace("{", "").replace("}", "")
    # markdown headers / bold / blockquote
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    # list markers -> plain lines
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)
    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


kept = changed = 0
with open(OUT, "w", encoding="utf-8") as w:
    for line in open(IN, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        for m in o.get("messages", []):
            if m.get("role") == "assistant":
                c = m.get("content", "")
                nc = strip_fmt(c)
                if nc != c:
                    changed += 1
                m["content"] = nc
        json.dump(o, w, ensure_ascii=False)
        w.write("\n")
        kept += 1

print(f"kept={kept} changed={changed}")
