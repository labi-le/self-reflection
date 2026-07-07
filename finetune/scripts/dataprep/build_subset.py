#!/usr/bin/env python3
# uv run --no-project --with polars /tmp/opencode/build_subset.py
import glob
import json
import random

import polars as pl

RAW = "/home/labile/GolandProjects/tg2llm/finetune/data/premium_raw"
OUT = f"{RAW}/instruct_subset.jsonl"
SEED = 42
N_SAIGA = 3000
N_GM = 2000

random.seed(SEED)


def find_parquets(subdir: str, pattern: str) -> list[str]:
    hits = sorted(glob.glob(f"{RAW}/{subdir}/**/{pattern}", recursive=True))
    bad = ("/.cache/", "/download/", "/huggingface/", "/test-")
    hits = [h for h in hits if not any(b in h for b in bad)]
    if not hits:
        raise FileNotFoundError(f"no parquet for {subdir}/{pattern}")
    return hits


def norm_role(r: str | None) -> str | None:
    m = {
        "human": "user", "user": "user", "prompter": "user",
        "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
        "system": "system",
    }
    return m.get((r or "").strip().lower())


def norm_msgs(raw) -> list[dict] | None:
    out = []
    for m in raw:
        d = dict(m)
        role = norm_role(d.get("role") or d.get("from"))
        content = d.get("content")
        if content is None:
            content = d.get("value")
        if role is None or not content:
            continue
        out.append({"role": role, "content": str(content)})
    if len(out) < 2 or out[-1]["role"] != "assistant":
        return None
    return out


# --- saiga ---
saiga_path = find_parquets("saiga_scored", "train*.parquet")[0]
s = pl.read_parquet(saiga_path)
s = s.filter(
    (pl.col("opus_score") >= 9)
    & (pl.col("language") == "Russian")
    & (~pl.col("is_bad_by_regex"))
    & (pl.col("sonnet_complexity").is_in(["medium", "hard"]))
    & (pl.col("turns") == 1)
)
print(f"saiga filtered: {s.height} (from {saiga_path})")

# --- grandmaster ---
gm_files = find_parquets("grandmaster", "train-*.parquet")
g = pl.concat([pl.read_parquet(f) for f in gm_files])
g = g.filter(
    (pl.col("answer_lang") == "ru")
    & (pl.col("answer_tokens") >= 300)
    & (pl.col("answer_tokens") <= 900)
).unique(subset=["cluster"], keep="first")
print(f"grandmaster filtered (1/cluster): {g.height} (from {len(gm_files)} files)")

rows: list[dict] = []
for r in s.iter_rows(named=True):
    msgs = norm_msgs(r["messages"])
    if msgs:
        rows.append({"messages": msgs, "src": "saiga"})
for r in g.iter_rows(named=True):
    msgs = norm_msgs(r["conversation"])
    if msgs:
        rows.append({"messages": msgs, "src": "grandmaster"})

saiga_rows = [r for r in rows if r["src"] == "saiga"]
gm_rows = [r for r in rows if r["src"] == "grandmaster"]
random.shuffle(saiga_rows)
random.shuffle(gm_rows)
picked = saiga_rows[:N_SAIGA] + gm_rows[:N_GM]
random.shuffle(picked)

with open(OUT, "w", encoding="utf-8") as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter

bd = Counter(r["src"] for r in picked)
print(f"WROTE {len(picked)} -> {OUT}  breakdown={dict(bd)}")
print("--- sample saiga ---")
print(json.dumps(saiga_rows[0]["messages"], ensure_ascii=False)[:600])
print("--- sample grandmaster ---")
print(json.dumps(gm_rows[0]["messages"], ensure_ascii=False)[:600])
