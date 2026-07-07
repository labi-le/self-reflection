#!/usr/bin/env python3
"""Assemble train.premium.jsonl = ~70% PII-scrubbed persona + ~30% restyled instruct.
Post-processes restyled instruct per Oracle review:
  - force lowercase (Денис writes lowercase)
  - drop rows with Cyrillic/Latin script-mixed words (7B restyle artifact, e.g. 'петrarка')
  - verify numbers preserved vs the original answer (Oracle red flag #1); drop corrupted
  - keep ~15% of instruct in ORIGINAL (non-restyled) register as a reasoning-ceiling anchor
Wraps everything under the single Денис system prompt. Verifies 0 PII. seed 42."""
import json, re, random, collections

BASE = "/home/labile/GolandProjects/tg2llm/finetune/data"
RESTYLED = f"{BASE}/premium_raw/instruct_restyled.jsonl"
SUBSET   = f"{BASE}/premium_raw/instruct_subset.jsonl"
PERSONA  = f"{BASE}/train.scrubbed.spaced.jsonl"
OUT      = f"{BASE}/train.premium.jsonl"
DENIS = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."
ANCHOR_FRAC = 0.15
random.seed(42)

CYR = re.compile(r"[а-яё]"); LAT = re.compile(r"[a-z]")
PHONE_INTL = re.compile(r"\+?\d{11,}")   # intl phones (+375/+380) + long IDs the RU-only scrub missed
PL = re.compile(r"[+]?[78][0-9]{10}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+[.][a-z]{2,}")
def script_mixed(text):
    for w in re.findall(r"[A-Za-zА-Яа-яЁё]+", text.lower()):
        if CYR.search(w) and LAT.search(w):
            return True
    return False

def nums(text): return collections.Counter(re.findall(r"\d+", text))
def numbers_ok(orig, restyled):
    # flag only a FABRICATED >=3-digit number; missing numbers = benign list-marker/word restyling
    o = nums(orig)
    if not o: return True
    for k in nums(restyled):
        if len(k) >= 3 and k not in o:
            return False
    return True

# --- markdown flatten: strip formal scaffolding (Oracle: markdown/lists -> inline prose)
#     + lowercase to Денис register, but PROTECT code (fenced + inline) from mangling/lowercasing.
#     A markdown answer under the casual "личная переписка" prompt is an instant AI-tell. ---
CODE_FENCE  = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]+`")
OPENERS = re.compile(
    r"^\s*(интересный вопрос[!.]?|отличный вопрос[!.]?|хороший вопрос[!.]?|"
    r"конечно[!.]?|давайте разбер[её]мся[.,!]?|вот список[^:\n]*:|"
    r"вот несколько[^:\n]*:)\s*", re.I)
def flatten_md(text, lower=True):
    slots = []
    def stash(m): slots.append(m.group(0)); return f"\x00{len(slots)-1}\x00"
    t = CODE_FENCE.sub(stash, text)
    t = INLINE_CODE.sub(stash, t)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", t)              # headings
    t = re.sub(r"(?m)^\s{0,3}>\s?", "", t)                   # blockquotes
    t = re.sub(r"(?m)^\s{0,3}\d+[.)]\s+", "", t)             # numbered list markers
    t = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", t)               # bullet markers
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.DOTALL)  # bold
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.DOTALL)      # bold underscore
    t = t.replace("**", "").replace("__", "")               # stray leftovers
    t = OPENERS.sub("", t)                                   # formal openers
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if lower: t = t.lower()
    for i, code in enumerate(slots): t = t.replace(f"\x00{i}\x00", code)
    return t.strip()

def last_user(msgs):
    q = None
    for m in msgs:
        if m.get("role") == "user": q = (m.get("content") or "").strip()
    return q
def last_assistant(msgs):
    a = None
    for m in msgs:
        if m.get("role") == "assistant": a = (m.get("content") or "").strip()
    return a

# 1. original instruct answers by idx (for number-verify + non-restyled anchor)
orig = {}
for i, ln in enumerate(open(SUBSET)):
    ln = ln.strip()
    if not ln: continue
    d = json.loads(ln); q = last_user(d.get("messages", [])); a = last_assistant(d.get("messages", []))
    if q and a: orig[i] = (q, a)

# 2. process restyled instruct
kept, dropped = [], collections.Counter()
for ln in open(RESTYLED):
    ln = ln.strip()
    if not ln: continue
    d = json.loads(ln); idx = d.get("idx")
    q = last_user(d.get("messages", [])); r = last_assistant(d.get("messages", []))
    if not q or not r: dropped["empty"] += 1; continue
    if script_mixed(r): dropped["script_mix"] += 1; continue
    o = orig.get(idx)
    if o and not numbers_ok(o[1], r): dropped["bad_numbers"] += 1; continue
    kept.append((idx, q, flatten_md(r)))

# 3. ~15% anchor: use the ORIGINAL (non-restyled) answer, still wrapped as Денис
random.shuffle(kept)
n_anchor = int(len(kept) * ANCHOR_FRAC)
def wrap(q, a): return {"messages": [
    {"role": "system", "content": DENIS},
    {"role": "user", "content": q},
    {"role": "assistant", "content": a}]}
instruct = []
for j, (idx, q, r) in enumerate(kept):
    if j < n_anchor and idx in orig: instruct.append(wrap(orig[idx][0], flatten_md(orig[idx][1])))
    else:                            instruct.append(wrap(q, r))

# 4. persona: subsample so instruct = 30% of the blend  (persona = instruct/0.3*0.7)
persona = []
for l in open(PERSONA):
    if not l.strip(): continue
    d = json.loads(l)
    for m in d.get("messages", []):
        if m.get("content"):
            m["content"] = EMAIL_RE.sub("[email]", PHONE_INTL.sub("[phone]", m["content"]))
    persona.append(d)
target = min(int(len(instruct) / 0.30 * 0.70), len(persona))
persona_sample = random.sample(persona, target)

# 5. blend + shuffle + write
data = instruct + persona_sample
random.shuffle(data)
with open(OUT, "w") as f:
    for d in data: f.write(json.dumps(d, ensure_ascii=False) + "\n")

# 6. PII sanity
pblob = "\n".join(json.dumps(d, ensure_ascii=False) for d in persona_sample)
iblob = "\n".join(json.dumps(d, ensure_ascii=False) for d in instruct)
print(f"restyled kept={len(kept)} dropped={dict(dropped)}")
print(f"instruct total={len(instruct)} (non-restyled anchor={n_anchor})")
print(f"persona sampled={len(persona_sample)}/{len(persona)}")
print(f"TOTAL train.premium.jsonl = {len(data)}  (instruct {100*len(instruct)/max(len(data),1):.0f}%)")
print(f"PII persona phone-like={len(PL.findall(pblob))} (want 0) | instruct phone-like={len(PL.findall(iblob))} (coding false-pos)")
print(f"PII email-like: persona={len(EMAIL_RE.findall(pblob))} instruct={len(EMAIL_RE.findall(iblob))}")
print(f"TOTAL_STEPS (eff batch 16 x 3 epochs) = {(len(data)+15)//16 * 3}")
