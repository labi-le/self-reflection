#!/usr/bin/env python3
"""Re-scrub finetune/data/train.jsonl for the gluing-fix retrain.

Difference vs the original scrub: invisible burst-delimiters (U+2063 etc.) are
REPLACED WITH A SPACE instead of stripped, so consecutive Денис message-bursts
stop being glued into 'word1word2'. Same PII redaction as before.
"""
import json
import re

SRC = "finetune/data/train.jsonl"
OUT = "finetune/data/train.scrubbed.spaced.jsonl"

INVIS = ["\u2060", "\u2061", "\u2062", "\u2063", "\u2064",
         "\u200b", "\u200c", "\u200d", "\ufeff"]
PHONE_INTL = re.compile(r"\+?\d{11,}")
PHONE_RU = re.compile(r"\+?[78]\d{10}")
PHONE_BARE10 = re.compile(r"\b9\d{9}\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
DOMAIN = re.compile(r"\blabile\.cc\b", re.IGNORECASE)
HWS = re.compile(r"[^\S\n]+")

# Example placeholders only — the real values live in the (gitignored) dataset,
# never in the repo; this list is just a post-scrub self-check probe.
REAL_PII = ["79001234567", "79007654321", "79009998877", "example.com",
            "9001234567", "9007654321", "9009998877"]


def scrub(text: str) -> str:
    if not text:
        return text
    text = PHONE_INTL.sub("[phone]", text)
    text = PHONE_RU.sub("[phone]", text)
    text = PHONE_BARE10.sub("[phone]", text)
    text = EMAIL.sub("[email]", text)
    text = DOMAIN.sub("[domain]", text)
    for ch in INVIS:
        text = text.replace(ch, " ")
    text = HWS.sub(" ", text)
    text = "\n".join(ln.strip() for ln in text.split("\n")).strip()
    return text


before = {"rows": 0, "u2063_turns": 0, "phone_hits": 0, "email_hits": 0, "domain_hits": 0}
with open(SRC, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        before["rows"] += 1
        for m in o.get("messages", []):
            c = m.get("content", "")
            if "\u2063" in c:
                before["u2063_turns"] += 1
            before["phone_hits"] += len(PHONE_INTL.findall(c)) + len(PHONE_RU.findall(c))
            before["email_hits"] += len(EMAIL.findall(c))
            before["domain_hits"] += len(DOMAIN.findall(c))
            m["content"] = scrub(c)
        out.write(json.dumps(o, ensure_ascii=False) + "\n")

print("BEFORE:", before)

after = {"rows": 0, "u2063_turns": 0, "real_pii_hits": 0}
glue_examples = []
for line in open(OUT, encoding="utf-8"):
    o = json.loads(line)
    after["rows"] += 1
    for m in o.get("messages", []):
        c = m.get("content", "")
        if "\u2063" in c:
            after["u2063_turns"] += 1
        for r in REAL_PII:
            if r in c:
                after["real_pii_hits"] += 1
                break

# show former-glue points now spaced (emoji+url pattern from the earlier probe)
for line in open(OUT, encoding="utf-8"):
    o = json.loads(line)
    for m in o.get("messages", []):
        c = m.get("content", "")
        if "\U0001f609 \U0001f92a" in c and len(glue_examples) < 3:
            i = c.find("\U0001f609 \U0001f92a")
            glue_examples.append(c[max(0, i - 5):i + 30])
    if len(glue_examples) >= 3:
        break

print("AFTER:", after)
print("out_size", __import__("os").path.getsize(OUT))
for g in glue_examples:
    print("  spaced:", repr(g))
