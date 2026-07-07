#!/usr/bin/env python3
"""Redact PII (RU phones, emails) + strip invisible chars from a ShareGPT jsonl.
Writes a scrubbed copy and self-verifies zero leftovers. Original is untouched."""
import json, re, collections, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "finetune/data/train.jsonl"
DST = sys.argv[2] if len(sys.argv) > 2 else "finetune/data/train.scrubbed.jsonl"

INVIS = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0x2062, 0x2063, 0x2064, 0xFEFF], None
)
# contiguous 11-digit RU phone, +7/7/8 prefix, not part of a longer digit run
phone_re = re.compile(r"(?<![0-9])\+?[78][0-9]{10}(?![0-9])")
email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# personal-domain placeholder — real value stays out of the repo (email-form caught above)
deny_re = re.compile(re.escape("example.com"), re.IGNORECASE)

stats = collections.Counter()


def scrub(text):
    text = text.translate(INVIS)
    text, n_e = email_re.subn("[email]", text)
    text, n_p = phone_re.subn("[phone]", text)
    text, n_d = deny_re.subn("[domain]", text)
    stats["emails"] += n_e
    stats["phones"] += n_p
    stats["domain"] += n_d
    return text


with open(SRC, encoding="utf-8") as f, open(DST, "w", encoding="utf-8") as out:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        obj = json.loads(line)
        for msg in obj.get("messages", []):
            c = msg.get("content")
            if isinstance(c, str):
                msg["content"] = scrub(c)
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stats["lines"] += 1

leftover_p = leftover_e = 0
with open(DST, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        for msg in obj.get("messages", []):
            c = msg.get("content", "")
            if isinstance(c, str):
                leftover_p += len(phone_re.findall(c))
                leftover_e += len(email_re.findall(c))

print(json.dumps({
    "lines": stats["lines"],
    "phones_replaced": stats["phones"],
    "emails_replaced": stats["emails"],
    "leftover_phones": leftover_p,
    "leftover_emails": leftover_e,
}, ensure_ascii=False))
