#!/usr/bin/env python3
"""Fast persona test: load base + LoRA adapter, run prompts at low temp, report
whether each reply STOPPED naturally (emitted eos before max tokens) and the text.

Usage: quick_test.py <adapter_dir> [BASE_MODEL]
  BASE_MODEL default Qwen/Qwen3-4B
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-4B"
SYS = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."
MAXTOK = 120

tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0")
m = PeftModel.from_pretrained(m, ADAPTER)
m.eval()
imend = tok.convert_tokens_to_ids("<|im_end|>")
eos_ids = {tok.eos_token_id, imend}
print(f"loaded {BASE} + {ADAPTER}", flush=True)

PROMPTS = [
    "привет, как дела?",
    "ты бот?",
    "как тебя зовут?",
    "что делал на выходных?",
    "чем занимаешься по жизни?",
    "17 на 23 сколько будет?",
]
for u in PROMPTS:
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": u}],
        add_generation_prompt=True, return_tensors="pt", enable_thinking=False, return_dict=True,
    ).to("cuda:0")
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAXTOK, do_sample=True, temperature=0.4,
                         top_p=0.9, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    gen = out[0][n_in:]
    stopped = (len(gen) < MAXTOK) or (gen[-1].item() in eos_ids)
    txt = tok.decode(gen, skip_special_tokens=True).strip()
    print(f"U:{u}\n  STOP={stopped} len={len(gen)} | {repr(txt[:160])}", flush=True)
