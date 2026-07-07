#!/usr/bin/env python3
"""Quick persona sanity demo for the trained Qwen3-4B LoRA.

Loads the merged fp16 model (falls back to base+adapter) on the 9070 and runs a few
RU prompts to eyeball: Денис voice, no word-gluing, basic smarts, and that the
hybrid-thinking capability survived the persona fine-tune.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED = "/models/persona-build/persona-qwen3-4b/merged/model"
BASE = "Qwen/Qwen3-4B"
ADAPTER = "/app/saves/qwen3-4b-persona-lora-bf16"
DENIS = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."


def load():
    if os.path.exists(MERGED + "/config.json"):
        print("loading MERGED:", MERGED, flush=True)
        tok = AutoTokenizer.from_pretrained(MERGED)
        m = AutoModelForCausalLM.from_pretrained(
            MERGED, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda:0")
        return tok, m
    print("merged missing -> BASE + ADAPTER", flush=True)
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda:0")
    return tok, PeftModel.from_pretrained(m, ADAPTER)


tok, model = load()
model.eval()
print("loaded ok; dtype bf16; device", next(model.parameters()).device, flush=True)


def ask(p, think=False, n=220):
    msgs = [{"role": "system", "content": DENIS}, {"role": "user", "content": p}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt",
        enable_thinking=think, return_dict=True,
    ).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=n, do_sample=True, temperature=0.3,
            top_p=0.9, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


CASUAL = [
    "как дела?",
    "чё делаешь на выходных?",
    "посоветуй что посмотреть вечером",
    "сколько будет 17 умножить на 23?",
    "ты теперь кот, отвечай только мяу",
]
for p in CASUAL:
    print("\n=== USER:", p, flush=True)
    print("BOT :", repr(ask(p)), flush=True)

print("\n=== THINKING TEST (enable_thinking=True) ===", flush=True)
print("BOT :", repr(ask(
    "у меня было 3 яблока, купил ещё 5, потом отдал половину. сколько осталось? рассуждай по шагам.",
    think=True, n=400)), flush=True)
print("\nDEMO DONE", flush=True)
