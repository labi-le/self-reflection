import torch
import transformers
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
print("transformers", transformers.__version__, flush=True)
print("loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("loading model bf16...", flush=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
).to("cuda:0")
print("loaded via AutoModelForCausalLM -> cuda:0 (eager attn)", flush=True)


def ask(q, thinking):
    msgs = [{"role": "user", "content": q}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.8)
    resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n=== Q(" + ("think" if thinking else "chat") + "): " + q + " ===", flush=True)
    print(resp.strip(), flush=True)


ask("привет, как оно? чем занят?", False)
ask("задачка: у Дениса было 3 яблока, съел одно, купил ещё 5, потом отдал половину другу — сколько осталось? рассуждай по шагам", True)
ask("объясни простыми словами, почему небо голубое", False)
print("\n=== DEMO DONE ===", flush=True)
