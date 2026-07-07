import torch, transformers, traceback
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
print("loading", MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
).to("cuda:0")
print("loaded to cuda:0", flush=True)

ids = tok("привет, как дела?", return_tensors="pt").to("cuda:0")

try:
    print("--- bare forward (no_grad)...", flush=True)
    with torch.no_grad():
        out = model(**ids)
    print("FORWARD OK logits", tuple(out.logits.shape), flush=True)
except Exception:
    traceback.print_exc()

try:
    print("--- train forward+backward...", flush=True)
    model.train()
    out2 = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"], labels=ids["input_ids"])
    print("LOSS", float(out2.loss), flush=True)
    out2.loss.backward()
    print("BACKWARD OK (training path works)", flush=True)
except Exception:
    traceback.print_exc()

print("=== isolate done", flush=True)
