# Persona Fine-Tuning Guide (tg2llm)

How we build the **"Денис" persona model** end-to-end: from a raw Telegram export to a
quantized GGUF served behind the Telegram bot. Every stage, with the exact scripts, configs,
and — most importantly — the **AMD RX 9070 XT (gfx1201 / RDNA4) GPU fix** that took days to
find.

> Target GPU: **AMD Radeon RX 9070 XT, 16 GB, gfx1201 / RDNA4**, ROCm.
> Trainer: **LLaMA-Factory**. Server: **llama.cpp** (`server-vulkan`).
> Everything runs in the **`tg2llm-finetune`** Docker container; nothing needs the host Python.

---

## 0. Pipeline at a glance

```
result.json                 Telegram/AyuGram export (repo root)
   │  parse + group into ShareGPT conversations
   ▼
finetune/data/train.jsonl                 24 762 persona convos (raw, has PII + burst-delimiters)
   │  scrub_pii_spaced.py   (PII redaction + un-glue bursts)
   ▼
finetune/data/train.scrubbed.spaced.jsonl
   │  assemble_premium.py   (blend 70% persona + 30% restyled RU instruct)
   ▼
finetune/data/train.premium.jsonl         15 650 rows, PII-clean, seed 42
   │  build_subset.py       (optional: random-3000 subsample for fast iteration)
   ▼
finetune/data/train.q3fast.jsonl          3 000 rows  ← fast-iteration dataset
   │  LLaMA-Factory  (QLoRA / bf16 LoRA on the 9070)
   ▼
finetune/output/<run>/adapter_model.safetensors    LoRA adapter
   │  scripts/build_gguf.sh (merge → convert → quantize, CPU-only)
   ▼
…/persona-<name>.Q4_K_M.gguf              served by llama.cpp → the Telegram bot
```

**Why the two-stage blend?** A naïve fine-tune on raw chat logs (short texts, 3 epochs, no
instruct data) causes **catastrophic forgetting** — the model gets the voice but goes "dumb".
Mixing in **30% instruction data restyled into the persona's casual voice** preserves reasoning
while still learning the style. This "premium" recipe is the whole reason the model can both
sound like Денис *and* answer a real question.

---

## 1. The GPU fix (read this first — it is the crown jewel)

On **gfx1201 / RDNA4** with the stock ROCm 7.2.3 libraries, LoRA training **crashes** a few
dozen–hundred steps in:

```
Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
```

### Root cause (confirmed, upstream [ROCm/rocm-libraries#7992](https://github.com/ROCm/rocm-libraries/issues/7992))

The rocBLAS **Tensile GEMM tile `…MT64x64x64_ISA1201…`** computes an address ~1 GB **outside**
the operand buffer when the `B` matrix is **column-major**. That column-major-B GEMM is
**intrinsic to the backward pass** (the weight-gradient `xᵀ·dY`). So:

- **Inference and a single forward/backward are stable** (no column-major-B GEMM), but
- **sustained training faults** — non-deterministically (step 8 with 4-bit bnb, step ~110 in
  bf16) because the faulting tile is only selected for some per-microbatch token counts.

It is **not** a VRAM/headroom problem. Giving it more memory does nothing.

### The fix — install the patched ROCm libraries (TheRock nightly)

The fix landed upstream ([ROCm/rocm-systems#6926](https://github.com/ROCm/rocm-systems)) and
ships in the **gfx120X nightly** wheels. Inside the container:

```sh
/opt/venv/bin/pip install --force-reinstall --pre \
  --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ \
  torch torchvision torchaudio
```

This pulls **`torch 2.13.0a0+rocm7.13`** and, crucially,
**`rocm-sdk-libraries-gfx120X-all-7.13`** (the patched rocBLAS/hipBLASLt).

**One follow-up is required:** the container's prebuilt **`apex`** (compiled against torch
2.7.1) breaks with an ABI/`undefined symbol` error under torch 2.13. Remove it — transformers
falls back to its own (correct) RoPE:

```sh
/opt/venv/bin/pip uninstall -y apex
```

Harmless noise you can ignore: `Configured ROCm binary not found …libbitsandbytes_rocm83.so`
— bitsandbytes is only probed, and we train in **bf16 with no quantization**, so it is never used.

**Verify before training** (GPU compute + stack imports):

```sh
rtk docker exec tg2llm-finetune /opt/venv/bin/python -c "
import torch; print(torch.__version__, torch.cuda.is_available())
x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); print(float((x@x).sum()))
import transformers, datasets, peft, trl, accelerate, llamafactory; print('stack OK')"
```

### Rollback (if a nightly ever regresses the stack)

The original torch is a **custom local build** (`2.7.1+rocm7.2.3.lw…`) that is **not on any
public index**, so the reliable rollback is to **recreate the container from its image**, which
has that torch baked in — all data/models/configs live on host mounts and survive:

```sh
rtk docker rm -f tg2llm-finetune
# re-run your `docker run …` for image tg2llm-finetune:latest with the same mounts
```

Keep a package snapshot too: `pip freeze > /tmp/tg2llm_pipfreeze_before.txt`.

**Durability tip:** once training is proven, bake the nightly libs into the image so they
survive a container recreate:

```sh
rtk docker commit tg2llm-finetune tg2llm-finetune:nightly-rocm713
```

### Two hard rules learned the hard way

1. **Never set `HSA_OVERRIDE_GFX_VERSION=12.0.0`.** It *segfaults* native gfx1201 torch. The
   card is natively supported — the only env you need is `HIP_VISIBLE_DEVICES=0`.
2. **`optim: adamw_torch` only.** Paged/8-bit bnb optimizers silently produce `loss=0`/NaN on
   gfx1201.

If you ever can't patch the libs, the fallbacks (in order) are: LoRA `r64→r32` (shape-dodges the
buggy tile), a checkpoint-resume supervisor (`save_steps 50`, restart+`resume_from_checkpoint`
after each fault — safe because a *faulted* process dies clean and frees VRAM), or drop to
**Qwen3-1.7B** (different GEMM shapes miss the tile).

---

## 2. Environment / container

The `tg2llm-finetune` container holds the Python/ROCm/LLaMA-Factory stack. Persistent layout:

| Host | In container | Purpose |
|---|---|---|
| `/mnt/ssd2tb/llm/models` | `/models` | model cache; `HF_HOME=/models/hf` (survives recreate) |
| `<repo>/finetune` | `/app/finetune` | configs + data + dataset registry |
| `<repo>/finetune/output` | `/app/saves` | training outputs / checkpoints / logs |

- Python: **`/opt/venv/bin/python`** (3.12). `docker exec … python - <<EOF` gives **no stdin** —
  use `python -c "…"` or a script file.
- GPU-idle gate before launching: `cat /sys/class/drm/card1/device/mem_info_vram_used` should be
  **< 2 000 000 000** (≈ 2 GB). Activity: `…/gpu_busy_percent`.

---

## 3. Stage 1 — Create the dump (`result.json`)

Export your Telegram history as **JSON** (the parser reads the standard Telegram Desktop /
AyuGram export shape):

1. **Telegram Desktop → Settings → Advanced → Export Telegram data.**
2. Select the chat(s), set format to **Machine-readable JSON**, include text (media optional —
   the persona model is text-only).
3. This produces **`result.json`**. Place it at the **repo root** (`<repo>/result.json`).

> Privacy: this file contains real private messages. It is git-ignored; never commit it.

---

## 4. Stage 2 — Parse into ShareGPT conversations (`train.jsonl`)

The export is grouped into **ShareGPT/OpenAI "messages"** conversations: the persona's own
messages become `assistant` turns, the other party's become `user` turns, under one fixed
system prompt. Output: **`finetune/data/train.jsonl`** (24 762 convos), one JSON object per line.

**Locked schema** (full contract in [`finetune/README.md`](../finetune/README.md)):

```json
{"messages":[
  {"role":"system","content":"Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."},
  {"role":"user","content":"привет, как дела?"},
  {"role":"assistant","content":"да норм, сижу кодю)"}
]}
```

Rules: optional single leading `system` turn; then strictly alternating `user`→`assistant`; the
line **must end on an `assistant` turn** (that final content is the supervised target). The
system prompt is **byte-identical** to what the bot injects at inference — do not drift it.

---

## 5. Stage 3 — PII scrub + un-glue bursts (`scrubbed.spaced.jsonl`)

Script: [`finetune/scripts/dataprep/scrub_pii_spaced.py`](../finetune/scripts/dataprep/scrub_pii_spaced.py)
(run from the repo root). It reads `train.jsonl` → writes `train.scrubbed.spaced.jsonl` and does
two jobs:

**(a) PII redaction** (regex, per turn):

| Pattern | → replacement |
|---|---|
| `\+?\d{11,}` (intl phones / long IDs) | `[phone]` |
| `\+?[78]\d{10}` (RU phones) | `[phone]` |
| `\b9\d{9}\b` (bare 10-digit mobiles) | `[phone]` |
| e-mail addresses | `[email]` |
| personal domain (e.g. `example.com`) | `[domain]` |

**(b) Un-glue message bursts.** Telegram exports separate rapid-fire messages with **invisible
delimiters** (`U+2063` and friends). The *original* scrub stripped them, which glued
`word1word2` together. This version **replaces each invisible with a space** (`U+2060–2064`,
`U+200B–200D`, `U+FEFF`), then collapses runs of horizontal whitespace. This is what fixed the
"word-gluing" the model exhibited.

The script prints a BEFORE/AFTER audit and asserts **0 remaining `U+2063`** and **0 real PII**
(checked against a known list of the owner's actual numbers/domain).

---

## 6. Stage 4 — Restyle instruct + assemble the blend (`train.premium.jsonl`)

### 6a. Restyle instruction data into the persona voice

Script: [`finetune/scripts/dataprep/restyle_batch.py`](../finetune/scripts/dataprep/restyle_batch.py).
Takes public RU instruction datasets (saiga-scored + GrandMaster-PRO-MAX answers) and rewrites
each answer **in Денис's casual register** using a local model on the GPU → `premium_raw/
instruct_restyled.jsonl`. The un-restyled originals are kept in `premium_raw/instruct_subset.jsonl`.

### 6b. Assemble the 70/30 blend

Script: [`finetune/scripts/dataprep/assemble_premium.py`](../finetune/scripts/dataprep/assemble_premium.py)
→ **`finetune/data/train.premium.jsonl`** (15 650 rows, `seed 42`). It:

- **Cleans each restyled answer:** `flatten_md()` strips markdown scaffolding (headings, lists,
  bold, formal openers like "отличный вопрос!") into inline prose — a markdown answer under a
  "личная переписка" prompt is an instant AI-tell — while **protecting fenced/inline code**;
  then lowercases (Денис writes lowercase).
- **Drops artifacts:** script-mixed words (Cyrillic+Latin in one token, e.g. `петrarка`) and
  answers with a **fabricated ≥3-digit number** vs the original (number-integrity check).
- **Keeps a 15% "reasoning anchor":** that fraction uses the *original* (non-restyled) answer,
  still wrapped as Денис, so the blend doesn't wash out reasoning quality.
- **Balances to 70/30:** subsamples persona rows so instruct = 30% of the final blend, wraps
  everything under the single Денис system prompt, shuffles, and prints a **PII sanity check**
  (expects 0 phone/email in the persona half).

> Known data nit: 6 premium rows have unescaped literal newlines (an email/form record split
> across physical lines). `build_subset.py` skips them via a `json.loads` filter; for a full run,
> re-escape newlines in `assemble_premium.py`.

---

## 7. Stage 5 — Fast-iteration subsample (`train.q3fast.jsonl`)

For quick model/recipe validation you don't need all 15 650 rows. Script:
[`finetune/scripts/dataprep/build_subset.py`](../finetune/scripts/dataprep/build_subset.py) →
**`train.q3fast.jsonl`** = a **random 3 000-row** (`seed 42`) subsample of the valid premium
rows. This trains in **~30 min** on the 9070 instead of hours — ideal for "does this base model
even work" passes.

> Want a **fresh, time-boxed** dataset instead (e.g. the last month)? The Go parser
> (`cmd/parse`) accepts `--date-from YYYY-MM-DD` / `--date-to` — re-parse `result.json` for the
> window, then re-run stages 3–4. The instruct half is reused unchanged.

---

## 8. Stage 6 — Register the dataset

LLaMA-Factory resolves datasets from **`<dataset_dir>/dataset_info.json`**. With
`--dataset_dir /app/finetune`, the registry lives at `finetune/dataset_info.json` and each entry's
`file_name` is relative to `finetune/`:

```json
{
  "persona_sharegpt_q3fast": {
    "file_name": "data/train.q3fast.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role", "content_tag": "content",
      "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
    }
  }
}
```

Siblings already registered: `persona_sharegpt` (full raw), `persona_sharegpt_premium` (the 15 650
blend), `persona_sharegpt_q3fast` (the subsample).

---

## 9. Stage 7 — The training config

Current run: **`finetune/configs/qlora_qwen3_4b_persona_bf16.yaml`**.

| Setting | Value | Why |
|---|---|---|
| `model_name_or_path` | `Qwen/Qwen3-4B` | Fits 16 GB **in bf16 with no quantization** → dodges the bnb path entirely. Standard GQA attention (Qwen3.5/3.6's Gated-DeltaNet linear attention needs kernels absent on ROCm). Hybrid thinking, strong RU. |
| *(no `quantization_bit`)* | — | **bf16, no quant.** ~8 GB weights + LoRA fit 16 GB. |
| `bf16` | `true` | compute dtype |
| `template` | `qwen3_nothink` | Casual chat → train without `<think>` traces; thinking stays available at inference via `enable_thinking=True`. |
| `finetuning_type` / `lora_rank` / `lora_alpha` | `lora` / `64` / `16` | adapter capacity |
| `lora_target` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | attention+MLP only; **exclude** `lm_head`/`embed_tokens` |
| `cutoff_len` | `512` | seq cap; 512 (not 1024) avoids an eager-attention VRAM-thrash hang |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | `1` / `8` | effective batch 8, tiny footprint |
| `gradient_checkpointing` | `true` | trade compute for memory |
| `optim` | **`adamw_torch`** | **mandatory** on gfx1201 (bnb/8-bit optimizers → NaN) |
| `flash_attn` | `disabled` | eager attention (math SDPA path is the stable one on gfx1201) |
| `num_train_epochs` / `learning_rate` | `3` / `5e-5` | cosine, `warmup_ratio 0.03` |
| `seed` | `42` | reproducibility |
| `torch.compile` | **off** | ROCm inductor is a known NaN source |

> For a **QLoRA (4-bit) Qwen2.5** run, the older config
> `configs/qlora_qwen25_7b_instruct.yaml` (+ `quantization_bit: 4`, `quantization_method: bnb`,
> `template: qwen`) is the reference — but on gfx1201 the 4-bit path needs the **patched libs**
> from §1 or it faults at step 8.

Validate cheaply before touching the GPU: `bash finetune/validate_config.sh`.

---

## 10. Stage 8 — Launch training

GPU must be idle (§2). Launch detached, **without** `HSA_OVERRIDE_GFX_VERSION`:

```sh
rtk docker exec -d tg2llm-finetune bash -lc '
  cd /app && unset HSA_OVERRIDE_GFX_VERSION;
  HIP_VISIBLE_DEVICES=0 HSA_ENABLE_COREDUMP=0 \
  /opt/venv/bin/llamafactory-cli train finetune/configs/qlora_qwen3_4b_persona_bf16.yaml \
  > /app/saves/qwen3_4b_bf16_train.log 2>&1'
```

Monitor (the log is at `finetune/output/qwen3_4b_bf16_train.log`):

```sh
python3 -c "
import os; L=open('finetune/output/qwen3_4b_bf16_train.log',errors='replace').read().splitlines()
p=[l for l in L if 's/it]' in l or 'it/s]' in l]
print('last:', p[-1].strip() if p else 'none', '| FAULT:', any('Memory access fault' in l for l in L))"
```

Healthy signs: progress advances past step ~110 (the old fault point), loss decreasing
(≈2.8→2.3), no NaN, VRAM ~13 GB, GPU busy ~95%. ~1104 steps ≈ 30 min for the q3fast set.

---

## 11. Stage 9 — Merge adapter → GGUF

Script: [`finetune/scripts/build_gguf.sh`](../finetune/scripts/build_gguf.sh)
`<adapter_dir_host> <out_basename>`. **CPU-only, offline, memory-capped** — never touches the
GPU, so it can't disturb training/serving. It:

1. **Merges** the LoRA into fp16 via `llamafactory-cli export` (in `tg2llm-finetune`).
2. **Converts** HF → GGUF f16 with `convert_hf_to_gguf.py` (in `llama.cpp:full`), first restoring
   the base tokenizer files (LLaMA-Factory re-saves `tokenizer_config.json` in a shape the older
   llama.cpp transformers can't parse; the LoRA never touched the tokenizer).
3. **Quantizes** f16 → **Q4_K_M**.
4. **Verifies** GGUF magic + size (no unattended inference — it can loop/OOM).

> **Qwen3 note:** `build_gguf.sh` currently hardcodes the *Qwen2.5* base in its `merge.yaml`
> (`model_name_or_path: Qwen/Qwen2.5-7B-Instruct`, `template: qwen`). For the Qwen3-4B adapter,
> change those to **`Qwen/Qwen3-4B`** and **`template: qwen3_nothink`**, and point the
> tokenizer-restore snapshot at the `Qwen3-4B` cache dir. (Parameterizing this is a TODO.)

Output: `/mnt/ssd2tb/llm/models/persona-build/<name>/gguf/<name>.Q4_K_M.gguf`.

---

## 12. Stage 10 — Serve + deploy to the bot

- **Serve** the GGUF with llama.cpp `server-vulkan` (see `docker/`), exposing an OpenAI-style
  `/v1/chat/completions`.
- **Point the bot** at it via `--llama-host`. The bot injects the same locked persona system
  prompt, applies anti-loop sampling (`temperature 0.3`, `repeat_penalty 1.1`,
  `presence_penalty 0.3`), redacts PII, and splits replies into human-paced bubbles.
- **Thinking on demand:** the persona is trained `qwen3_nothink`, so casual replies carry no
  `<think>` traces; for a hard query you can re-render with `enable_thinking=True` and strip the
  `<think>…</think>` block before display.
- **Vision** is handled by a *separate* describer model (e.g. Qwen2.5-VL / Qwen3-VL) that turns a
  photo into `[картинка: …]` text — the persona brain stays text-only.

---

## 13. gfx1201 / RDNA4 troubleshooting table

| Symptom | Cause | Fix |
|---|---|---|
| `Memory access fault … Page not present` mid-training | rocBLAS Tensile tile OOB on column-major-B backward GEMM (ROCm#7992) | **Install TheRock gfx120X nightly torch** (§1) |
| Segfault at model load, no traceback, VRAM frees | `HSA_OVERRIDE_GFX_VERSION=12.0.0` kills native gfx1201 | **Unset it**; use only `HIP_VISIBLE_DEVICES=0` |
| `loss = 0` or NaN immediately | bnb/paged 8-bit optimizer on gfx1201 | `optim: adamw_torch` |
| `apex … undefined symbol` after nightly install | apex built against old torch | `pip uninstall -y apex` (native RoPE takes over) |
| `libbitsandbytes_rocm83.so not found` | bnb probed but unused in bf16 | **ignore** (harmless) |
| Hang (not fault) at step ~8, cutoff 1024 | eager-attention seq² VRAM thrash | `cutoff_len: 512` |
| Silent process death loading Qwen3.5/3.6 | Gated-DeltaNet linear attention needs `flash-linear-attention`/`causal-conv1d` (absent on ROCm) | use a **standard-attention** model (Qwen3-4B / Qwen3-8B) |
| `NotImplementedError: QuantizationMethod.HQQ` | transformers 5.6 doesn't implement HQQ loading | use **bf16 no-quant** or bnb-4bit + patched libs |
| torch.compile NaN | ROCm inductor backend | keep compile **off** |

---

*Provenance: this guide documents the actual pipeline used to train the Денис persona on an
RX 9070 XT. Data-prep scripts live in `finetune/scripts/dataprep/`, configs in
`finetune/configs/`, the GGUF builder in `finetune/scripts/build_gguf.sh`.*
