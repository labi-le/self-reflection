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

> **Recipe evolution — read this first.** The table below is the **first** run,
> `qlora_qwen3_4b_persona_bf16.yaml` (4B, bf16 no-quant, `lora_alpha 16`). The **shipped** models
> use the **v2** recipe: `lora_alpha 128`, the cleaned **`persona_sharegpt_premium2`** dataset, an
> identity block, a unified Денис prompt, and **4-bit QLoRA** (bnb) so the 8B fits 16 GB. The best
> model is **Qwen3-8B v2** — `configs/qlora_qwen3_8b_instruct_v2.yaml`, train/eval **2.05 / 2.26**
> (the 4B v2 is 2.16 / 2.36). See **§14** for the full comparison and the serving config that
> actually works. v2 deltas vs the table below: `quantization_bit: 4` + `quantization_method: bnb`,
> `lora_alpha: 128`, `lora_dropout: 0.10`, `num_train_epochs: 2`, dataset `persona_sharegpt_premium2`.
>
> The 4-bit QLoRA path on gfx1201 also needs a bnb symlink
> `libbitsandbytes_rocm83.so → libbitsandbytes_rocm72.so` (bnb 0.49.2 ships only up to rocm72;
> nightly HIP 7.13 probes rocm83) — without it bnb fails to load.

First run (baseline): **`finetune/configs/qlora_qwen3_4b_persona_bf16.yaml`**.

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

### 11b. Low-RAM path for big adapters (the 8B): apply the LoRA at runtime, don't merge

`build_gguf.sh` step 1 merges the LoRA into an **fp16** copy of the base. For an **8B** that peaks
~18 GB RAM — which a 30 GB-RAM box with no swap can't spare — and `llama-export-lora` (the
GGUF-space merger) isn't in `ghcr.io/ggml-org/llama.cpp:full`. So the 8B ships **without a merge**:

1. `convert_hf_to_gguf.py <base-snapshot> --outtype f16` → `base-8b-f16.gguf` (streaming, low RAM).
2. `convert_lora_to_gguf.py <adapter> --base <base-snapshot> --outtype f16` → `lora-f16.gguf`.
3. `llama-quantize base-8b-f16.gguf … Q8_0` → `base.Q8_0.gguf` (~8.7 GB; `q8_0` ≈ fp16 perplexity,
   and pure fp16 at ~16 GB won't fit the 16 GB card).
4. **Serve the quantized base with the adapter applied at runtime:**
   `llama-server -m base.Q8_0.gguf --lora lora-f16.gguf …` — confirm `GET /lora-adapters` shows
   `scale: 1.0` (a no-system-prompt reply that's still in-character proves the adapter is live).

This is exactly how the 8B is served. Bonus: one quantized base can host several adapters with no
re-merge.

---

## 12. Stage 10 — Serve + deploy to the bot

- **Serve** the GGUF with llama.cpp `server-vulkan` (see `docker/`), exposing an OpenAI-style
  `/v1/chat/completions`.
- **Point the bot** at it via `--llama-host`. The bot injects the persona system prompt, applies
  the sampling in **§14** (NOT the old "anti-loop" defaults), redacts PII, and splits replies into
  human-paced bubbles — a "typing…" indicator plus a jittered pause before **every** bubble, so it
  never answers instantly.
- **Thinking on demand:** the persona is trained `qwen3_nothink`, so casual replies carry no
  `<think>` traces; for a hard query you can re-render with `enable_thinking=True` and strip the
  `<think>…</think>` block before display.
- **Vision** is a *separate* describer model (Qwen2.5-VL-7B) that turns a photo into text — the
  persona brain stays text-only.

> ⚠️ **Do not serve with the intuitive "anti-loop" sampling (low temp + repeat/presence/frequency
> penalties).** That is exactly what makes these persona models ramble into long, dark, incoherent
> monologues — the penalties strangle the turn-end token. **§14** has the why and the config that
> actually works. This was the single biggest serving lesson.

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

## 14. Retrospective — which base won, and the serving playbook

Persona bases trained on the same **v2** data (cleaned persona + restyled instruct + identity,
unified Денис prompt, `lora_rank 64` / `lora_alpha 128`) and served behind the bot:

| Base | Recipe | train / eval loss | Served as | Verdict |
|---|---|---|---|---|
| **Qwen3-8B v2** | 4-bit QLoRA, 2 ep — `qlora_qwen3_8b_instruct_v2.yaml` | **2.05 / 2.26** | `q8_0` GGUF **+ runtime `--lora`** (§11b) | **Best.** Most coherent, stays in-voice on longer answers. `q8_0` is effectively lossless. Needs the runtime-lora serve (fp16 merge won't fit 16 GB RAM). |
| **Qwen3-4B v2** | 4-bit QLoRA, α128, premium2 | 2.16 / 2.36 | `Q4_K_M` GGUF | **Good, lighter fallback.** Terser and a touch less coherent than the 8B, but very usable and it fits on the server GPU next to the vision model — no dedicated GPU needed. |
| Qwen3-4B (bf16, α16) | the original `qlora_qwen3_4b_persona_bf16.yaml` (§9) | baseline | — | First recipe; superseded by v2 (α128 + cleaned data). |

**Bottom line:** **8B v2 = best**, **4B v2 = runs-anywhere fallback**. But a persona model is only
as good as its sampling — which was the real fight:

### The sampling lesson (this took the longest to find)

These models *look* like they ramble uncontrollably — runaway length, dark drift ("нет смысла
жизни…"), name confusion. **It is not the model. It is the repetition penalties.**

- `presence_penalty` / `frequency_penalty` over a wide `repeat_last_n` window also penalise the
  **turn-end token** (`<|im_end|>`, which recurs every turn already in the context). Penalise it
  enough and the model literally **cannot stop** → it keeps generating → rambles and drifts dark.
- Proof (same model, same prompt): **with** the old anti-loop penalties, `ку` → 8–128 tokens
  (often a `finish_reason: length` runaway). **With penalties off** → every reply is 2–17 tokens,
  `finish_reason: stop`, terse and in-character.

**The config that actually works (4B and 8B alike):**

| Param | Value | Why |
|---|---|---|
| `temperature` | **0.7** | Natural. Low temp (0.3) makes it fixate/loop, not "less random". |
| `presence_penalty` | **0** | Off — this is what strangled the stop token. |
| `frequency_penalty` | **0** | Off — worse than presence (scales by count). |
| `repeat_penalty` | **1.1** | Mild anti-loop that barely touches the stop token. |
| `repeat_last_n` | **64** | Small window, so the stop token isn't repeatedly penalised. |
| system-prompt nudge | append **"Пиши коротко, 1-2 фразы, без монологов."** | Terse in-voice replies, no truncation. |
| `--n-predict` (server) | ~**80** | Optional hard safety cap; with penalties off it's rarely hit. |

Bot flags that encode it:
`--temperature 0.7 --repeat-penalty 1.1 --presence-penalty 0 --frequency-penalty 0 --repeat-last-n 64`
plus `--persona-system "<locked prompt> Пиши коротко, 1-2 фразы, без монологов."`

### Two features tried and then removed

- **Interlocutor profile (long-term memory).** A separate instruct model re-derived a "досье" on
  the chat partner and injected it into every reply. It (a) confused identity (invented/swapped
  names), (b) the extractor kept *preserving* a bad profile verbatim, and (c) fed the rambling.
  **Disabled** — the 100-turn context window is enough memory for a casual bot. (Re-enable via
  `--profile-host`; the extraction prompt was also rewritten to keep only confirmed facts — no
  "неизвестно" placeholders, no character judgements.) The `persona-embed` (bge-m3) service in
  `docker-compose.yml` was the semantic-recall backend for this and is likewise idle.
- **No-commitment filter.** A keyword deflector meant to stop the clone "promising money / signing"
  fired on ordinary chatter ("выставлю комп по 20к **рублей**"). **Removed entirely.** PII
  redaction (phone/email) stays.

### Deploy topology as run

- **8B** — main-PC GPU (RX 9070 XT), `q8_0 + --lora` on `:8090`; the bot on the LAN server points
  at `http://<main-pc-ip>:8090`. Trade-off: the main PC must stay on.
- **4B** — server GPU (6700XT) next to the vision model, `:8090` on the `botnet` network; the bot
  reaches it by container name. No main-PC dependency → the default fallback.

---

*Provenance: this guide documents the actual pipeline used to train the Денис persona on an
RX 9070 XT. Data-prep scripts live in `finetune/scripts/dataprep/`, configs in
`finetune/configs/`, the GGUF builder in `finetune/scripts/build_gguf.sh`.*
