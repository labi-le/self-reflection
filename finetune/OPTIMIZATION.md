# Training speedup — experiments to test later

Grounded in a live measurement of the current QLoRA run.
**Every lever here needs a *fresh* run to test — none can be applied to the in-flight
training.** Benchmarks also need the GPU free, so run them **after the current training
finishes** (or later on the 6700XT box).

## Baseline (measured Sat 2026-07-04 ~17:00, mid-run @ ~step 4400/9102)

| Signal | Value | Reading |
|---|---|---|
| GPU busy | **97%** (`rocm-smi --showuse`) | compute-bound |
| VRAM | **15.4 / 17.1 GB** (~1.7 GB free) | **memory is the binding constraint** |
| CPU | load **3.3 / 16** cores (~20%) | idle — *healthy* for a GPU-bound run |
| Throughput | ~200–250 steps / 30 min ≈ **7–8 s/step** | bs1×ga8, seq≤1024, 58/64 CUs |

Config: `configs/qlora_qwen25_7b_instruct.yaml` · env: `../docker/docker-compose.yml`

## Why the CPU can't be "utilized more" (settled)

GPU is at 97%; the CPU is idle **because** the GPU is the bottleneck and the dataloader
(`preprocessing_num_workers: 8`, `dataloader_num_workers: 2`) trivially keeps it fed. You
cannot offload 7B transformer matmuls to the CPU (≈100× slower per op + PCIe stalls would
only *stall* the GPU). The one "CPU helps" trick — optimizer-state offload to RAM — frees
nothing for **LoRA** (the optimizer state is tiny; offload only matters for full
fine-tuning). Idle CPU here = correct, not wasted.

## How to benchmark any change (do this per lever)

1. GPU must be free (after the current run) — only ~1.7 GB is spare while training runs.
2. Make a throwaway copy of the yaml with: `max_steps: 100`, `save_steps: 100000`,
   `eval_strategy: "no"`, `report_to: none`, same `seed: 42`, same data.
3. Run:
   `docker exec tg2llm-finetune bash -lc 'llamafactory-cli train finetune/configs/<variant>.yaml'`
4. Record **`train_samples_per_second`** (printed at the end) **+ peak VRAM**
   (`watch -n1 "docker exec tg2llm-finetune rocm-smi --showmeminfo vram"`).
5. Compare against a baseline variant (identical, but current settings).
   - ⚠️ For **packing**, compare **samples/sec or time-per-epoch**, *not* steps/sec —
     packing changes what a "step" is.
   - ⚠️ Only ~1.7 GB VRAM headroom — several levers can OOM; watch it.

---

## Lever 1 — sample packing   *(biggest potential, most uncertain)*

- **Now:** OFF (no `packing` key → LLaMA-Factory default `false`).
- **Change:** `packing: true` + `neat_packing: true` (neat = keep per-sample attention
  masked so packed chats don't attend across boundaries).
- **Honest expectation:** I earlier ballparked ×2–5 assuming *padding waste* — **that
  assumption is wrong at `per_device_train_batch_size: 1`**: a 1-sequence batch has nothing
  to pad against, so there is no padding compute to reclaim. The real mechanism is
  different — packing concatenates several short chats into one ~1024-token window →
  **fewer steps / optimizer updates** and **larger, more efficient matmuls per step**.
  Note `rocm-smi`'s 97% is *occupancy*, not FLOP-efficiency, so there may still be real
  headroom on short samples — but the gain is genuinely unknown. **Measure, don't assume.**
- **Risks:** longer sequences → **higher per-step VRAM** (OOM risk given ~1.7 GB free);
  many-more-tokens-per-update → effective batch (in tokens) grows → may need **LR / epoch
  retune**; changes final quality → **must eval the resulting adapter**, not just throughput.

## Lever 2 — free the 6 reserved CUs

- **Now:** `HSA_CU_MASK: "0:0-57"` → training on **58/64 CUs (~90%)**; 6 CUs held back for
  the desktop compositor.
- **Change:** drop the mask → all 64 CUs.
- **Expected:** ~**+10%** compute (near-linear in CU count).
- **Risk:** desktop stutter / hang during the run. Only worthwhile if the desktop is unused
  while training. Needs restart.

## Lever 3 — flash attention

- **Now:** `flash_attn` unset (LLaMA-Factory `auto` → SDPA). `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  is already set, so an AOTriton flash-attention path *via* SDPA may already be active.
- **Change:** try `flash_attn: fa2` — but the `flash-attn` package likely **won't build for
  gfx1201/RDNA4**; expect a fallback. Alternatively just confirm the current SDPA/AOTriton
  path is the efficient one.
- **Expected:** uncertain; possibly already covered. Low priority.

---

## Off the table (recorded so we don't retry)

- **CPU utilization** — pointless (see above).
- **Bigger `per_device_train_batch_size` / disabling `gradient_checkpointing`** — no VRAM
  for it (~1.7 GB free; that scarcity is *why* checkpointing is on and bs=1 in the first place).
- **torch.compile** — deliberately OFF (ROCm inductor is a known NaN source on gfx11xx;
  see the config comment).
- **`HSA_XNACK` / `HSA_ENABLE_SDMA` / `GPU_MAX_HW_QUEUES`** — load-bearing gfx1201 GPU-hang
  fixes. Do **not** touch these for speed; they are the reason the run is stable.

## Verdict for the CURRENT run

Let it finish (~7 h). GPU 97% + VRAM ~90% = the hardware is already saturated; there is no
idle capacity to reclaim mid-run. Restarting to reconfigure risks re-triggering the
gfx1201 GPU-hangs (hard-won stability) and loses progress to re-warmup — a bad trade for
modest / uncertain gains. Revisit every lever above on the **next** training run.
