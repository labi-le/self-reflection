# finetune/ — Qwen2.5-7B-Instruct persona LoRA (QLoRA, LLaMA-Factory)

Config + dataset registry for fine-tuning a **persona LoRA** on top of
`Qwen/Qwen2.5-7B-Instruct` using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
4-bit QLoRA (bitsandbytes NF4), tuned for an **AMD ROCm / RDNA** GPU.

This directory is intentionally **outside the Go module** — it ships no Go code and
does not touch `go.mod`. The training run itself lives in Task 9's container.

```
finetune/
├── configs/
│   └── qlora_qwen25_7b_instruct.yaml   # LLaMA-Factory train config
├── data/
│   ├── dataset_info.json               # LLaMA-Factory dataset registry (name: persona_sharegpt)
│   └── train.jsonl                     # <- produced by Tasks 5/6 (NOT committed here)
├── validate_config.sh                  # config gate (bash + jq, no Python)
└── README.md
```

---

## CONTRACT for Tasks 5/6 — the exact `train.jsonl` schema (LOCKED)

Tasks 5 and 6 (the dataset builder) **MUST** emit `finetune/data/train.jsonl` in this
exact shape. This is the frozen interface between the dataset builder and training.

### Physical format

- **JSONL**: one complete JSON object **per line**, UTF-8, LF newlines, **no BOM**,
  no trailing comma, no surrounding array. One line = one training example.
- Each line is an object with a single required key, `messages`.

### Per-line schema

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Formal rules:

- `messages` — a **non-empty array** of turn objects.
- Each turn object has exactly two keys: `role` (string) and `content` (string).
- `role` ∈ **`"system"` | `"user"` | `"assistant"`** (lowercase, exactly these values).
- `content` — a non-empty string (the message text). Never `null`, never an array.
- **Ordering** (what the `qwen` template + SFT masking assume):
  1. An **optional** single `system` turn, and if present it is **first**.
  2. Then **alternating** `user` → `assistant` → `user` → `assistant` …
  3. The conversation **MUST end with an `assistant` turn** — that final assistant
     content is the supervised target. A line whose last turn is not `assistant`
     has nothing to train on and is invalid.
- At least one `user` turn and one `assistant` turn per line (system-only or
  user-only lines are invalid).
- Multi-turn is allowed and encouraged (several user/assistant pairs per line);
  keep the alternation intact.

### Canonical examples

Single-turn (system + user + assistant):

```json
{"messages":[{"role":"system","content":"Ты — Алиса. Отвечай в её стиле."},{"role":"user","content":"привет, как дела?"},{"role":"assistant","content":"да норм, сижу кодю)"}]}
```

Multi-turn:

```json
{"messages":[{"role":"system","content":"Ты — Алиса."},{"role":"user","content":"что делаешь?"},{"role":"assistant","content":"обедаю"},{"role":"user","content":"а потом?"},{"role":"assistant","content":"потом работать"}]}
```

### How the registry maps this schema

`data/dataset_info.json` registers the dataset under the name **`persona_sharegpt`**
using LLaMA-Factory's ShareGPT/OpenAI ("messages") formatting:

```json
{
  "persona_sharegpt": {
    "file_name": "data/train.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

- `columns.messages = "messages"` → LLaMA-Factory reads each line's `messages` array.
- `tags.role_tag = "role"`, `tags.content_tag = "content"` → each turn is
  `{"role": ..., "content": ...}` (NOT ShareGPT's default `{"from": ..., "value": ...}`).
- `tags.user_tag / assistant_tag / system_tag` → the three allowed `role` values.

**If Tasks 5/6 change any key name or role value, they MUST update these tags in
lock-step — the training run parses strictly by these tags.**

---

## Training config (`configs/qlora_qwen25_7b_instruct.yaml`)

Key choices (see the YAML for the full set):

| Setting | Value | Why |
|---|---|---|
| `model_name_or_path` | `Qwen/Qwen2.5-7B-Instruct` | **Instruct**, not Base — we adapt an already chat-tuned model. |
| `stage` / `finetuning_type` | `sft` / `lora` | Supervised fine-tune, LoRA adapter. |
| `quantization_bit` / `quantization_method` | `4` / `bnb` | QLoRA, bitsandbytes NF4 4-bit base weights. |
| `lora_target` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Attention + MLP projections **only**. `lm_head` and `embed_tokens` are excluded on purpose. |
| `lora_rank` / `lora_alpha` | `64` / `16` | Adapter capacity. |
| `cutoff_len` | `1024` | Sequence cap (≤ 1024) — VRAM budget on RDNA. |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | `1` / `8` | Effective batch 8 with a tiny per-step footprint. |
| `gradient_checkpointing` | `true` | Trade compute for memory. |
| `optim` | `paged_adamw_8bit` | Paged 8-bit optimizer — the QLoRA default. |
| `num_train_epochs` / `learning_rate` | `3` / `1.0e-4` | |
| `lr_scheduler_type` / `warmup_ratio` | `cosine` / `0.03` | |
| `bf16` | `true` | bf16 compute dtype. |
| `template` | `qwen` | Qwen2.5 chat template. |
| `val_size` / `eval_strategy` / `eval_steps` | `0.02` / `steps` / `200` | 2% held-out eval every 200 steps. |
| `seed` | `42` | Reproducibility. |

> **AMD RDNA / ROCm — torch.compile OFF.** If you see NaN loss on gfx11xx (RDNA3),
> do **not** enable `torch.compile` / `use_torch_compile`. The ROCm inductor backend
> is a known NaN source there; LLaMA-Factory leaves compile off by default and this
> config keeps it off.

---

## Run steps

1. **Produce the data.** Run Tasks 5/6 to emit `finetune/data/train.jsonl` in the
   contract above.

2. **Point LLaMA-Factory at this directory.** LLaMA-Factory resolves a dataset's
   `file_name` relative to `--dataset_dir`, and reads the registry from
   `<dataset_dir>/dataset_info.json`. Because `file_name` here is `data/train.jsonl`,
   set the dataset dir to the **`finetune/` directory** and make the registry visible
   at its root, e.g.:

   ```sh
   # from the repo root
   ln -sf data/dataset_info.json finetune/dataset_info.json   # registry at dataset_dir root
   llamafactory-cli train finetune/configs/qlora_qwen25_7b_instruct.yaml \
     --dataset_dir finetune
   ```

   With `--dataset_dir finetune`, `file_name: data/train.jsonl` resolves to
   `finetune/data/train.jsonl` and `dataset: persona_sharegpt` is looked up from the
   registry. (Task 9's container wires this up; the symlink step is only needed for a
   bare local run.)

3. **Validate the config first** (cheap, no GPU):

   ```sh
   bash finetune/validate_config.sh
   ```

---

## Config gate (`validate_config.sh`)

`validate_config.sh` is a pure bash + `jq` gate (no Python). It **fails** before the
configs exist and **passes** once they are correct. It asserts:

1. `finetune/data/dataset_info.json` is valid JSON (`jq -e .`).
2. The YAML sets `quantization_bit: 4`.
3. `cutoff_len` is present and **≤ 1024**.
4. `model_name_or_path` contains **`Instruct`** (guards against the Base model).
5. `lora_target` does **not** contain `lm_head` or `embed_tokens`.

Exit `0` = all green; non-zero = at least one check failed (details on stderr).
Checks 2–5 parse the specific YAML key's value only, so explanatory comments that
mention a forbidden token never produce a false pass/fail.
