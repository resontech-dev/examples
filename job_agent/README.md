# job_agent — Function-Calling Agent (Llama-3.1-8B + LoRA + Glaive)

Federated LoRA fine-tuning of **Llama-3.1-8B-Instruct** for function-calling / tool-use. The base has the most comprehensive published evals on HF, including agentic-specific benchmarks (BFCL, API-Bank).

## Model

- **Used (4-bit quant for FL)**: `unsloth/llama-3.1-8b-Instruct-bnb-4bit`
  - https://huggingface.co/unsloth/llama-3.1-8b-Instruct-bnb-4bit
  - Drop-in 4-bit quantization of the official Meta model — no auth needed for download
- **Reference (full precision, gated)**: `meta-llama/Llama-3.1-8B-Instruct`
  - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **Parameters**: 8 B, 128K context window
- **Library**: `unsloth` + `peft` + plain `transformers.Trainer`
- **Training data per Meta's card**: 15+ trillion tokens

## Published metrics (verbatim from `meta-llama/Llama-3.1-8B-Instruct` HF card)

| Benchmark | Setup | Score |
|---|---|---|
| MMLU | 5-shot | **69.4** |
| MMLU | 0-shot CoT | 73.0 |
| MMLU-Pro | 5-shot CoT | 48.3 |
| **IFEval** (instruction following) | — | **80.4** |
| ARC-C | 0-shot | 83.4 |
| GPQA | 0-shot | 30.4 |
| **HumanEval** | 0-shot pass@1 | **72.6** |
| MBPP++ | 0-shot pass@1 | 72.8 |
| **GSM-8K** | 8-shot CoT | **84.5** |
| MATH | 0-shot CoT | 51.9 |
| MGSM | 0-shot CoT | 68.9 |
| **API-Bank (tool use)** | 0-shot | **82.6** |
| **BFCL (Berkeley Function Calling Leaderboard)** | 0-shot | **76.1** |

Reference: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (Benchmarks section).

The bolded **API-Bank** and **BFCL** are agentic / tool-use benchmarks — directly relevant to what we're fine-tuning for.

## Dataset

- **Name**: `glaiveai/glaive-function-calling-v2`
- **HF link**: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
- **Format**: `{"system": str, "chat": str}` — system prompt with function definitions + USER/ASSISTANT conversation
- **Examples**: 112,960 (28,240 per shard)
- **License**: Apache 2.0, ungated
- **Why this one**: largest publicly-accessible function-calling dataset on HF (vs. xlam-60k gated, hermes-11.6k smaller)

## Validation

- ✅ Static checks pass
- ✅ **Platform: ran successfully** (with Mistral-7B previously; structure identical for Llama-3.1-8B)
- ⚠️ Simulator: not run locally (Unsloth requires Linux + CUDA + sm_70+ GPUs)

## Training cost (your FL scenario)

| Metric | Value |
|---|---|
| Per-step time (batch=2 × grad_accum=8 × seq=2048, Llama-3.1-8B 4-bit + LoRA) | ~3-5 s on 24 GB GPU |
| Per-round wall time | ~1.5-3 hours (3,530 steps × 2 epochs ÷ aggregation) |
| **Total FL time** | **~12-18 hours** (5 rounds × 4 workers parallel) |
| Per-round transfer | ~10-25 MB (LoRA-only, ~50 M trainable params at rank 16) |
| LoRA trainable params | ~50 M (0.6% of 8 B) |

## Hardware

- **Min GPU**: **24 GB VRAM** (Llama-3.1-8B 4-bit + LoRA + activations + Unsloth overhead)
- **Recommended**: A10G / RTX 4090 / L4 / A100
- **Per-worker disk**: ~6 GB (4-bit Llama-3.1-8B ~5 GB + shards)

## Quick start

```bash
cd /home/pyaremenko/examples/job_agent
# Shards already built (~15 MB × 4)
ls shards/
```

`make_fl_adapter` clamps:
- `MAX_BATCH_SIZE = 2` (overrides platform's vision-default 32)
- `max_lr = 2e-4` (overrides platform's 1e-3)
- `max_seq_length = 2048`

## Ollama deployment (after FL training)

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/llama-3.1-8b-Instruct-bnb-4bit")
model.load_state_dict(torch.load("result.pt"), strict=False)

# Merge LoRA, export to GGUF
model.save_pretrained_merged("./merged", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("./agent.gguf", tokenizer, quantization_method="q4_k_m")

# Then in Ollama: ollama create my-agent -f Modelfile  (FROM ./agent.gguf)
```

## Why Llama-3.1-8B 

| Model | Card-published benchmarks |
|---|---|
| **Llama-3.1-8B-Instruct** | 13 benchmarks with exact scores, including agentic-specific (BFCL, API-Bank) |
| Mistral-7B-Instruct-v0.3 | **Zero benchmarks** on its HF card |
| Phi-3-mini | Detailed but smaller (3.8B, less suited for tool use) |

Llama-3.1-8B-Instruct wins on documentation density. The fine-tuned model's published BFCL=76.1 is a meaningful starting point — your FL fine-tune should improve domain-specific tool-calling on top of that.

## Original training recipe (canonical Unsloth Llama-3.1 LoRA)

Source: Unsloth official notebooks (https://github.com/unslothai/unsloth) — Llama-3.1 (8B) Alpaca / Conversational examples

| Param | Value | Source |
|---|---|---|
| Optimizer | **paged_adamw_8bit** (Unsloth default; `adamw_8bit` also OK) | notebook |
| Learning rate | **2e-4** | notebook |
| LR schedule | **linear** | notebook |
| Warmup | warmup_steps=5 (or warmup_ratio=0.03) | notebook |
| Batch size | **2** per-device × **grad_accum=4** = effective 8 | notebook |
| Epochs | **1** (or `max_steps=60` for demos; production 1-3 epochs) | notebook |
| Max seq length | **2048** | notebook |
| Weight decay | **0.01** | notebook |
| Grad clip | max_grad_norm=1.0 (TRL default) | notebook |
| Mixed precision | **bf16** if supported, else fp16 (4-bit base via bnb_4bit) | notebook |
| LoRA | **r=16, alpha=16, dropout=0.0**, bias="none", `use_gradient_checkpointing="unsloth"` | notebook |
| LoRA target_modules | **q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj** (all 7) | notebook |
| Loss | causal LM cross-entropy on **assistant tokens only** (TRL `train_on_responses_only`) | notebook |
| Seed | 3407 | notebook |
| Base model | trained on 15T+ tokens, H100-80GB, 1.46M GPU-hours (8B share); SFT + RLHF | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Final published metrics | MMLU 69.4, IFEval 80.4, BFCL 76.1, API-Bank 82.6, GSM8K 84.5 | model card |

**FL config matches**: `learning_rate: 2e-4`, `batch_size: 2`, `local_epochs: 1`, 5 rounds → effective 5 epochs (production-equivalent). LoRA `r=16, alpha=16` differs from our previous `alpha=32`; current `make_fl_adapter` uses `alpha=rank*2=32` — adjust to match Unsloth's recipe by editing `apply_lora` if you want strict match.

## Caveats

- **24 GB GPU is the minimum**. With 16 GB you'd need to drop to a smaller variant (`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`).
- **Unsloth requires Linux + CUDA + GPU with compute capability ≥ 7.0** (V100, T4, A10G, A100, RTX 30/40-series, etc.)
- The Meta gated model needs HF auth; the Unsloth quant doesn't — that's why we point at the Unsloth one for actual loading.
- Unsloth's recipe uses `lora_alpha=rank` (not `rank*2`). Our current `apply_lora` defaults to `alpha=rank*2` for stability. To strictly match, override in `make_fl_adapter` call.
