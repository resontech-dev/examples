# job_lm_scratch — Pythia-1B from scratch on The Pile (full pretraining)

**Corporate-grade from-scratch LM pretraining**: random-init GPTNeoX 1B parameters, multi-org federation, full Pile-style tokenized corpus. The most demanding job in the catalog — tests federated aggregation under genuine pretraining load.

## Model

- **Architecture**: GPTNeoX 1B (Pythia spec)
- **Parameters**: 1,011,781,632 (~1.01 B)
- **Init**: random (no pretrained weights)
- **Library**: HuggingFace `transformers` (GPTNeoXForCausalLM)
- **Reference checkpoint**: [`EleutherAI/pythia-1b`](https://huggingface.co/EleutherAI/pythia-1b) — what we're trying to match via FL

## Reference recipe (verbatim from EleutherAI Pythia)

Source: https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1B.yml

| Param | Value |
|---|---|
| Optimizer | **Adam** (β1=0.9, β2=0.95, ε=1e-8) |
| Weight decay | **0.1** |
| Learning rate | **6e-4** (peak), cosine to **6e-5** |
| Warmup | **1%** of total steps |
| Gradient clip | **1.0** |
| Batch size | **2M tokens** (= 1024 sequences × 2048 tokens) |
| Sequence length | **2048** |
| Total steps | **143,000** (~300B tokens, 1 epoch over Pile) |
| Mixed precision | **bf16** (or fp16) |
| Gradient checkpointing | yes |
| Model arch | hidden=2048, layers=16, heads=8, intermediate=8192, rotary_pct=0.25 |
| Pretraining hardware | **32× A100-40GB × ~1 day** for 1B variant |
| Pretraining data | The Pile (uncopyrighted variant: `monology/pile-uncopyrighted`) |
| Final published metrics | LAMBADA, PIQA, WinoGrande, ARC-easy, SciQ — see [Pythia results](https://github.com/EleutherAI/pythia/tree/main/results) |

## Smaller variants (set `CONFIG_NAME` in `model_def.py`)

If full 1B is too heavy, drop to a smaller Pythia variant — same recipe, less compute:

| Variant | Params | Centralized GPU-hours | Recommended for |
|---|---|---|---|
| `pythia-70m` | 70 M | ~50 | quick FL smoke test, dev |
| `pythia-160m` | 160 M | ~190 | reasonable FL benchmark |
| `pythia-410m` | 410 M | ~600 | substantial FL benchmark |
| **`pythia-1b`** | **1.01 B** | **~1,300 (32× A100 × 1 day)** | corporate-grade demo |

## Dataset

- **Default**: `monology/pile-uncopyrighted` (Pile minus copyrighted books, ~300 GB raw text, public)
  - https://huggingface.co/datasets/monology/pile-uncopyrighted
- **Format**: `data.jsonl` with `{"text": "..."}` rows; tokenized on-the-fly per worker
- **Per-shard at full scale**: ~75 GB raw text → ~75 B tokens

For development, `MAX_EXAMPLES = 4000` in `build_shards_hf.py` builds tiny test shards (~10 MB each).

## FL config (matches Pythia-1B recipe across 4 workers)

| FL knob | Value | Maps to centralized recipe |
|---|---|---|
| `learning_rate` | **6e-4** | Pythia peak LR |
| `batch_size` | **4** per worker (effective 16 across 4 workers; with grad_accum=4 → effective 64; far below Pythia's 1024 — bump grad_accum if you have budget) | Pythia uses 1024 seqs × 2048 tokens = 2M tokens/step |
| `local_epochs` | **1** (1 epoch over local shard ≈ 1/4 of full corpus) | Pythia trains 1 epoch on Pile |
| `num_rounds` (server) | 5 | for FL aggregation cadence |

To actually match Pythia's 2M tokens/step, increase `gradient_accumulation_steps` in `pythia_utils.py` to ~256. With 4 workers each running effective batch 256 × 2048 = 524k tokens/step, sum = 2M tokens/step across the federation.

## Compute estimate (4 workers × 1× H100-80GB each, in parallel)

| Phase | Per worker | Wall time |
|---|---|---|
| Per-step time (batch 4 × seq 2048, fp16, grad_accum 4) | ~7 s | — |
| Steps per worker (143k / 4) | ~36k | ~70 hr |
| **5-round wall time + aggregation** | — | **~3.5 days** |

Single-worker centralized (1 H100, full 143k steps):
- 143,000 × 7s ≈ **278 hours ≈ 12 days**

**FL speedup: ~3.4×** (4 workers in parallel + aggregation overhead).

## Hardware

- **Min GPU per worker**: H100-80GB or A100-80GB (Pythia-1B + bf16 + grad_accum)
- **Smaller variants** (`pythia-70m`/`160m`): A10G / RTX 4090 (24 GB) suffices

## Quick start

```bash
cd /home/pyaremenko/examples/job_lm_scratch
pip install datasets transformers accelerate
python build_shards_hf.py        # smoke-test default (4k rows); bump MAX_EXAMPLES for real
```

## Federation value proposition

Multiple cloud providers / model marketplaces / sovereign-AI consortia want to jointly pretrain a foundation model from scratch but cannot pool training corpora due to:
- Per-jurisdiction copyright/licensing constraints
- Customer-data residency requirements
- Competitive non-disclosure of training mixes

FL gives the consortium:
- A jointly-owned 1B foundation model with quality close to centralized
- ~3.4× faster wall time than any one consortium member alone
- No raw text leaves any participant's storage; only aggregated weights move

This is the textbook FL-pretraining scenario.

## Caveats

- **Pretraining is 1000× more sensitive to FL aggregation** than fine-tuning. Non-IID shards or bad aggregation cadence will diverge fast. Test with `pythia-70m` first.
- **Tokenizer must be consistent across workers**. We use `EleutherAI/pythia-1b`'s tokenizer everywhere; if you swap, all shards must use the same one.
- **Large state-dict transfer**: 1B fp32 = ~4 GB per round per direction. With 5 rounds × 4 workers, total network ≈ 80 GB. Plan inter-worker bandwidth accordingly (>1 Gbps).
- **Embedding dimension is huge**: vocab × hidden (50304 × 2048) is ~200 MB by itself. Embeddings dominate the state-dict for small Pythia variants.
