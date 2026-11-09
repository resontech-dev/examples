# MamayLM 27B v2.0 (official FP8) — chat + vision, TP=2 on 2×24 GB (vLLM)

The **Ukrainian quality tier**: Gemma 3 27B adapted to Ukrainian by
INSAIT (June 2026), text and images. INSAIT ships the FP8 build
themselves (~30 GB, vision kept in bf16), so no home-made quantization is
needed — but it does not fit 16 GB cards: 2×24 GB (4090), 2×32 GB
(5090 / RTX PRO 4500) or one 48 GB card.

Shared vLLM contract & defaults: [../../README.md](../../README.md).
Folder index and Gemma 3 engine notes: [../README.md](../README.md).

## Resources

| VRAM min→rec (total) | CPU | RAM | Min free disk | Shape |
|---|---|---|---|---|
| 40 → 48 GB | 8 | 64 GB | ~54 GB (image 15 + 1.3×30 GB checkpoint) | 2×24 GB TP=2 (as configured) · 1×48 GB TP=1 |

FP8 weights need compute capability ≥ 8.9 (Ada / Blackwell). Both GPUs
in **one** worker container.

Expected speed (estimate from the memory-bandwidth rule, measure before
quoting): ~60–70 generated tok/s per answer on 2×4090, ~30 on 2×RTX PRO
4500; 6–8 concurrent users under 5 s to first token with 4k-token prompts.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                   # Ukrainian business reply
python predict.py --image                           # sample_data/invoice.png → JSON
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where | for what |
|---|---|
| `tensor_parallel_size` (yaml + py) and `gpus_per_worker` | TP=1 on a single 48 GB card |
| `model_source` | `INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0` (bf16, ~54 GB → 2×32 GB or 80 GB); your own W4A16 build for 32 GB |
| `max_model_len`, `max_num_seqs`, `limit_mm_per_prompt` | context, concurrency, pages per request |

Never set `quantization="fp8"` on this (or any Gemma 3) checkpoint. License: Gemma Terms of Use.
