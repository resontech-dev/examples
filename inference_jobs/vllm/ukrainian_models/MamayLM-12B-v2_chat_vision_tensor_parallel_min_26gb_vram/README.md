# MamayLM 12B v2.0 (official bf16) — chat + vision, TP=2 on 2×16 GB (vLLM)

The documented INSAIT checkpoint, June 2026: Gemma 3 12B adapted to
Ukrainian, **text and images**, Ukrainian/English. bf16 is ~24 GB, so two
16 GB cards in one machine shard it; a single 24 GB card runs it at TP=1.
The official 12B is the honest quality reference for the one-card int4
jobs in this folder.

Shared vLLM contract & defaults: [../../README.md](../../README.md).
Folder index and Gemma 3 engine notes: [../README.md](../README.md).

## Resources

| VRAM min→rec (total) | CPU | RAM | Min free disk | Shape |
|---|---|---|---|---|
| 26 → 32 GB | 8 | 32 GB | ~47 GB (image 15 + 1.3×24 GB checkpoint) | 2×16 GB TP=2 (as configured) · 1×24 GB TP=1 · 2×24 GB TP=2 for 32k context |

Both GPUs must be visible to **one** worker container (start the worker
with both cards, not one container per card). TP over Ethernet does not
work.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                   # Ukrainian business reply
python predict.py --image                           # sample_data/invoice.png → JSON
python predict.py --image ./skan.jpg "Перепиши текст зі скану дослівно"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

Images travel as standard OpenAI content parts; Gemma 3 normalizes each
image to 896×896 = 256 tokens, so pages are cheap — resize photos to
~1 000 px on the long side client-side.

## What to change

| where | for what |
|---|---|
| `tensor_parallel_size` (yaml + py) and `gpus_per_worker` | TP=1 on a single 24 GB card |
| `model_source` | `lapa-llm/lapa-v0.1.3-instruct` (Lapa 12B official), `…-12B-IT-v1.0` |
| `max_model_len`, `max_num_seqs`, `limit_mm_per_prompt` | 32k / 8 / 4 on 2×24 GB |

License: Gemma Terms of Use.
