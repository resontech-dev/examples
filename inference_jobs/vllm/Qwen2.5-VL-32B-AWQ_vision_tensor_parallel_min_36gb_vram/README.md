# Qwen2.5-VL-32B (AWQ) — document vision quality tier, TP=2 (vLLM)

The quality tier for **hard documents** (`vision-qwen25-vl-32b`,
Apache-2.0): degraded scans, handwriting, dense tables — the cases where
the 7B sibling starts guessing. AWQ ~19 GB sharded **TP=2 across 2×24 GB
GPUs in one machine**. Same OpenAI vision protocol and same `vision`
model alias as the 7B — clients upgrade by pointing at this endpoint.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 36 → 48 GB | 8 | 96 GB | ~46 GB (image 15 + 1.3×19 GB checkpoint) | 2x24gb |

Suggested GPUs: one box with 2× RTX 3090 / 4090 (as configured,
`STRICT_PACK`); a single L40S / A40 (48 GB) runs it at TP=1 — simpler
and lower latency; A100-80G fits the fp16 checkpoint outright.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                  # sample invoice → JSON
python predict.py ./hard_scan.png "Transcribe the handwritten notes"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | VERIFY the official AWQ repo id on HF before first deploy |
| `tensor_parallel_size` | 2 shipped; TP=1 on a single 48 GB card (update yaml too!) |
| `limit_mm_per_prompt` | 6 shipped — the pooled KV affords more pages per call |

## Notes

- AWQ is the right trade at 32B: the bigger brain compensates the 4-bit
  loss that hurts the 7B on small text.
- TP=2 also cuts per-token latency (each token's math splits across both
  cards over PCIe) — this tier is *faster* per token than the 7B fp16 on
  one card, not just smarter.
