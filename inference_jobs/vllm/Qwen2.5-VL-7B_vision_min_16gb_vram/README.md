# Qwen2.5-VL-7B (fp16) — document vision (vLLM)

The **document-understanding workhorse** (`vision-qwen25-vl-7b`, priority
v1, Apache-2.0): invoices, scans, receipts, screenshots, dashboards →
structured JSON. A 24 GB worker processes tens of thousands of pages/day
flat-rate vs Azure Doc Intelligence ≈ $1.5/1k pages.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB | 4 | 48 GB | ~40 GB (image 15 + 1.3×16.6 GB checkpoint) | 24gb / 2x12gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured);
16 GB cards (4080/5080) fit only with reduced context + `limit_mm_per_prompt=1`
— tight, prefer 24 GB; 2× 12 GB → PP=2 (copy the PP sibling's shape).

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                                  # sample invoice → JSON
python predict.py ./scan.png "List all line items"
python predict.py https://example.com/receipt.jpg
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

`sample_data/invoice.png` is a synthetic invoice generated for smoke
tests — the default `predict.py` run extracts it to JSON.

## How images travel

Standard OpenAI vision content parts on `/v1/chat/completions`:

```python
{"role": "user", "content": [
    {"type": "text", "text": "Extract vendor, total… JSON only."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}},
]}
```

Any OpenAI vision client works unmodified. `predict.py` shows both the
data-URL (local file) and plain-URL paths.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `limit_mm_per_prompt` | images per request (4 shipped) — vision tokens eat KV fast |
| `max_model_len` | 32k shipped; a page ≈ 1–2.5k vision tokens |
| the extraction prompt (client-side) | your schema — that's the paid setup work |
| `model_source` | quality tier → the 32B AWQ sibling (TP=2, 2×24 GB) |

## Notes

- fp16 on purpose: 4-bit visibly hurts small-text OCR at 7B. The 32B
  sibling takes the AWQ trade instead (bigger brain compensates).
- Prefix caching pays off here too — the extraction prompt is identical
  across requests; only image tokens differ.
