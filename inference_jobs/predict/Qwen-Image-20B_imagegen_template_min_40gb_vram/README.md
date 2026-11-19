# ⚠️ TEMPLATE — Qwen-Image (20B MMDiT) image generation

**Not deployable on the current consumer fleet.** The scale-up image
generator (Apache-2.0): a 20B MMDiT with the **best
open text-rendering in images** — posters, UI mockups, signage with real
typography. Ships now so testing starts the day a frontier host (40–80 GB)
joins the fleet; everything inside is a planning estimate.

**Superseded:** this runs the August 2025 `Qwen/Qwen-Image` weights.
[`../Qwen-Image-2512-Edit-2511_imagegen_edit_template_min_80gb_vram`](../Qwen-Image-2512-Edit-2511_imagegen_edit_template_min_80gb_vram/README.md)
serves the December 2025 Qwen-Image-2512 plus the Edit-2511 editor on one
endpoint, with the FLUX.2 klein job's request shape.

## Resources (catalog planning numbers — unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 40 → 80 GB | 8 | 160 GB | ~60 GB (predictor image 8 + 1.3×41 GB pipeline) | frontier (40–80gb) |

Suggested GPUs: A100-80G / H100 (fully resident, fastest); A100-40G /
L40S 48 GB with CPU offload (`offload: "auto"` handles it, ~2× latency).
Consumer 24 GB cards: no — use the FLUX.1-schnell sibling instead.

## Before first deploy (checklist)

1. Confirm a frontier worker (≥ 40 GB free VRAM, ≥ 60 GB free disk).
2. Pre-mirror the ~41 GB pipeline into Garage/S3 if possible.
3. Validate output + latency; tune `default_steps` / `true_cfg_scale`.
4. Drop the TEMPLATE banners.

## Deploy & test (once hardware exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py 'poster: "GRAND OPENING — Saturday 10:00", art-deco style'
```

Or paste `scripts/qwen_image_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

Identical contract to the FLUX sibling (`prompt`, `width`, `height`,
`steps`, `seed`, plus `negative_prompt` / `true_cfg_scale` — this is a
full diffusion model, not distilled): clients swap endpoints, not code.

**300 s per request.** One request to a predictor job is cut off after 300 s (set on the Ray head node; the predict wrapper answers 504 at 290 s). The
wait in the queue counts too: `max_ongoing_requests: 1` keeps the job
strictly serial, so a request queued behind others waits for them, and
with several in flight the last ones can get 504. Send one request at a
time.
