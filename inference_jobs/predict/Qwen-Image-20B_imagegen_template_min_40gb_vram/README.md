# ⚠️ TEMPLATE — Qwen-Image (20B MMDiT) image generation

**Not deployable on the current consumer fleet.** The scale-up image
generator (`img-qwen-image`, Apache-2.0): a 20B MMDiT with the **best
open text-rendering in images** — posters, UI mockups, signage with real
typography. Ships now so testing starts the day a frontier host (40–80 GB)
joins the fleet; everything inside is a planning estimate.

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
