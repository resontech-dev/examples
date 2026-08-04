# FLUX.1-schnell — image generation (predictor mode)

Best-quality **Apache-clean** image generation (`img-flux-schnell`):
product shots, marketing visuals, game assets at ~$0.0002 compute/image
vs ~$0.04/image API pricing. 4-step distilled — ~1–2 s/image fully
resident, ~2× that with CPU offload.

## ⚠️ The license trap (catalog rule)

**FLUX.1-schnell is Apache-2.0. FLUX.1-dev is NON-COMMERCIAL.** Never
swap `-dev` into `model_id` for a customer deploy — it looks like a free
quality upgrade and it's a license violation.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB | 4 | 48 GB (offload streams ~24 GB of weights through system RAM) | ~40 GB (predictor image 8 + 1.3×24 GB pipeline) | 24gb |

Suggested GPUs: RTX 3090 / 4090 / L4 (24 GB — CPU offload auto-enables,
~2× latency); RTX 4080/5080 (16 GB — offload, floor); A100-40G+ / L40S
run fully resident (`offload: "off"` territory, fastest).

## Folder layout

```
.
├── inference.yaml          # predictor contract + tight queue
├── scripts/
│   └── flux_serve.py       # FluxImageGen.predict(data) → PNG b64
├── submit.py
├── predict.py              # prompt → out/flux_<ts>.png
└── .env.example
```

## Deploy & test

```bash
cp .env.example .env
python submit.py            # first boot pulls ~24 GB of pipeline weights
python predict.py "product shot of a ceramic mug on slate, softbox lighting"
python predict.py --size 1280x768 --seed 7 "isometric game asset, watermill"
```

Or paste `scripts/flux_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

JSON via `predict_json`: `prompt` (required), `width`/`height`
(snapped to the 16-px grid, ≤ 1536), `steps` (4 default — schnell is
4-step distilled, more wastes compute), `seed`. Response: `image_b64`
PNG + timings. `guidance_scale` is pinned to 0 — that's how distilled
FLUX works, not a missing feature.

## What to change

| where | for what |
|---|---|
| `init_args.offload` | `"auto"` shipped; `"off"` on 40 GB+ cards for full speed |
| `init_args.max_side` | canvas ceiling (VRAM/latency guard) |
| style LoRAs | the paid customization — diffusers `load_lora_weights` in `__init__` is the extension point |
