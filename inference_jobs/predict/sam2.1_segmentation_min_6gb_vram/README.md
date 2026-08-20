# SAM 2.1 — promptable segmentation (predictor mode)

Point/box-prompted image segmentation with **SAM 2.1 hiera-large**
(`cv-sam2`, Apache-2.0): click a point, get pixel-perfect masks. Strong
demo material, and the base for interactive labeling / QC tooling. Runs
the standard `predict(data: bytes) -> dict` contract — no vLLM here.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 6 → 12 GB | 4 | 24 GB | ~13 GB (predictor image 8 + 1.3×3.5 GB weights) | 12gb |

Suggested GPUs: any 12 GB card (RTX 5070 / 4070 / 3060-12G); it also
*shares* a worker at these sizes. `sam2.1-hiera-small` (~1 GB) halves the
footprint for shared deployments.

## Folder layout

```
.
├── inference.yaml          # predictor contract: module/class + cluster
├── scripts/
│   └── sam2_serve.py       # Sam2Segmenter.predict(data) → masks
├── sample_data/
│   └── shapes.png          # synthetic test image (circle/rect/triangle)
├── submit.py               # deploy via SDK
├── predict.py              # send image + point, save masks to ./out/
└── .env.example
```

## Deploy & test

```bash
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py
python predict.py                       # segments the circle in shapes.png
python predict.py ./photo.jpg 420 310   # your image, your point
```

Or paste `scripts/sam2_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

JSON via `predict_json` (⚠️ octet-stream, NOT application/json — the
server's JSON envelope drops prompt fields): `image_b64`, `points`,
`labels` (1=fg / 0=bg), optional `boxes`, `multimask`. Raw image bytes
also work (defaults to a center-point prompt). Response: top-K masks as
1-bit PNGs (base64) with score / area / bbox.

## ⚠️ Verify before selling

Uses the **transformers Sam2 integration** (`transformers>=4.56`,
checkpoints `facebook/sam2.1-hiera-*`). Smoke-test on the pinned
predictor image; the official `sam2` package is the fallback backend.

## Video

SAM 2.1's headline feature — video object tracking — needs the
`Sam2VideoModel` streaming API and a frame-pipeline predictor; that's a
follow-up template. This job is image-only.
