# FLUX.2 [klein] 4B: text-to-image and image editing (predictor mode)

**Written 2026-11-19, not booted.** Black Forest Labs' smallest FLUX.2,
**Apache-2.0**, step-distilled to 4 steps. One model does both jobs:

- **prompt → image**: posters, product shots, illustrations, readable text in the image;
- **image(s) + instruction → edited image**: send a screenshot or a photo
  with "make the header dark navy", "remove the person on the left",
  "put the product from image 1 on the table in image 2". Up to 4
  reference images per request.

Seconds per image, so the call is synchronous: no job ids, no polling.
The edit follows the instruction you write; the model does not decide on
its own what to change. To have an LLM look at a screenshot and write the
instruction, put a vision chat job in front of it (see Variants).

For video from the result, send the image to
[`../Wan2.2-TI2V-5B_videogen_min_16gb_vram`](../Wan2.2-TI2V-5B_videogen_min_16gb_vram/README.md)
as `image_b64`.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB | 8 | 24 GB (32 comfortable) | ~35 GB (image 8 + 1.3 × ~16 GB pipeline) | 16gb / 24gb |

The pipeline is ~16 GB in bf16: the 4B transformer (~8 GB), a Qwen3 text
encoder (~8 GB) and the VAE. Below 22 GB of VRAM the job turns on model CPU
offload: the text encoder and the transformer take turns on the GPU, the
other waits in RAM. BFL quotes "as little as 13 GB VRAM".

| Card | Mode | 1024×1024, 4 steps (estimate) |
|---|---|---|
| RTX 5070 Ti / 5080 / 4080, 16 GB | model offload | ~3–8 s (most of it is moving weights over PCIe) |
| RTX 4090 / 3090, 24 GB | resident | ~1–3 s |

The times are estimates, not measurements. The first request after boot is
slower (CUDA warm-up).

## Folder layout

```
.
├── inference.yaml          # predictor contract
├── scripts/
│   └── klein_serve.py      # KleinImageGen.predict(data) → images_b64
├── submit.py
├── predict.py              # prompt [+ images] → out/klein_<seed>.png
└── .env.example
```

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py            # first boot pulls ~16 GB from HF Hub
python predict.py "a ceramic coffee cup on a wooden desk, morning light, product photo"
python predict.py --n 4 --size 1344x768 "poster: 'RESON STUDIO' in bold neon letters, night city"
python predict.py --image screen.png "make the header dark navy and the main button orange"
python predict.py --image person.jpg --image room.jpg "put the person from image 1 on the sofa in image 2"
```

Or paste `scripts/klein_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

JSON body, sent as `application/octet-stream` (`client.predict_json`):

| field | default | notes |
|---|---|---|
| `prompt` | required | for edits: the instruction ("change X to Y, keep everything else") |
| `images_b64` | none | list of base64 PNG/JPEG/WebP, up to 4; `image_b64` for one; `data:` URLs accepted |
| `width` / `height` | 1024×1024, or the first image's aspect at ~1 MP | snapped to 16 px, capped at 1536×1536 of area |
| `num_images` | 1 | up to 4, seeds `seed`, `seed+1`, … |
| `steps` | 4 | at most 8 |
| `seed` | random | always returned |
| `format` | `png` | `jpeg` / `webp` for smaller responses |

Response: `{"images_b64": [...], "image_b64": "<first>", "mode": "t2i" | "edit", "width", "height", "steps", "seeds": [...], "seed", "format", "elapsed_ms"}`.

- Reference images are downscaled to ~1 MP before encoding; a 1920×1080
  screenshot comes back at 1360×768 unless you send a size.
- The request cap is 32 MB: four large screenshots in base64 can pass it.
  Downscale on the client.
- One request to a predictor job is cut off after 300 s (set on the Ray
  head node; the predict wrapper answers 504 at 290 s), far above the
  seconds klein needs per image.
- Small UI text in screenshots may come back blurred or rewritten. Name the
  exact text you want in the prompt, in quotes.

## What to change

| where | for what |
|---|---|
| `init_args.offload` | `"auto"`; `"on"` forces offload on 24 GB if it runs out of memory; `"off"` resident |
| `init_args.max_area` | lower it to cap latency |
| `init_args.max_refs` | each reference adds tokens and memory |

## Variants

| Variant | Edits | Status |
|---|---|---|
| FLUX.2-klein-base-4B (undistilled, Apache-2.0) | `model_id: "black-forest-labs/FLUX.2-klein-base-4B"`, more steps, real CFG | for fine-tuning and LoRA; slower |
| FLUX.2-klein-9B | `model_id: "black-forest-labs/FLUX.2-klein-9B"` | better quality; gated, **non-commercial FLUX license** (check the card); about twice the weights, 24 GB+ |
| Screenshot → "what to change" → edit | a vision chat job (e.g. `../../vllm/Gemma-3-12B-W4A16_chat_vision_min_14gb_vram`) writes the instruction, this job applies it | two jobs; on one 16 GB card they cannot run at the same time |

## Before listing

1. Boot on a 16 GB card; record peak VRAM and the time for 1024×1024
   text-to-image and for an edit with one and with four references.
2. Check `diffusers>=0.37,<0.40` + `transformers<5` resolve with the
   platform's `numpy<2` for predictor jobs.
3. Try 10 real screenshots: which edits hold (colors, layout, text) and which do not.
4. Drop the "not booted" marker.

## License

Weights Apache-2.0 (Black Forest Labs). The 9B klein and FLUX.2-dev are
non-commercial: never swap them into `model_id` for a customer deploy.
