# ⚠️ TEMPLATE: Qwen-Image-2512 + Qwen-Image-Edit-2511, text-to-image and image editing (predictor mode)

**Written 2026-11-19, not booted. Needs an 80 GB card.** Two 20B MMDiT
models from the Qwen team, both **Apache-2.0**, behind one endpoint:

- **prompt → image**: Qwen-Image-2512 makes posters, product shots and
  people, and renders text in the image in Chinese and English;
- **image(s) + instruction → edited image**: Qwen-Image-Edit-2511 takes
  "make the header dark navy", "remove the person on the left" or "the bear
  from image 1 shakes hands with the bear in image 2". Up to 3 reference
  images per request.

The request and response are the same as the
[FLUX.2 klein job](../FLUX.2-klein-4B_imagegen_edit_min_16gb_vram/README.md),
plus `true_cfg_scale` and `negative_prompt`. A client moves from klein to
this job by changing the endpoint, not the code. On the Artificial
Analysis open-weights editing board, Edit-2511 scores 996 against 941 for
klein 4B (FLUX.2 [dev] = 1000, November 2026). Both are full diffusion
models with CFG, not distilled ones, so each image takes tens of seconds
instead of klein's few.

The older text-to-image template [`../Qwen-Image-20B_imagegen_template_min_40gb_vram`](../Qwen-Image-20B_imagegen_template_min_40gb_vram/README.md)
runs the August 2025 `Qwen/Qwen-Image` weights. This job replaces it.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 80 → 141 GB | 8 | 128 GB | ~140 GB (image 8 + 1.3 × ~99 GB) | frontier (80gb+) |

| Part | Size (bf16) | Source |
|---|---|---|
| Qwen-Image-2512 transformer | 40.9 GB | `Qwen/Qwen-Image-2512` |
| Qwen-Image-Edit-2511 transformer | 40.9 GB | `Qwen/Qwen-Image-Edit-2511` |
| Qwen2.5-VL-7B text encoder | 16.6 GB | shared: the same files in both repos |
| VAE | 0.25 GB | shared |

| Card | Mode | Notes |
|---|---|---|
| H200 141 GB, B200 | resident | both models on the card (~92 GiB of weights + activations) |
| H100 / A100 80 GB | model CPU offload | one component on the GPU at a time; each image also moves ~58 GB to the GPU over PCIe, several seconds |
| 48 GB (L40S, RTX 6000 Ada) | model CPU offload | peak is one 38 GiB transformer plus activations: may fit at 1 MP, untested |

There are no measured timings yet. Text-to-image runs 50 steps with CFG
(100 transformer passes at 1328×1328); edits run 40.

## How it is wired

```
predict(data)
 ├─ no image  ─► QwenImagePipeline         (Qwen-Image-2512 transformer)  ─┐
 └─ 1–3 images ─► QwenImageEditPlusPipeline (Qwen-Image-Edit-2511 transformer) ─┤
                                     shared: Qwen2.5-VL-7B text encoder, VAE ◄─┘
```

- Both Hub repos ship the same text encoder, tokenizer and VAE files (the
  same sha256 hashes). The edit pipeline is loaded with
  `text_encoder=` and `vae=` from the text-to-image one, so diffusers
  neither downloads nor loads them twice.
- Under offload, each pipeline's hooks chain its own components (text
  encoder → transformer → VAE), and the shared modules can carry only one
  pipeline's hooks. When the mode changes, the job re-runs
  `enable_model_cpu_offload()` on the pipeline about to run. That call
  also moves its components back to the CPU.
- `QwenImageEditPlusPipeline` cannot generate without an input image (it
  reads `image[-1].size` first), so text-to-image needs the 2512 model.

## Folder layout

```
.
├── inference.yaml                 # predictor contract
├── scripts/
│   └── qwen_image_edit_serve.py   # QwenImageGenEdit.predict(data) → images_b64
├── submit.py
├── predict.py                     # prompt [+ images] → out/qwen_<mode>_<seed>.png
└── .env.example
```

## Deploy & test (once an 80 GB host exists)

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py            # first boot pulls ~99 GB from HF Hub
python predict.py 'poster: "GRAND OPENING Saturday 10:00", art-deco style, gold on black'
python predict.py --image screen.png "make the header dark navy and the main button orange, keep everything else"
python predict.py --image bear1.png --image bear2.png "the bear from image 1 and the bear from image 2 shake hands in a park"
```

Or paste `scripts/qwen_image_edit_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

JSON body, sent as `application/octet-stream` (`client.predict_json`):

| field | default | notes |
|---|---|---|
| `prompt` | required | for edits: the instruction ("change X to Y, keep everything else") |
| `images_b64` | none | list of base64 PNG/JPEG/WebP, up to 3; `image_b64` for one; `data:` URLs accepted. Any image switches to edit mode |
| `width` / `height` | 1328×1328, or the first image's aspect at ~1 MP | snapped to 16 px, capped at 1328×1328 of area; the 2512 card's presets (1664×928, 1472×1104, 1584×1056 and their portrait versions) all fit |
| `num_images` | 1 | up to 2, seeds `seed`, `seed+1` |
| `steps` | 50 (text-to-image) / 40 (edit) | at most 80 |
| `true_cfg_scale` | 4.0 | 1 turns CFG off: half the passes, weaker prompt following |
| `negative_prompt` | the 2512 card's quality list (text-to-image) / `" "` (edit) | |
| `seed` | random | always returned |
| `format` | `png` | `jpeg` / `webp` for smaller responses |

Response: `{"images_b64": [...], "image_b64": "<first>", "mode": "t2i" | "edit", "width", "height", "steps", "seeds": [...], "seed", "format", "model_id", "elapsed_ms"}`.

- **300 s per request.** One request to a predictor job is cut off after
  300 s (set on the Ray head node; the predict wrapper answers 504 at
  290 s), and the wait in the queue counts too. `max_images: 2` and
  `max_ongoing_requests: 2` (one running, one waiting) keep a request
  inside it. Send one request at a time and raise neither before
  measuring.
- The request cap is 32 MB. Downscale large screenshots on the client.
- The edit pipeline resizes references itself: about 384×384 for the
  vision encoder and about 1 MP for the VAE.

## What to change

| where | for what |
|---|---|
| `init_args.edit_model_id: null` | text-to-image only: ~58 GB of weights, resident on an 80 GB card |
| `init_args.t2i_model_id: null` | edits only, same size |
| `init_args.offload` | `"auto"`; `"on"` forces offload; `"off"` resident |
| `init_args.t2i_steps` / `edit_steps` | fewer steps to cut latency, at some quality cost |
| `init_args.max_images` | raise only after timing one image on the target card |

## Variants

| Variant | Edits | Status |
|---|---|---|
| Lightning LoRAs (4–8 steps, no CFG) | lightx2v publishes them for the Qwen-Image family; check one exists for 2512 / Edit-2511, then `pipe.load_lora_weights(...)` and `true_cfg_scale: 1` | about 10× faster per image; not wired here |
| FP8 transformers | `transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)` | halves each transformer to ~20 GB, so 48 GB cards fit resident with one model; untested here |
| Qwen-Image-2512 alone on 80 GB | `edit_model_id: null` | resident, no PCIe traffic per request |

Models left out, all checked in November 2026:

| Model | Why not |
|---|---|
| MiniMax H3-Image | best open model on both boards, but its community license is reported to exclude the US, the EU, the UK and South Korea (read the text before relying on it) |
| Qwen-Image-2.1 | Qwen Research License, non-commercial |
| FLUX 3 Image Klein | FLUX Kommunity License: only businesses under $5M revenue, and no right to host it as an API for others |
| Bria Fibo 2.0 | weights CC BY-NC |
| HiDream-O1-Image | MIT, but not in a diffusers release: needs the vendor's git code, exactly `transformers==4.57.1` and flash-attn |
| HunyuanImage 3.0 Instruct | license excludes the EU, the UK and South Korea; 8 × 80 GB recommended |

## Before listing

1. Boot on an 80 GB card (offload) and on an H200 (resident). Record
   peak VRAM and RAM, and the time for a 1328×1328 text-to-image image
   and for an edit with one and with three references.
2. Under offload, measure what the ~58 GB moved to the GPU per image
   costs, and check that alternating text-to-image and edit requests keep
   working (the re-hook on a mode switch).
3. Check that `diffusers>=0.37,<0.40` + `transformers<5` resolve with the
   platform's `numpy<2` for predictor jobs.
4. Try 10 real screenshots for edits and 10 poster or text prompts for
   text-to-image.
5. Drop the TEMPLATE and "not booted" markers.

## License

Weights and code are Apache-2.0 (Qwen team, Alibaba). Qwen-Image-2.1 is
not: its Qwen Research License is non-commercial, so never put it in
`t2i_model_id` for a customer deploy.
