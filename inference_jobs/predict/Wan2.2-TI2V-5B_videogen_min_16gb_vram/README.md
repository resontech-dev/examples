# Wan2.2-TI2V-5B: text-to-video and image-to-video (predictor mode)

**Written 2026-11-19, not booted.** One 5B model from the Wan team,
**Apache-2.0**, that turns a prompt, or a prompt plus a still, into a
1280×704 clip at 24 fps, up to 121 frames (5 s). No sound: for video with
audio see [`../LTX-2.3-22B_videogen_audio_min_16gb_vram`](../LTX-2.3-22B_videogen_audio_min_16gb_vram/README.md).
To have a chat model plan the shots and call this generator, see
[`../../vllm/Qwen3-8B-AWQ_media_studio_min_32gb_vram`](../../vllm/Qwen3-8B-AWQ_media_studio_min_32gb_vram/README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB (40 GB keeps it all resident) | 8 | 48 GB (64 comfortable) | ~55 GB (image 8 + 1.3 × 34.2 GB repo) | 16gb / 24gb |

The repo is 34.2 GB: the transformer is stored in fp32 (20 GB, 10 GB once
loaded in bf16), the UMT5 text encoder is 11.4 GB, the VAE 2.8 GB. Below
40 GB the job runs with model CPU offload: one component on the GPU at a
time, the rest in RAM. VAE tiling is always on; without it the decode of a
720p clip spikes to ~26 GB.

| Card | Mode | 121 frames, 50 steps (estimate) |
|---|---|---|
| RTX 5070 Ti / 5080 / 4080, 16 GB | model offload (`offload: "group"` if it OOMs) | ~12–18 min |
| RTX 4090 / 3090, 24 GB | model offload | ~9 min (official 4090 figure: 535 s with offload) |
| RTX 5090, 32 GB | model offload | ~5–7 min |
| L40S / A6000 48 GB, H100 | resident | ~3–4 min on H100 |

The times are estimates from the official numbers, not measurements.
The draft preset `--frames 49 --steps 30` (2 s) cuts them by roughly 4×.

## Folder layout

```
.
├── inference.yaml          # predictor contract, ONE replica (job ids live in its memory)
├── scripts/
│   └── wan_serve.py        # WanVideoGen.predict(data) → MP4 b64, or a job id to poll
├── submit.py
├── predict.py              # prompt [+ image] → out/wan_<id>.mp4, polls until done
└── .env.example
```

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py            # first boot pulls 34.2 GB from HF Hub
python predict.py "a red fox trots through fresh snow, low tracking shot, golden hour"
python predict.py --image photo.jpg "the camera slowly pushes in, leaves drift in the wind"
python predict.py --frames 49 --steps 30 "draft: waves crash on black rocks"
```

Or paste `scripts/wan_serve.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Request / response

A clip takes minutes and one request to a predictor job is cut off after
300 s (set on the Ray head node; the predict wrapper answers 504 at
290 s), so the clip renders in a background thread and each call returns
within `wait_s` (default 20, so a start returns the id fast; at most 280):

| Call | Body | Returns |
|---|---|---|
| start | `{"prompt", "image_b64"?, "width"?, "height"?, "num_frames"?, "steps"?, "guidance_scale"?, "seed"?, "negative_prompt"?, "wait_s"?}` | the finished clip, or `{"id", "status": "queued" \| "in_progress", ...}` |
| poll | `{"action": "status", "id": "...", "wait_s"?}` | same |

Finished: `{"status": "completed", "video_b64": "<MP4>", "mode": "t2v" | "i2v", "width", "height", "num_frames", "fps", "seconds", "steps", "seed", "elapsed_ms"}`.
Failed: `{"status": "failed", "error": "..."}`.

- **Sizes** snap to multiples of 32 and are scaled down to 1280×704 of area.
  `1280x720` becomes `1280x704`. Vertical `704x1280` works.
- **Frames** snap to 4k+1, at most 121.
- **Image-to-video** keeps the image's aspect ratio: the job resizes and
  center-crops it to the requested area. The image becomes the first frame.
- **Negative prompt** defaults to the official Chinese one the model was tuned with.
- **Seed** is random when omitted and always returned, so a good take can
  be re-rendered with more steps.
- Finished clips stay fetchable for an hour (`keep_results_s`); ids do not
  survive a restart. At most 4 clips wait behind the running one.

## What to change

| where | for what |
|---|---|
| `init_args.offload` | `"auto"`; `"off"` on 40 GB+; `"group"` if 16 GB runs out of memory |
| `init_args.default_steps` | 50 official; 30 drafts |
| `init_args.cache_threshold` | 0.05–0.2 turns on diffusers' first-block cache: fewer transformer passes, softer detail |
| `init_args.max_area` | lower it to cap latency (e.g. 832×480 = 399360) |

## Variants

| Variant | Edits | Status |
|---|---|---|
| FastWan2.2-TI2V-5B (FastVideo DMD distill, Apache-2.0) | `model_id: "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"`, `default_steps: 4`, `guidance_scale: 1.0` | loads with the explicit pipeline classes used here; the exact DMD timesteps live only in FastVideo, quality in plain diffusers unverified |
| Wan2.2-T2V-A14B / I2V-A14B (MoE, two 14B experts) | different pipelines (`boundary_ratio`, `guidance_scale_2`), 81 frames at 16 fps, 126 GB repos | needs 48 GB+ with offload; Lightning 4-step LoRAs (`lightx2v/Wan2.2-Lightning`, Apache-2.0) make it fast |
| vLLM-Omni | `vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni` on the `vllm/vllm-omni:v0.32.0` image | needs vLLM 0.32; the platform's Ray 2.60 image has vLLM 0.29, so local only for now (see the studio README) |

## Before listing

1. Boot on a 16 GB card; record peak VRAM (text encoder, denoise, decode)
   and the time for 49 / 121 frames.
2. If it OOMs: `offload: "group"`, then measure the slowdown.
3. Check one T2V and one I2V clip play in a browser (H.264 via imageio-ffmpeg).
   The platform installs `numpy<2` for predictor jobs; diffusers < 0.40
   with transformers 4.57 runs on numpy 1.x, which is why they are pinned
   below the transformers-5 line.
4. Drop the "not booted" marker.

## License

Weights Apache-2.0 (Wan-AI). Generated content is yours; the model card
asks you not to produce illegal or harmful content.
