# LTX-2.3 22B: text/image-to-video with sound (predictor mode)

**Written 2026-11-19, not booted.** Lightricks' 22B audio-video model: one
pass makes the clip and a matching stereo track (ambience, effects,
speech). Runs Lightricks' own `ltx-pipelines` package and its
`DistilledPipeline`: 8 steps at half resolution, a 2× spatial upsample,
4 refinement steps at full resolution, no CFG. 1280×704 at 24 fps, 121
frames (5 s) by default, up to 241 (10 s).

Silent clips under Apache-2.0: [`../Wan2.2-TI2V-5B_videogen_min_16gb_vram`](../Wan2.2-TI2V-5B_videogen_min_16gb_vram/README.md).

## ⚠️ License: LTX-2 Community License, not Apache

- Companies with **≥ $10M annual revenue** (affiliates included) need a
  paid commercial license from Lightricks (§2).
- Hosting as a service is allowed if your terms pass on the use
  restrictions of Attachment A (§3).
- Attachment A #20: no use in a product that **directly competes with
  Lightricks' products** without a separate license. A hosted
  text-to-video API can fall under it.
- Lightricks may restrict use remotely (§6).

Read `LICENSE-2` in the model repo before selling an endpoint backed by
this job. The `ltx-pipelines` code itself is Apache-2.0.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 32 GB (48 GB runs resident) | 8 | 64 GB | ~105 GB (image 10 + 1.3 × 72 GB) | 16gb (slow) / 32gb / 48gb |

Weights: the distilled checkpoint (46.2 GB bf16, cast to FP8 on load →
22 GB), Gemma 3 12B as text encoder (24.4 GB, gated) and the x2 spatial
upsampler (1 GB). Below 48 GB the job sets `offload: "cpu"`: weights stay
pinned in RAM and stream to the GPU layer by layer, about 5 GB of VRAM
for the weights plus activations. FP8 needs compute capability 8.9 or
newer: RTX 40xx / 50xx, L40S, H100. No RTX 3090, no A100.

No measured timings yet. The DOU article's RTX 5080 Laptop (16 GB, NVFP4,
ComfyUI) needed ~51 s cold for a 2 s 512×288 clip; 1280×704 × 5 s is
roughly 10× the pixels. Plan for minutes per clip on 16–24 GB.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" python-dotenv   # Python 3.11+
cp .env.example .env
# HF_TOKEN: accept the terms on https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
# with your HF account, then create a read token for it
python submit.py            # first boot pulls ~72 GB
python predict.py "rain on a tin roof at night, a dog barks in the distance, slow dolly in"
python predict.py --image photo.jpg "she turns to the camera and says: 'we open at nine'"
```

`submit.py` copies `inference.yaml` to a temp file with `hf_token` filled
in and submits the copy, so the token never lands in the repo. It does
land in the job's config on the platform: use a fine-grained read-only
token limited to the Gemma repo.

## Request / response

Same polling contract as the Wan job: one request to a predictor job is cut off after 300 s (set on the Ray head node; the predict wrapper answers 504 at 290 s), so each call returns
within `wait_s` (default 20, so a start returns the id fast; at most 280) with the clip or a job to poll with
`{"action": "status", "id": ...}`.

| field | default | notes |
|---|---|---|
| `prompt` | required | describe the sound too: the audio follows the prompt |
| `image_b64` | none | becomes frame 0; resized and center-cropped by the pipeline |
| `image_strength` | 1.0 | lower lets the model drift from the image |
| `width` / `height` | 1280×704, or the image's aspect at that area | multiples of 64 (two-stage) |
| `num_frames` | 121 | 8k+1; at most `max_frames` (241) |
| `seed` | random | always returned |

Finished: `{"status": "completed", "video_b64": "<MP4: H.264 + AAC>", "mode", "width", "height", "num_frames", "fps", "seconds", "seed", "elapsed_ms"}`.
While rendering the job reports `running_s`; the pipeline exposes no step
counter.

## What to change

| where | for what |
|---|---|
| `init_args.offload` | `"auto"`; `"none"` resident (48 GB+ with FP8), `"cpu"` streams from RAM, `"disk"` when RAM is short too (slowest) |
| `init_args.quantization` | `"fp8-cast"`; `null` for bf16 (80 GB resident); see the FP8 checkpoint variant |
| `init_args.max_frames` | 241 = 10 s; longer clips cost memory and time |

## Variants

| Variant | Edits | Status |
|---|---|---|
| FP8 checkpoint (29.5 GB download instead of 46.2) | `model_repo: "Lightricks/LTX-2.3-fp8"`, `checkpoint: "ltx-2.3-22b-distilled-fp8.safetensors"`, `quantization: "fp8-scaled-mm"` | native FP8 matmul, "best on Hopper+" per Lightricks; verify on RTX 50xx |
| NVFP4 (as in the DOU article) | `Lightricks/LTX-2.3-nvfp4`, `quantization: "nvfp4-prequant"` | dev model only (distilled NVFP4 "coming soon"); needs `ltx-kernels` built with CUDA ≥ 12.8 nvcc, which the predictor image does not have |
| Two-stage dev (`TI2VidTwoStagesPipeline`) | dev checkpoint + distilled LoRA 0.8, 30 steps with CFG/STG | higher quality, ~3× slower; not wired here |
| diffusers `LTX2Pipeline` (≥ 0.38) | `diffusers/LTX-2.3-Diffusers` | ~101 GB repo (fp32 text encoder), single-stage only in the docs |
| vLLM-Omni | `vllm serve diffusers/LTX-2.3-Distilled-Diffusers --omni --model-class-name LTX2DistilledTwoStagePipeline` | needs vLLM 0.32 (image `vllm/vllm-omni:v0.32.0`); local only for now |

## Before listing

1. **Torch match.** `ltx-core` pins `torch~=2.7` and pulls `torchaudio` /
   `torchvision`. Check the predictor image's torch (`pip show torch`) and
   pin `torchaudio==<same>` / `torchvision==<matching>` in `requirements:`
   if pip tries to replace torch at boot.
2. **Archive installs.** The two GitHub archive URLs build with `uv_build`;
   confirm the platform's installer accepts direct-URL requirements.
   **numpy:** the platform installs `numpy<2` for every predictor job;
   check that `colour-science`, `scipy>=1.14` and `openimageio` resolve
   against numpy 1.x (pyannote's job is blocked by the same pin).
3. Boot on a 24 GB card first (`offload: "cpu"`), then on 16 GB; record
   peak VRAM, RAM and the time for 97 / 121 frames.
4. Check the MP4 plays in a browser with sound.
5. Drop the "not booted" marker.
