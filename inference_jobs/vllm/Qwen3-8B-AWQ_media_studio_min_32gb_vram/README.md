# Media studio: chat model + image + video generators in one job

**Needs Ray 2.60.0. Written 2026-11-19, not booted.** One endpoint, one
GPU, three models, all Apache-2.0:

| Route | Model | What |
|---|---|---|
| `POST /v1/chat/completions` (`model: "assistant"`) | Qwen3-8B AWQ on vLLM | plans the shots, writes the prompts, calls the generators as tools; thinking on by default |
| `POST /v1/images/generations` | Z-Image-Turbo 6B (diffusers) | stills in 8 steps, OpenAI Images API shape |
| `POST /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content` | Wan2.2-TI2V-5B (diffusers) | 5 s clips at 24 fps from text or from a still, OpenAI / vLLM-Omni Videos API shape |

[`predict.py`](predict.py) runs the agent loop: the brief goes to the
assistant, it makes a keyframe with `generate_image`, then animates it with
`generate_video(image_id=...)`, and answers with the file names. Any
OpenAI-compatible agent framework can drive the same three routes.

Stand-alone generators: [`../../predict/Wan2.2-TI2V-5B_videogen_min_16gb_vram`](../../predict/Wan2.2-TI2V-5B_videogen_min_16gb_vram/README.md)
(16 GB), [`../../predict/LTX-2.3-22B_videogen_audio_min_16gb_vram`](../../predict/LTX-2.3-22B_videogen_audio_min_16gb_vram/README.md)
(video with sound), [`../../predict/FLUX.1-schnell_imagegen_min_16gb_vram`](../../predict/FLUX.1-schnell_imagegen_min_16gb_vram/README.md).

## How it is wired

```
OpenAI router (StudioIngress = OpenAiIngress + 4 routes)
 ├── /v1/chat/completions ─► LLMServer "assistant"  (vLLM, 0.30 of the GPU)
 └── /v1/images, /v1/videos ─► MediaStudio          (diffusers, 0.65 of the GPU)
                                ├─ Z-Image-Turbo     one job at a time on the GPU
                                └─ Wan2.2-TI2V-5B    clips render in a background thread
```

- `build_openai_app()` takes a subclass of the OpenAI router plus extra
  constructor kwargs (`ingress_cls_config`). The subclass adds the four
  routes and gets a handle to the `MediaStudio` deployment.
- Ray Serve LLM routes only the method names in its endpoint map, so
  `_build_app()` extends the map (`DEFAULT_ENDPOINTS`, an internal Ray name)
  for the duration of the build.
- diffusers is pip-installed into the media actor only
  (`runtime_env`), so the vLLM processes keep the image's exact packages.

**Why not vLLM for the generators?** vLLM-Omni serves Wan2.2, LTX-2.3,
FLUX and Qwen-Image with these same `/v1/images/generations` and
`/v1/videos` routes. It needs vLLM 0.32 and transformers < 5.15, and the
Ray 2.60 image has vLLM 0.29 with transformers 5.17. vLLM-Omni also runs
one model per server, with no Qwen3 → Wan chain. The routes here use the
same request shape, so a client moves to a vLLM-Omni endpoint unchanged.

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 32 → 48 GB on one card (or 2 × 16 GB, see Layouts) | 12 | 64 GB (96 comfortable) | ~115 GB (image 20 + 1.3 × 73 GB) | 32gb / 48gb |

| Part | GPU share | Memory |
|---|---|---|
| assistant (Qwen3-8B AWQ) | 0.30 of the card | 9.6 GB on 32 GB: 5.9 weights + ~0.8 activations + 2.4 GiB fp8 KV (~34k tokens), 16k context, 4 sequences |
| media (Wan2.2 + Z-Image) | 0.65 of the card | model CPU offload below 60 GB of share: one component on the GPU at a time, peak ≈ 16 GB (Wan's 11.4 GB text encoder, or its transformer with 720p activations); VAE decode tiled. Resident on 96 GB cards |

Images and clips take turns on the GPU. An image request waits up to
240 s behind a running clip, then gets `503` (predict.py retries).

| Estimate (not measured) | RTX 5090 32 GB | L40S 48 GB |
|---|---|---|
| Z-Image 1024×1024, 8 steps (offloaded) | ~5–10 s | ~6–12 s |
| Wan clip 5 s 1280×704, 50 steps | ~5–7 min | ~6–8 min |

### Layouts (`LAYOUT` in serve_module.py)

| LAYOUT | Hardware | Edits |
|---|---|---|
| `"one_gpu"` (shipped) | one card ≥ 32 GB: RTX 5090, L40S / L40, H100 | none |
| `"two_gpus"` | one worker with two 16 GB cards (2 × RTX 5070 Ti / 5080) | `gpus_per_worker: 2`, `min_vram_gb_per_worker: 32`; the assistant takes 0.85 of card A (6 GiB KV), the generators card B |

A single 16 GB card cannot hold the chat engine next to the generators.
On one 16 GB card, run the stand-alone Wan job and drive it from any chat
endpoint.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv httpx   # Python 3.11+
cp .env.example .env
python submit.py            # ~73 GB of weights + diffusers install on first boot
python predict.py           # brief → keyframe → 5 s clip, files in out/
python predict.py "Тизер на 5 секунд для відкриття кав'ярні у Львові: ранкове світло, пара над чашкою"
python predict.py --no-think "Постер для вебінару про AI у банкінгу, без тексту"
python predict.py --image-only "isometric game asset, a watermill by a river"
python predict.py --video-only "a paper boat drifts down a rainy street, low tracking shot"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## API

The key goes as both `Authorization: Bearer` and `X-API-Key`.

**Images** (JSON): `prompt`, `size` (`"1024x1024"`, multiples of 16, ≤ 2048
per side), `n` (1–4), `seed`, `num_inference_steps` (9 = 8 passes, the card's
setting). Only `response_format: "b64_json"`. Reply:
`{"created", "data": [{"b64_json": "<PNG>"}], "model", "size", "seed"}`.

**Videos** (JSON, or multipart with an `input_reference` file as openai-python sends it):

| field | default | notes |
|---|---|---|
| `prompt` | required | motion and camera move; English or Chinese work best |
| `input_reference_b64` / `input_reference` file | none | image-to-video; the image keeps its aspect, becomes frame 0 |
| `seconds` or `num_frames` | 5 s / 121 | frames snap to 4k+1, at most 121 |
| `size` | `"1280x704"` | multiples of 32, at most 1280×704 of area; `"704x1280"` vertical |
| `num_inference_steps`, `guidance_scale`, `seed`, `negative_prompt` | 50, 5.0, random, official | |

`POST /v1/videos` returns at once with `{"id": "video_…", "object": "video",
"status": "queued", "progress": 0, ...}`. Poll `GET /v1/videos/{id}` until
`status` is `completed` or `failed`, then download the MP4 from
`GET /v1/videos/{id}/content`. Clips stay for an hour and are lost on a
restart. At most 4 wait behind the running one (`429` after that).

```bash
curl -s "$URL/v1/videos" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a paper boat drifts down a rainy street, low tracking shot", "seconds": "3"}'
curl -s "$URL/v1/videos/video_…" -H "X-API-Key: $KEY"
curl -s "$URL/v1/videos/video_…/content" -H "X-API-Key: $KEY" -o clip.mp4
```

`GET /v1/models` lists only `assistant`; the generators are routes, not
model ids.

## Try vLLM-Omni locally (e.g. one RTX 5070 Ti)

vLLM-Omni 0.32 serves the same generators with the same API on its own
image. It does not fit the platform's Ray image yet. On a local
Blackwell card with driver R580+ (CUDA 13):

```bash
# Wan2.2-TI2V-5B, text/image-to-video on :8091 (16 GB: offload on)
docker run --gpus all --ipc=host -p 8091:8091 -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-omni:v0.32.0 \
  vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --port 8091 \
    --vae-use-tiling --enable-cpu-offload --enforce-eager

curl --fail-with-body -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F "prompt=a paper boat drifts down a rainy street, low tracking shot" \
  -F "width=1280" -F "height=704" -F "num_frames=49" -F "fps=24" \
  -F "num_inference_steps=30" -F "guidance_scale=5.0" -F "seed=42" --output boat.mp4

# LTX-2.3 distilled with sound (diffusers-format repo, ~101 GB)
docker run --gpus all --ipc=host -p 8092:8092 -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-omni:v0.32.0 \
  vllm serve diffusers/LTX-2.3-Distilled-Diffusers --omni --port 8092 \
    --model-class-name LTX2DistilledTwoStagePipeline --enable-layerwise-offload --stage-init-timeout 600
```

Notes from the vLLM-Omni docs:
- `num_frames` defaults to 1 online, so always set it: Wan wants 4k+1, LTX 8k+1.
- `/v1/videos/sync` times out at 600 s (`VLLM_OMNI_VIDEO_SYNC_TIMEOUT`). Use
  async `POST /v1/videos` for long clips.
- There is no sm_120 CI lane yet, and 16 GB video runs are not validated upstream.

## Before listing

1. **Routes through the edge.** Confirm the platform's proxy forwards
   `/v1/images/generations` and `/v1/videos*` (not only the chat routes)
   and allows a 10 MB MP4 response.
2. **runtime_env pip.** Confirm the platform lets a deployment pip-install
   into its actor; if not, move `MEDIA_PIP` to `requirements:` and check
   vLLM still boots (diffusers 0.43 wants huggingface-hub ≥ 1.32).
3. **Ray internals.** `DEFAULT_ENDPOINTS` and the
   `OpenAiIngress(llm_deployments, model_cards, ...)` constructor are
   from Ray 2.60.0; re-check them on an image change.
4. Boot on an RTX 5090. Record VRAM per part (`nvidia-smi`), one image,
   and one 5 s clip, and check vLLM keeps answering while a clip renders.
5. Run `predict.py` end to end; check that the hermes parser turns
   Qwen3's calls into `tool_calls` with thinking on.
6. Drop the "not booted" marker.

## Variants

| Variant | Edits |
|---|---|
| Bigger planner | `Qwen3.5-9B-AWQ` (sees images too: it can review the keyframe) with `tool_call_parser="qwen3_coder"`; budget ~1 GB more |
| Video with sound | swap Wan for LTX-2.3: not in this image (`ltx-core` pins transformers < 5.15); use the stand-alone LTX job as a second endpoint |
| Best text in images | Qwen-Image 20B instead of Z-Image: 40 GB+ on its own, so a 96 GB card |
