"""Media studio: a chat model that plans + image and video generators, ONE job, one GPU.

    "assistant" → Qwen3-8B AWQ (vLLM)            POST /v1/chat/completions    tool calling, reasoning
    images      → Z-Image-Turbo 6B (diffusers)   POST /v1/images/generations  OpenAI Images API shape
    videos      → Wan2.2-TI2V-5B (diffusers)     POST /v1/videos              OpenAI / vLLM-Omni Videos API shape
                                                 GET  /v1/videos/{id}          status + progress
                                                 GET  /v1/videos/{id}/content  the MP4

All three models are Apache-2.0. The chat model writes the shot list and
the prompts and calls the generators through ordinary OpenAI tool calls;
predict.py runs that loop on the client (brief → keyframe image → the
keyframe animated into a clip). Any OpenAI-compatible agent framework can
do the same against this endpoint.

NEEDS the Ray 2.60.0 image (vLLM 0.29.0, transformers 5.17, CUDA 13,
driver R580+). Written 2026-11-19; nothing has booted.

How it fits in one Ray Serve app
--------------------------------
build_openai_app() builds the OpenAI router (OpenAiIngress) in front of
the vLLM engine. Its `ingress_cls_config` takes a subclass plus extra
constructor kwargs: StudioIngress adds the image and video routes and
receives a handle to the MediaStudio deployment, which runs diffusers on
its own share of the GPU. Ray Serve LLM only routes the method names in an
endpoint map, and build_openai_app always uses the module default, so
_build_app() swaps that default for the duration of the build. That map
(ray.llm._internal.serve.core.ingress.ingress.DEFAULT_ENDPOINTS) is the one
internal Ray name used here; re-check it when the Ray image changes.

diffusers is installed into the media actor only (runtime_env pip), so the
vLLM processes keep the image's exact package set.

Why diffusers and not vLLM for the generators: vLLM-Omni serves Wan2.2,
LTX-2.3, FLUX and Qwen-Image with these same /v1/images and /v1/videos
routes, but needs vLLM 0.32 and transformers < 5.15; this image has vLLM
0.29 and transformers 5.17. Clients written against this job move to a
vLLM-Omni endpoint unchanged.

GPU budget (LAYOUT below)
-------------------------
"one_gpu" (default): one card ≥ 32 GB (RTX 5090, L40S, RTX PRO 6000, H100)
    assistant  0.30 of the card: on 32 GB that is 9.6 GB = 5.9 weights
               + ~0.8 activations + 2.4 GiB fp8 KV (~34k tokens)
    media      the rest. Below 60 GB of share the generators run with
               model CPU offload, one component on the GPU at a time:
               peak ≈ 16 GB (Wan's 11.4 GB text encoder, or its transformer
               plus 720p activations; VAE decode is tiled). Images and
               clips take turns on the GPU.
"two_gpus": one worker with two 16 GB cards (2 × RTX 5070 Ti / 5080)
    assistant  0.85 of card A, media all of card B. Set gpus_per_worker: 2
               in inference.yaml.

The yaml's `vllm:` block MUST mirror the two *_size values (1 and 1).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
import random
import tempfile
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
from ray.serve.llm.ingress import OpenAiIngress


LAYOUT = "one_gpu"   # "one_gpu" | "two_gpus" (then gpus_per_worker: 2 in inference.yaml)

if LAYOUT == "one_gpu":
    LLM_FRACTION, LLM_KV_GIB, MEDIA_GPUS = 0.30, 2.4, 0.65
elif LAYOUT == "two_gpus":
    LLM_FRACTION, LLM_KV_GIB, MEDIA_GPUS = 0.85, 6.0, 1.0
else:
    raise ValueError(f"LAYOUT must be 'one_gpu' or 'two_gpus', got {LAYOUT!r}")

VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"   # 34.2 GB repo
IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"          # 32.9 GB repo

# Installed into the media actor's own virtualenv on top of the image.
# diffusers ≥ 0.40 is the line that supports transformers 5.
MEDIA_PIP = [
    "diffusers>=0.43.0",
    "accelerate>=1.0",
    "ftfy",                  # Wan's prompt cleaning
    "imageio>=2.30",         # H.264 MP4; without it export_to_video writes mp4v
    "imageio-ffmpeg>=0.5",
]

VIDEO_FPS = 24
VIDEO_MAX_FRAMES = 121          # 5 s, the model's training length
VIDEO_MAX_AREA = 1280 * 704
VIDEO_GRID = 32                 # Wan TI2V-5B: sides multiples of 32
IMAGE_GRID = 16                 # Z-Image: VAE 8× × patch 2
IMAGE_MAX_SIDE = 2048
GPU_WAIT_S = 240                # an image request waits this long behind a clip, then gets 503
MAX_QUEUED_VIDEOS = 4
KEEP_VIDEOS_S = 3600

# Official Wan2.2 `sample_neg_prompt`; the model was tuned with it.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def card_share(fraction):
    """The chat engine's share of the GPU (same pattern as the agents bundle).

    Ray schedules GPUs separately from gpu_memory_utilization: the bundle
    reserves the fraction, VLLM_RAY_PER_WORKER_GPUS makes vLLM's worker
    request the same fraction instead of a whole GPU.
    """
    return dict(
        placement_group_config=dict(bundles=[dict(GPU=fraction)]),
        runtime_env=dict(env_vars={
            "VLLM_RAY_PER_WORKER_GPUS": str(fraction),
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        }),
    )


assistant = LLMConfig(
    model_loading_config=dict(
        model_id="assistant",
        model_source="Qwen/Qwen3-8B-AWQ",        # official AWQ 4-bit, 6.1 GB
    ),
    accelerator_type=None,                       # consumer cards carry no Ray label
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=1),
        max_ongoing_requests=8,
        max_queued_requests=64,
        graceful_shutdown_timeout_s=120,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,                  # = yaml vllm.tensor_parallel_size
        pipeline_parallel_size=1,                # = yaml vllm.pipeline_parallel_size
        max_model_len=16384,
        max_num_seqs=4,
        gpu_memory_utilization=LLM_FRACTION,
        # Fixed KV size: with it gpu_memory_utilization only checks free
        # VRAM at boot, so the media actor's CUDA context next to it does
        # not shrink the cache (see the agents bundle).
        kv_cache_memory_bytes=int(LLM_KV_GIB * 2**30),
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,              # the agent loop resends the history every turn
        enable_chunked_prefill=True,
        dtype="auto",
        # ── Tool calling (Qwen3 → hermes) and thinking ──
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
        # Qwen3 thinks by default; the parser moves it to the `reasoning`
        # field. Callers turn it off with chat_template_kwargs={"enable_thinking": False}.
        reasoning_parser="qwen3",
    ),
    **card_share(LLM_FRACTION),
)


@serve.deployment(
    name="media",
    num_replicas=1,                  # ONE replica: video ids live in its memory
    max_ongoing_requests=32,         # status polls and downloads are cheap; the GPU work is serialized
    graceful_shutdown_timeout_s=60,
    ray_actor_options={
        "num_gpus": MEDIA_GPUS,
        "num_cpus": 4,
        "runtime_env": {"pip": {"packages": MEDIA_PIP, "pip_check": False}},
    },
)
class MediaStudio:
    """Z-Image-Turbo stills and Wan2.2 clips on one GPU share, one job at a time."""

    def __init__(self, video_model: str, image_model: str, offload: str = "auto") -> None:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, ZImagePipeline

        self.video_model, self.image_model = video_model, image_model

        # Wan: VAE in fp32 (model card), the rest bf16; one set of weights
        # serves text-to-video and image-to-video.
        vae = AutoencoderKLWan.from_pretrained(video_model, subfolder="vae", torch_dtype=torch.float32)
        self.t2v = WanPipeline.from_pretrained(video_model, vae=vae, torch_dtype=torch.bfloat16)
        self.i2v = WanImageToVideoPipeline.from_pipe(self.t2v)
        vae.enable_tiling()                      # the untiled 720p decode spikes to ~26 GB
        # Z-Image-Turbo: 6B DiT + Qwen3-4B text encoder, 8 steps, no CFG.
        self.t2i = ZImagePipeline.from_pretrained(image_model, torch_dtype=torch.bfloat16)

        share_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 * min(MEDIA_GPUS, 1.0)
        if offload == "auto":
            # Resident: Wan ~24 GB + Z-Image ~20 GB + activations.
            offload = "off" if share_gb >= 60 else "model"
        if offload == "off":
            self.t2v.to("cuda")
            self.t2i.to("cuda")
        else:
            self.t2i.enable_model_cpu_offload()  # Wan's hooks are installed per run in _use()
        self.offload = offload
        self._hooked = None

        self._gpu = threading.Lock()             # one image or one clip on the GPU at a time
        self._videos: dict[str, dict[str, Any]] = {}
        self._vlock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._video_worker, name="wan-worker", daemon=True).start()

    # ── images ──────────────────────────────────────────────────────────────

    async def generate_image(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            params = _image_params(body)
        except ValueError as exc:
            return _err(400, str(exc))
        try:
            pngs = await asyncio.to_thread(self._render_images_locked, params)
        except torch.cuda.OutOfMemoryError:
            return _err(500, "CUDA out of memory: request a smaller size or fewer images")
        if pngs is None:
            return _err(503, "the GPU is busy rendering a video clip; retry in a minute")
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(p).decode("ascii")} for p in pngs],
            "model": "z-image-turbo",
            "size": f"{params['width']}x{params['height']}",
            "seed": params["seed"],
        }

    def _render_images_locked(self, p: dict[str, Any]) -> list[bytes] | None:
        """Lock, render, unlock in one thread: a cancelled request cannot leave the GPU locked."""
        if not self._gpu.acquire(timeout=GPU_WAIT_S):
            return None
        try:
            return self._render_images(p)
        finally:
            self._gpu.release()

    def _render_images(self, p: dict[str, Any]) -> list[bytes]:
        with torch.inference_mode():
            images = self.t2i(
                prompt=p["prompt"],
                height=p["height"],
                width=p["width"],
                num_inference_steps=p["steps"],
                guidance_scale=0.0,              # Turbo is distilled: CFG must stay off
                num_images_per_prompt=p["n"],
                generator=torch.Generator("cpu").manual_seed(p["seed"]),
            ).images
        torch.cuda.empty_cache()
        out = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out

    # ── videos ──────────────────────────────────────────────────────────────

    async def create_video(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            params = _video_params(body)
        except ValueError as exc:
            return _err(400, str(exc))
        with self._vlock:
            cutoff = time.time() - KEEP_VIDEOS_S
            for vid in [k for k, j in self._videos.items() if j["finished"] and j["finished"] < cutoff]:
                del self._videos[vid]
            waiting = sum(1 for j in self._videos.values() if j["status"] == "queued")
            if waiting >= MAX_QUEUED_VIDEOS:
                return _err(429, f"{waiting} clips are already waiting; retry later")
            job = {
                "id": f"video_{uuid.uuid4().hex[:16]}", "status": "queued", "progress": 0.0,
                "created": time.time(), "finished": None, "params": params,
                "mp4": None, "error": None, "elapsed_s": None,
            }
            self._videos[job["id"]] = job
        self._queue.put(job)
        return self._video_object(job)

    async def get_video(self, video_id: str) -> dict[str, Any]:
        job = self._videos.get(video_id)
        return self._video_object(job) if job else _err(404, f"no video {video_id!r} (kept 1 h, lost on restart)")

    async def video_content(self, video_id: str) -> bytes | dict[str, Any]:
        job = self._videos.get(video_id)
        if job is None:
            return _err(404, f"no video {video_id!r} (kept 1 h, lost on restart)")
        if job["status"] != "completed":
            return _err(409, f"video {video_id} is {job['status']}, not completed")
        return job["mp4"]

    def _video_object(self, job: dict[str, Any]) -> dict[str, Any]:
        p = job["params"]
        return {
            "id": job["id"], "object": "video", "model": "wan2.2-ti2v-5b",
            "status": job["status"], "progress": int(job["progress"] * 100),
            "created_at": int(job["created"]),
            "completed_at": int(job["finished"]) if job["status"] == "completed" else None,
            "size": f"{p['width']}x{p['height']}",
            "seconds": f"{p['num_frames'] / VIDEO_FPS:.2f}",
            "num_frames": p["num_frames"], "fps": VIDEO_FPS, "seed": p["seed"],
            "mode": "i2v" if p["had_image"] else "t2v",
            "elapsed_s": job["elapsed_s"],
            "error": {"code": "generation_failed", "message": job["error"]} if job["error"] else None,
        }

    def _video_worker(self) -> None:
        while True:
            job = self._queue.get()
            with self._gpu:
                job["status"] = "in_progress"
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode():
                        job["mp4"] = self._render_video(job)
                    job["status"] = "completed"
                except torch.cuda.OutOfMemoryError:
                    job["error"] = "CUDA out of memory: lower seconds/num_frames or the size"
                    job["status"] = "failed"
                except Exception as exc:  # report, keep the worker alive
                    job["error"] = f"{type(exc).__name__}: {exc}"
                    job["status"] = "failed"
                finally:
                    job["elapsed_s"] = round(time.perf_counter() - t0, 1)
                    job["finished"] = time.time()
                    job["params"]["image"] = None
                    torch.cuda.empty_cache()

    def _render_video(self, job: dict[str, Any]) -> bytes:
        from diffusers.utils import export_to_video

        p = job["params"]

        def on_step(pipe, step, timestep, callback_kwargs):
            job["progress"] = (step + 1) / p["steps"]
            return callback_kwargs

        kwargs = dict(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            height=p["height"],
            width=p["width"],
            num_frames=p["num_frames"],
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance_scale"],
            generator=torch.Generator("cpu").manual_seed(p["seed"]),
            callback_on_step_end=on_step,
        )
        if p["image"] is not None:
            pipe, kwargs["image"] = self.i2v, p["image"]
        else:
            pipe = self.t2v
        self._use(pipe)
        frames = pipe(**kwargs).frames[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            export_to_video(frames, path, fps=VIDEO_FPS, quality=8)
            with open(path, "rb") as f:
                return f.read()

    def _use(self, pipe) -> None:
        """The two Wan pipelines share modules; re-install offload hooks on a switch."""
        if self.offload == "model" and self._hooked is not pipe:
            pipe.enable_model_cpu_offload()
            self._hooked = pipe


class StudioIngress(OpenAiIngress):
    """The OpenAI router plus image and video routes backed by MediaStudio."""

    def __init__(self, *args, media=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._media = media

    async def images_generations(self, request: Request) -> Response:
        body = await _json_body(request)
        if body is None:
            return _respond(_err(400, "send a JSON body"))
        return _respond(await self._media.generate_image.remote(body))

    async def videos_create(self, request: Request) -> Response:
        """JSON (`input_reference_b64`) or multipart (`input_reference` file, as openai-python sends)."""
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            body: dict[str, Any] = {}
            for key, value in form.items():
                if hasattr(value, "read"):
                    if key == "input_reference":
                        body["image_bytes"] = await value.read()
                else:
                    body[key] = value
        else:
            body = await _json_body(request)
            if body is None:
                return _respond(_err(400, "send JSON or multipart/form-data"))
            body.pop("image_bytes", None)        # internal key; JSON clients send base64
            ref = body.pop("input_reference_b64", None) or body.pop("image_b64", None)
            if ref:
                if ref.startswith("data:"):
                    ref = ref.split(",", 1)[1]
                body["image_bytes"] = base64.b64decode(ref)
        return _respond(await self._media.create_video.remote(body))

    async def videos_retrieve(self, video_id: str) -> Response:
        return _respond(await self._media.get_video.remote(video_id))

    async def videos_content(self, video_id: str) -> Response:
        result = await self._media.video_content.remote(video_id)
        if isinstance(result, (bytes, bytearray)):
            return Response(content=bytes(result), media_type="video/mp4")
        return _respond(result)


MEDIA_ENDPOINTS = {
    "images_generations": lambda app: app.post("/v1/images/generations"),
    "videos_create": lambda app: app.post("/v1/videos"),
    "videos_retrieve": lambda app: app.get("/v1/videos/{video_id}"),
    "videos_content": lambda app: app.get("/v1/videos/{video_id}/content"),
}


def _build_app():
    from ray.llm._internal.serve.core.ingress import ingress as ray_ingress

    media = MediaStudio.bind(video_model=VIDEO_MODEL, image_model=IMAGE_MODEL)
    defaults = ray_ingress.DEFAULT_ENDPOINTS
    ray_ingress.DEFAULT_ENDPOINTS = {**defaults, **MEDIA_ENDPOINTS}
    try:
        return build_openai_app({
            "llm_configs": [assistant],
            "ingress_cls_config": {
                "ingress_cls": StudioIngress,
                "ingress_extra_kwargs": {"media": media},
            },
        })
    finally:
        ray_ingress.DEFAULT_ENDPOINTS = defaults


# ── request parsing (shared by the routes above) ────────────────────────────

def _err(status: int, message: str) -> dict[str, Any]:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return {"_status": status, "error": {"message": message, "type": kind, "code": status}}


def _respond(result: dict[str, Any]) -> JSONResponse:
    result = dict(result)
    return JSONResponse(result, status_code=result.pop("_status", 200))


async def _json_body(request: Request) -> dict[str, Any] | None:
    try:
        body = json.loads(await request.body() or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _parse_size(size: Any, default: tuple[int, int]) -> tuple[int, int]:
    if size in (None, "", "auto"):
        return default
    try:
        width, height = (int(v) for v in str(size).lower().split("x"))
    except ValueError:
        raise ValueError(f"size must look like '1280x704', got {size!r}") from None
    return width, height


def _seed(body: dict[str, Any]) -> int:
    return int(body["seed"]) if body.get("seed") not in (None, "") else random.randint(0, 2**31 - 1)


def _image_params(body: dict[str, Any]) -> dict[str, Any]:
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("missing 'prompt'")
    if body.get("response_format", "b64_json") != "b64_json":
        raise ValueError("only response_format 'b64_json' is supported")
    width, height = _parse_size(body.get("size"), (1024, 1024))
    width = max(256, min(width, IMAGE_MAX_SIDE)) // IMAGE_GRID * IMAGE_GRID
    height = max(256, min(height, IMAGE_MAX_SIDE)) // IMAGE_GRID * IMAGE_GRID
    return {
        "prompt": prompt, "width": width, "height": height,
        "n": max(1, min(int(body.get("n", 1)), 4)),
        # 9 scheduler steps = 8 transformer passes (the model card's setting).
        "steps": max(1, min(int(body.get("num_inference_steps", body.get("steps", 9))), 30)),
        "seed": _seed(body),
    }


def _video_params(body: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("missing 'prompt'")
    width, height = _parse_size(body.get("size"), (1280, 704))
    # Floor to the grid first (1280x720 → 1280x704), then scale down if still too big.
    width = max(256, width // VIDEO_GRID * VIDEO_GRID)
    height = max(256, height // VIDEO_GRID * VIDEO_GRID)
    if width * height > VIDEO_MAX_AREA:
        scale = (VIDEO_MAX_AREA / (width * height)) ** 0.5
        width = max(256, int(width * scale) // VIDEO_GRID * VIDEO_GRID)
        height = max(256, int(height * scale) // VIDEO_GRID * VIDEO_GRID)

    image = None
    if body.get("image_bytes"):
        image = Image.open(io.BytesIO(body["image_bytes"])).convert("RGB")
        image = _fit_image(image, width * height)    # keep the image's aspect ratio
        width, height = image.size

    if body.get("num_frames") not in (None, ""):
        frames = int(body["num_frames"])
    elif body.get("seconds") not in (None, ""):
        frames = round(float(body["seconds"]) * VIDEO_FPS) + 1
    else:
        frames = VIDEO_MAX_FRAMES
    frames = max(5, min(frames, VIDEO_MAX_FRAMES))
    frames = (frames - 1) // 4 * 4 + 1                 # Wan needs 4k+1

    return {
        "prompt": prompt,
        "negative_prompt": body.get("negative_prompt") or WAN_NEGATIVE_PROMPT,
        "image": image, "had_image": image is not None,
        "width": width, "height": height, "num_frames": frames,
        "steps": max(1, min(int(body.get("num_inference_steps", body.get("steps", 50))), 100)),
        "guidance_scale": float(body.get("guidance_scale", 5.0)),
        "seed": _seed(body),
    }


def _fit_image(image, area: int):
    """Wan-style input sizing: keep the aspect, resize, center-crop to the 32-px grid."""
    from PIL import Image

    w, h = image.size
    ratio = w / h
    out_w = max(256, int((area * ratio) ** 0.5) // VIDEO_GRID * VIDEO_GRID)
    out_h = max(256, int((area / ratio) ** 0.5) // VIDEO_GRID * VIDEO_GRID)
    scale = max(out_w / w, out_h / h)
    image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    left, top = (image.width - out_w) // 2, (image.height - out_h) // 2
    return image.crop((left, top, left + out_w, top + out_h))


app = _build_app()
