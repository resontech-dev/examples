"""Wan2.2-TI2V-5B: text or image in, MP4 out (predictor mode).

One 5B model (Apache-2.0) does text-to-video and image-to-video at
1280×704, 24 fps; 121 frames is 5 s. The video has no sound.

Request (JSON via ``client.predict_json``):

    {"prompt": "a red fox trots through fresh snow, low tracking shot",
     "image_b64": "<PNG or JPEG, base64>",   # optional: animate this image (it becomes frame 0)
     "width": 1280, "height": 704,           # optional; multiples of 32, area capped at max_area
     "num_frames": 121,                      # optional; 4k+1: 49 ≈ 2 s, 81 ≈ 3.4 s, 121 ≈ 5 s
     "steps": 50, "guidance_scale": 5.0,     # optional
     "seed": 42, "negative_prompt": "...",   # optional
     "wait_s": 20}                           # optional: how long this call waits for the clip (max 280)

A bare text body is treated as {"prompt": <body>}.

A clip takes minutes and one request to a predictor job is cut off after
300 s (set on the Ray head node; the predict wrapper answers 504 at 290 s),
so the clip is rendered by a background thread. Every call returns within
``wait_s`` with either the finished clip or a job to poll:

    {"action": "status", "id": "<id>", "wait_s": 20}

Responses:

    {"id": "...", "status": "queued", "queue_position": 0}
    {"id": "...", "status": "in_progress", "progress": 0.42}
    {"id": "...", "status": "completed", "video_b64": "<base64 MP4>", "mode": "t2v" | "i2v",
     "width": 1280, "height": 704, "num_frames": 121, "fps": 24, "seconds": 5.04,
     "steps": 50, "seed": 42, "elapsed_ms": 512000.0}
    {"id": "...", "status": "failed", "error": "..."}

Job ids live in this replica's memory: keep ``num_workers: 1`` and expect
them to vanish on a restart.
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


# Official Wan2.2 `sample_neg_prompt` (wan/configs/shared_config.py). The
# model was tuned with it; diffusers ships no default, so pass it.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

MAX_WAIT_S = 280.0   # under the predict wrapper's 504 at 290 s (300 s per predictor request)
GRID = 32            # TI2V-5B: VAE 16× spatial × patch 2 → sides must be multiples of 32


class WanVideoGen:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        offload: str = "auto",
        default_steps: int = 50,
        default_frames: int = 121,
        max_frames: int = 121,
        fps: int = 24,
        guidance_scale: float = 5.0,
        max_area: int = 1280 * 704,
        cache_threshold: float = 0.0,
        default_wait_s: float = 20.0,
        max_queued: int = 4,
        keep_results_s: int = 3600,
    ) -> None:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2 needs a CUDA GPU (device: gpu in inference.yaml)")

        self.model_id = model_id
        self.default_steps = int(default_steps)
        self.default_frames = int(default_frames)
        self.max_frames = int(max_frames)
        self.fps = int(fps)
        self.guidance_scale = float(guidance_scale)
        self.max_area = int(max_area)
        self.default_wait_s = float(default_wait_s)
        self.max_queued = int(max_queued)
        self.keep_results_s = int(keep_results_s)

        # The VAE stays fp32 (model card); everything else runs in bf16. The
        # repo stores the transformer in fp32, so loading peaks at ~20 GB RAM.
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        self.t2v = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        # Same weights, second entry point: the image is VAE-encoded into the
        # first latent frame (TI2V-5B has no CLIP image encoder).
        self.i2v = WanImageToVideoPipeline.from_pipe(self.t2v)
        # Without tiling the decode of a 720p clip spikes to ~26 GB.
        vae.enable_tiling()

        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if offload == "auto":
            offload = "off" if total_gb >= 40 else "model"
        if offload == "off":
            self.t2v.to("cuda")
        elif offload == "group":
            self._enable_group_offload()
        elif offload != "model":
            raise ValueError(f"offload must be auto | off | model | group, got {offload!r}")
        self.offload = offload
        self._hooked = None   # pipeline that currently owns the model-offload hooks

        if float(cache_threshold) > 0:
            # First-block cache: skips the rest of the transformer on steps
            # where the first block's output barely moved. 0.05–0.2; higher is
            # faster and blurrier.
            from diffusers import FirstBlockCacheConfig

            self.t2v.transformer.enable_cache(FirstBlockCacheConfig(threshold=float(cache_threshold)))

        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._worker, name="wan-worker", daemon=True).start()

    # ── request handling ────────────────────────────────────────────────────

    async def predict(self, data: bytes) -> dict[str, Any]:
        req = _parse_request(data)
        action = req.get("action", "generate")
        if action == "generate":
            job = self._submit(req)
            wait = req.get("wait_s", self.default_wait_s)
        elif action == "status":
            job = self._jobs.get(str(req.get("id", "")))
            if job is None:
                raise ValueError(
                    "unknown id: finished clips are kept for keep_results_s and ids do not survive a restart"
                )
            wait = req.get("wait_s", 0)
        else:
            raise ValueError(f"action must be 'generate' or 'status', got {action!r}")

        deadline = time.monotonic() + max(0.0, min(float(wait), MAX_WAIT_S))
        while job["status"] in ("queued", "in_progress") and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
        return self._view(job)

    def health(self) -> dict[str, Any]:
        statuses = [j["status"] for j in self._jobs.values()]
        return {
            "status": "ok", "model_id": self.model_id, "offload": self.offload,
            "queued": statuses.count("queued"), "in_progress": statuses.count("in_progress"),
        }

    def _submit(self, req: dict[str, Any]) -> dict[str, Any]:
        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("missing 'prompt' field")

        width, height = _snap_size(
            int(req.get("width", 1280)), int(req.get("height", 704)), self.max_area
        )
        image = None
        if req.get("image_b64"):
            # Keep the image's aspect ratio: crop it to the requested area.
            image = _fit_image(_decode_image(req["image_b64"]), width * height)
            width, height = image.size

        params = {
            "prompt": prompt,
            "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT),
            "image": image,
            "width": width,
            "height": height,
            "num_frames": _snap_frames(int(req.get("num_frames", self.default_frames)), self.max_frames),
            "steps": max(1, min(int(req.get("steps", self.default_steps)), 100)),
            "guidance_scale": float(req.get("guidance_scale", self.guidance_scale)),
            "seed": int(req["seed"]) if req.get("seed") is not None else random.randint(0, 2**31 - 1),
        }
        with self._lock:
            self._prune()
            waiting = sum(1 for j in self._jobs.values() if j["status"] == "queued")
            if waiting >= self.max_queued:
                raise RuntimeError(f"queue full ({waiting} clips waiting), retry later")
            job = {
                "id": uuid.uuid4().hex[:16], "status": "queued", "progress": 0.0,
                "mode": "i2v" if image is not None else "t2v",
                "created": time.time(), "finished": None, "params": params,
                "result": None, "error": None,
            }
            self._jobs[job["id"]] = job
        self._queue.put(job)
        return job

    def _view(self, job: dict[str, Any]) -> dict[str, Any]:
        view = {"id": job["id"], "status": job["status"], "mode": job["mode"]}
        if job["status"] == "queued":
            with self._lock:
                ahead = [j for j in self._jobs.values()
                         if j["status"] == "queued" and j["created"] < job["created"]]
            view["queue_position"] = len(ahead)
        elif job["status"] == "in_progress":
            view["progress"] = job["progress"]
        elif job["status"] == "completed":
            view.update(job["result"])
        else:
            view["error"] = job["error"]
        return view

    def _prune(self) -> None:
        cutoff = time.time() - self.keep_results_s
        for job_id in [k for k, j in self._jobs.items() if j["finished"] and j["finished"] < cutoff]:
            del self._jobs[job_id]

    # ── generation (background thread, one clip at a time) ──────────────────

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            job["status"] = "in_progress"
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    job["result"] = self._generate(job)
                job["result"]["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                job["status"] = "completed"
            except torch.cuda.OutOfMemoryError:
                job["error"] = (
                    "CUDA out of memory: lower num_frames or the size, or set "
                    "offload: \"group\" in inference.yaml"
                )
                job["status"] = "failed"
            except Exception as exc:  # report, keep the worker alive
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["status"] = "failed"
            finally:
                job["finished"] = time.time()
                job["params"]["image"] = None
                torch.cuda.empty_cache()

    def _generate(self, job: dict[str, Any]) -> dict[str, Any]:
        p = job["params"]

        def on_step(pipe, step, timestep, callback_kwargs):
            job["progress"] = round((step + 1) / p["steps"], 3)
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
            pipe = self.i2v
            kwargs["image"] = p["image"]
        else:
            pipe = self.t2v
        self._use(pipe)
        frames = pipe(**kwargs).frames[0]

        return {
            "video_b64": base64.b64encode(_encode_mp4(frames, self.fps)).decode("ascii"),
            "width": p["width"], "height": p["height"], "num_frames": p["num_frames"],
            "fps": self.fps, "seconds": round(p["num_frames"] / self.fps, 2),
            "steps": p["steps"], "guidance_scale": p["guidance_scale"], "seed": p["seed"],
            "model_id": self.model_id,
        }

    def _use(self, pipe) -> None:
        """Give the model-offload hooks to the pipeline about to run.

        Both pipelines share one set of modules, but each installs its own
        offload hooks in its own order. Re-installing on a switch (cheap: the
        weights are already in RAM) keeps one consistent set.
        """
        if self.offload == "model" and self._hooked is not pipe:
            pipe.enable_model_cpu_offload()
            self._hooked = pipe

    def _enable_group_offload(self) -> None:
        """Stream weights layer by layer: lowest VRAM, slower; for when 16 GB OOMs."""
        from diffusers.hooks import apply_group_offloading

        gpu, cpu = torch.device("cuda"), torch.device("cpu")
        self.t2v.transformer.enable_group_offload(
            onload_device=gpu, offload_device=cpu, offload_type="leaf_level", use_stream=True
        )
        apply_group_offloading(
            self.t2v.text_encoder, onload_device=gpu, offload_device=cpu,
            offload_type="block_level", num_blocks_per_group=2,
        )
        apply_group_offloading(
            self.t2v.vae, onload_device=gpu, offload_device=cpu, offload_type="leaf_level"
        )


def _parse_request(data: bytes) -> dict[str, Any]:
    try:
        req = json.loads(data.decode("utf-8"))
        if isinstance(req, dict):
            return req
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {"prompt": data.decode("utf-8", errors="replace")}


def _decode_image(b64: str):
    from PIL import Image

    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _snap_size(width: int, height: int, max_area: int) -> tuple[int, int]:
    """Floor to the 32-px grid; if still over max_area, scale down keeping the aspect."""
    width, height = max(256, width // GRID * GRID), max(256, height // GRID * GRID)
    if width * height > max_area:
        scale = (max_area / (width * height)) ** 0.5
        width, height = max(256, int(width * scale) // GRID * GRID), max(256, int(height * scale) // GRID * GRID)
    return width, height


def _fit_image(image, area: int):
    """Wan-style input sizing: keep the image's aspect, resize, center-crop to the grid."""
    from PIL import Image

    w, h = image.size
    ratio = w / h
    out_w = max(256, int((area * ratio) ** 0.5) // GRID * GRID)
    out_h = max(256, int((area / ratio) ** 0.5) // GRID * GRID)
    scale = max(out_w / w, out_h / h)
    image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    left, top = (image.width - out_w) // 2, (image.height - out_h) // 2
    return image.crop((left, top, left + out_w, top + out_h))


def _snap_frames(frames: int, max_frames: int) -> int:
    """Wan needs 4k+1 frames."""
    frames = max(5, min(frames, max_frames))
    return (frames - 1) // 4 * 4 + 1


def _encode_mp4(frames, fps: int) -> bytes:
    """H.264 MP4 via imageio-ffmpeg (browsers play it; OpenCV's mp4v they do not)."""
    from diffusers.utils import export_to_video

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        export_to_video(frames, path, fps=fps, quality=8)
        with open(path, "rb") as f:
            return f.read()
