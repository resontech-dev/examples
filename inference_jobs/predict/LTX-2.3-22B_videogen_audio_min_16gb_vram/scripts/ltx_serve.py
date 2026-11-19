"""LTX-2.3 (22B, Lightricks): text or image in, MP4 with sound out (predictor mode).

Video and a matching audio track (ambience, effects, speech) come from one
model in one pass. This runs Lightricks' own `ltx-pipelines` package and
its DistilledPipeline: 8 steps at half resolution, a 2× spatial upsample,
then 4 refinement steps at full resolution, no CFG. 1280×704, 24 fps,
121 frames = 5 s by default.

⚠️ LICENSE: the weights are under the LTX-2 Community License, not
Apache. Companies with ≥ $10M annual revenue need a paid license, and
Attachment A #20 forbids use in a product that competes with Lightricks'
own without a separate license. Read LICENSE-2 in the model repo before
selling an endpoint backed by this job.

Request (JSON via ``client.predict_json``):

    {"prompt": "rain on a tin roof at night, a dog barks in the distance, slow dolly in",
     "image_b64": "<PNG or JPEG, base64>",   # optional: this image becomes frame 0
     "image_strength": 1.0,                   # optional; lower lets the model drift from it
     "width": 1280, "height": 704,            # optional; multiples of 64, area capped at max_area
     "num_frames": 121,                       # optional; 8k+1: 97 ≈ 4 s, 121 ≈ 5 s, 241 ≈ 10 s
     "seed": 42,                              # optional
     "wait_s": 20}                            # optional: how long this call waits for the clip (max 280)

A bare text body is treated as {"prompt": <body>}.

Like the Wan job, a clip takes minutes and one request to a predictor job
is cut off after 300 s (set on the Ray head node; the predict wrapper
answers 504 at 290 s), so the clip renders in a background thread. Every call returns
within ``wait_s`` with the finished clip or a job to poll:

    {"action": "status", "id": "<id>", "wait_s": 20}

Responses:

    {"id": "...", "status": "queued", "queue_position": 0}
    {"id": "...", "status": "in_progress", "running_s": 95}
    {"id": "...", "status": "completed", "video_b64": "<base64 MP4, H.264 + AAC>", "mode": "t2v" | "i2v",
     "width": 1280, "height": 704, "num_frames": 121, "fps": 24.0, "seconds": 5.04,
     "seed": 42, "elapsed_ms": 240000.0}
    {"id": "...", "status": "failed", "error": "..."}

Job ids live in this replica's memory: keep ``num_workers: 1``.
"""
from __future__ import annotations

import os

# Lightricks' recommendation for this model; must be set before CUDA starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402


MAX_WAIT_S = 280.0   # under the predict wrapper's 504 at 290 s (300 s per predictor request)
GRID = 64            # two-stage pipeline: the final width/height must be multiples of 64


class LtxVideoGen:
    def __init__(
        self,
        model_repo: str = "Lightricks/LTX-2.3",
        checkpoint: str = "ltx-2.3-22b-distilled-1.1.safetensors",
        upsampler_repo: str = "Lightricks/LTX-2.3",
        spatial_upsampler: str = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        gemma_repo: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
        quantization: str | None = "fp8-cast",
        offload: str = "auto",
        default_frames: int = 121,
        max_frames: int = 241,
        fps: float = 24.0,
        max_area: int = 1280 * 704,
        default_wait_s: float = 20.0,
        max_queued: int = 4,
        keep_results_s: int = 3600,
        hf_token: str | None = None,
    ) -> None:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.errors import GatedRepoError
        from ltx_pipelines import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.utils.types import OffloadMode

        if not torch.cuda.is_available():
            raise RuntimeError("LTX-2.3 needs a CUDA GPU (device: gpu in inference.yaml)")

        self.checkpoint = checkpoint
        self.default_frames = int(default_frames)
        self.max_frames = int(max_frames)
        self.fps = float(fps)
        self.max_area = int(max_area)
        self.default_wait_s = float(default_wait_s)
        self.max_queued = int(max_queued)
        self.keep_results_s = int(keep_results_s)

        # The distilled checkpoint bundles transformer, video VAE, audio VAE
        # and text projection in one 46 GB file; the Gemma 3 12B text encoder
        # (24.4 GB, gated by Google) is a separate HF directory.
        ckpt = hf_hub_download(model_repo, checkpoint)
        upsampler = hf_hub_download(upsampler_repo, spatial_upsampler)
        token = hf_token or os.environ.get("HF_TOKEN") or None
        try:
            gemma_root = snapshot_download(gemma_repo, token=token)
        except GatedRepoError as exc:
            raise RuntimeError(
                f"{gemma_repo} is gated: accept the Gemma terms on its HF page with the account "
                "behind HF_TOKEN, put the token in .env, and submit again"
            ) from exc

        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if offload == "auto":
            # Resident needs the whole transformer on the card (22 GB in FP8,
            # 44 GB in bf16) plus activations; below that, stream layers from
            # pinned RAM (~5 GB VRAM for the weights' buffer).
            resident_gb = 48 if quantization else 80
            offload = "none" if total_gb >= resident_gb else "cpu"
        self.offload = offload

        policy = QuantizationKind(quantization).to_policy(ckpt) if quantization else None
        self.quantization = quantization
        self.pipe = DistilledPipeline(
            model_paths=ModelPaths.from_monolith(ckpt, gemma_root),
            spatial_upsampler_path=upsampler,
            loras=[],
            quantization=policy,
            offload_mode=OffloadMode(offload),
        )

        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._worker, name="ltx-worker", daemon=True).start()

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
            "status": "ok", "checkpoint": self.checkpoint, "quantization": self.quantization,
            "offload": self.offload,
            "queued": statuses.count("queued"), "in_progress": statuses.count("in_progress"),
        }

    def _submit(self, req: dict[str, Any]) -> dict[str, Any]:
        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("missing 'prompt' field")

        image = _decode_image(req["image_b64"]) if req.get("image_b64") else None
        if image is not None and "width" not in req and "height" not in req:
            # No size given: follow the image's aspect ratio at the default area.
            width, height = _size_for_aspect(image.width / image.height, self.max_area)
        else:
            width, height = _snap_size(
                int(req.get("width", 1280)), int(req.get("height", 704)), self.max_area
            )

        params = {
            "prompt": prompt,
            "image": image,
            "image_strength": max(0.0, min(float(req.get("image_strength", 1.0)), 1.0)),
            "width": width,
            "height": height,
            "num_frames": _snap_frames(int(req.get("num_frames", self.default_frames)), self.max_frames),
            "seed": int(req["seed"]) if req.get("seed") is not None else random.randint(0, 2**31 - 1),
        }
        with self._lock:
            self._prune()
            waiting = sum(1 for j in self._jobs.values() if j["status"] == "queued")
            if waiting >= self.max_queued:
                raise RuntimeError(f"queue full ({waiting} clips waiting), retry later")
            job = {
                "id": uuid.uuid4().hex[:16], "status": "queued",
                "mode": "i2v" if image is not None else "t2v",
                "created": time.time(), "started": None, "finished": None,
                "params": params, "result": None, "error": None,
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
            view["running_s"] = round(time.time() - job["started"])
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
            job["started"] = time.time()
            t0 = time.perf_counter()
            try:
                # Required: without inference_mode the text encoder's outputs
                # keep its activations alive and memory blows up.
                with torch.inference_mode():
                    job["result"] = self._generate(job)
                job["result"]["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                job["status"] = "completed"
            except torch.cuda.OutOfMemoryError:
                job["error"] = (
                    "CUDA out of memory: lower num_frames or the size, or set offload: \"cpu\" "
                    "(\"disk\" if RAM is short) in inference.yaml"
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
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.types import ImageConditioningInput

        p = job["params"]
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            if p["image"] is not None:
                # The pipeline loads stills from paths, then resizes and
                # center-crops them to width × height itself.
                still = os.path.join(tmp, "first_frame.png")
                p["image"].save(still)
                images = [ImageConditioningInput.replace(still, frame_idx=0, strength=p["image_strength"])]

            out = self.pipe(
                prompt=p["prompt"],
                seed=p["seed"],
                height=p["height"],
                width=p["width"],
                frame_rate=self.fps,
                images=images,
                num_frames=p["num_frames"],
            )
            # out.video is a lazy iterator of decoded chunks; encode_video
            # drains it and muxes the stereo track as AAC.
            path = os.path.join(tmp, "clip.mp4")
            encode_video(out.video, self.fps, out.audio, path, video_chunks_number=None)
            with open(path, "rb") as f:
                mp4 = f.read()

        return {
            "video_b64": base64.b64encode(mp4).decode("ascii"),
            "width": p["width"], "height": p["height"], "num_frames": p["num_frames"],
            "fps": self.fps, "seconds": round(p["num_frames"] / self.fps, 2),
            "seed": p["seed"], "checkpoint": self.checkpoint,
        }


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
    """Floor to the 64-px grid; if still over max_area, scale down keeping the aspect."""
    width, height = max(256, width // GRID * GRID), max(256, height // GRID * GRID)
    if width * height > max_area:
        scale = (max_area / (width * height)) ** 0.5
        width, height = max(256, int(width * scale) // GRID * GRID), max(256, int(height * scale) // GRID * GRID)
    return width, height


def _size_for_aspect(ratio: float, area: int) -> tuple[int, int]:
    return (max(256, int((area * ratio) ** 0.5) // GRID * GRID),
            max(256, int((area / ratio) ** 0.5) // GRID * GRID))


def _snap_frames(frames: int, max_frames: int) -> int:
    """LTX needs 8k+1 frames."""
    frames = max(9, min(frames, max_frames))
    return (frames - 1) // 8 * 8 + 1
