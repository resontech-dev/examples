"""
Drive the media studio: the chat model plans, then calls the image and video generators.

One endpoint, three routes (see scripts/serve_module.py):
    "assistant" → POST /v1/chat/completions   Qwen3-8B, tool calling, thinking on
    images      → POST /v1/images/generations  Z-Image-Turbo
    videos      → POST /v1/videos, GET /v1/videos/{id}, GET /v1/videos/{id}/content  Wan2.2-TI2V-5B

The default run is an agent loop: the brief goes to the assistant, which
has two tools, generate_image and generate_video. It usually makes a
keyframe first and then animates it. This script executes the tool calls
against the same endpoint, saves the files to out/, and hands the results
back until the assistant answers in text.

Run
---
    pip install openai python-dotenv httpx
    python predict.py
    python predict.py "Тизер на 5 секунд для відкриття кав'ярні у Львові: ранкове світло, пара над чашкою"
    python predict.py --no-think "Постер для вебінару про AI у банкінгу, без тексту"
    python predict.py --image-only "isometric game asset, a watermill by a river"
    python predict.py --video-only "a paper boat drifts down a rainy street, low tracking shot"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
load_dotenv(HERE / ".env")

DEFAULT_BRIEF = (
    "Зроби 5-секундний тизер для відкриття кав'ярні у Львові: затишний інтер'єр, "
    "ранкове світло з вікна, пара над чашкою кави."
)
SYSTEM = """You are a creative director with two tools:
- generate_image: one still from a text prompt (Z-Image-Turbo).
- generate_video: a silent clip of up to 5 seconds (Wan2.2), from a text prompt,
  optionally starting from an image you generated (pass its image_id).

For a video brief: first generate a keyframe with generate_image, then animate it
with generate_video(image_id=...). Write generator prompts in English: concrete
subject, setting, light, lens and style. For video prompts describe the motion and
the camera move, not the scene alone. Never put readable text in a video prompt.
When the tools have returned, answer the user briefly, in the user's language:
what you made and the file names."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate one still image. Returns an image_id usable in generate_video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed visual prompt, in English"},
                    "size": {"type": "string", "enum": ["1024x1024", "1280x720", "720x1280"],
                             "description": "1280x720 for a landscape video keyframe"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "Generate a silent clip of 1-5 seconds at 24 fps. Takes minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Motion and camera move, in English"},
                    "image_id": {"type": "string", "description": "Start from this generated image"},
                    "seconds": {"type": "number", "minimum": 1, "maximum": 5},
                },
                "required": ["prompt"],
            },
        },
    },
]


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _base_url() -> str:
    """Normalize the platform endpoint to an OpenAI base_url (…/v1)."""
    base = _required("RESON_INFERENCE_URL").rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


class Studio:
    def __init__(self) -> None:
        key = _required("RESON_INFERENCE_API_KEY")
        headers = {"Authorization": f"Bearer {key}", "X-API-Key": key}
        self.base = _base_url()
        self.llm = OpenAI(base_url=self.base, api_key=key, default_headers={"X-API-Key": key},
                          timeout=float(os.getenv("PREDICT_TIMEOUT", "300")))
        self.http = httpx.Client(base_url=self.base, headers=headers, timeout=300.0)
        self.images: dict[str, Path] = {}
        OUT.mkdir(exist_ok=True)

    # ── tools ───────────────────────────────────────────────────────────────

    def generate_image(self, prompt: str, size: str = "1024x1024") -> dict:
        print(f"[image] {size}: {prompt!r}")
        body = {"model": "z-image-turbo", "prompt": prompt, "size": size, "response_format": "b64_json"}
        for attempt in range(5):
            resp = self.http.post("/images/generations", json=body)
            if resp.status_code != 503:          # 503 = the GPU is busy with a clip
                break
            print("[image] GPU busy with a clip, retrying in 30 s")
            time.sleep(30)
        _raise_for(resp)
        result = resp.json()
        image_id = f"img_{len(self.images) + 1}"
        path = OUT / f"studio_{int(time.time())}_{image_id}.png"
        path.write_bytes(base64.b64decode(result["data"][0]["b64_json"]))
        self.images[image_id] = path
        print(f"[image] → {path.relative_to(HERE)} (seed {result.get('seed')})")
        return {"image_id": image_id, "file": path.name, "size": result.get("size")}

    def generate_video(self, prompt: str, image_id: str | None = None, seconds: float = 5) -> dict:
        body = {"model": "wan2.2-ti2v-5b", "prompt": prompt, "seconds": str(seconds)}
        if image_id:
            if image_id not in self.images:
                return {"error": f"unknown image_id {image_id!r}; known: {sorted(self.images)}"}
            body["input_reference_b64"] = base64.b64encode(self.images[image_id].read_bytes()).decode("ascii")
        print(f"[video] {seconds} s{' from ' + image_id if image_id else ''}: {prompt!r}")
        resp = self.http.post("/videos", json=body)
        _raise_for(resp)
        video = resp.json()
        while video["status"] in ("queued", "in_progress"):
            time.sleep(10)
            resp = self.http.get(f"/videos/{video['id']}")
            _raise_for(resp)
            video = resp.json()
            print(f"[video] {video['id']}: {video['status']} {video.get('progress', 0)} %")
        if video["status"] != "completed":
            return {"error": (video.get("error") or {}).get("message", video["status"])}
        resp = self.http.get(f"/videos/{video['id']}/content")
        _raise_for(resp)
        path = OUT / f"studio_{video['id']}.mp4"
        path.write_bytes(resp.content)
        print(f"[video] → {path.relative_to(HERE)} ({video['size']}, {video['seconds']} s, "
              f"rendered in {video.get('elapsed_s')} s)")
        return {"video_id": video["id"], "file": path.name, "seconds": video["seconds"], "size": video["size"]}

    # ── the agent loop ──────────────────────────────────────────────────────

    def run(self, brief: str, think: bool, max_turns: int = 6) -> None:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": brief}]
        for _ in range(max_turns):
            resp = self.llm.chat.completions.create(
                model="assistant",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.6,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            msg = resp.choices[0].message
            reasoning = (msg.model_extra or {}).get("reasoning") or (msg.model_extra or {}).get("reasoning_content")
            if reasoning:
                print(f"[think] {reasoning.strip()[:400]}{'…' if len(reasoning) > 400 else ''}")
            if not msg.tool_calls:
                print(f"\n[assistant] {msg.content}")
                return
            # Qwen3 guidance: keep the thinking out of the history.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    tool = {"generate_image": self.generate_image,
                            "generate_video": self.generate_video}[call.function.name]
                    result = tool(**args)
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    result = {"error": f"bad tool call {call.function.name}: {exc}"}
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        print("[predict] stopped after max_turns without a final answer")


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            message = resp.json().get("error", {}).get("message", resp.text)
        except ValueError:
            message = resp.text
        sys.exit(f"[predict] {resp.request.method} {resp.request.url.path} → {resp.status_code}: {message}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brief", nargs="?", default=DEFAULT_BRIEF)
    ap.add_argument("--no-think", action="store_true", help="turn Qwen3 thinking off (faster planning)")
    ap.add_argument("--image-only", action="store_true", help="skip the assistant: one image from the prompt")
    ap.add_argument("--video-only", action="store_true", help="skip the assistant: one text-to-video clip")
    args = ap.parse_args()

    studio = Studio()
    try:
        if args.image_only:
            studio.generate_image(args.brief, "1280x720")
        elif args.video_only:
            studio.generate_video(args.brief)
        else:
            studio.run(args.brief, think=not args.no_think)
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {studio.base}: deployment not RUNNING yet, or URL wrong: {exc}")
    except APIStatusError as exc:
        sys.exit(f"[predict] {exc.status_code} from the chat route: {exc.message}")


if __name__ == "__main__":
    main()
