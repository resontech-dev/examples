"""
Smoke-test Gemma 4 26B A4B (QAT, AWQ INT4): text, image, tool calling, thinking.

BLOCKED on the pinned image (vLLM 0.18.0): run this after the job boots on
an image with the vLLM version named in README.md.

Run
---
    pip install openai python-dotenv
    python predict.py                                  # English business reply, thinking off
    python predict.py --uk                             # Ukrainian business reply
    python predict.py --think "Plan a 3-step migration from Gemma 3 to Gemma 4"
    python predict.py --image                          # sample_data/invoice.png -> JSON
    python predict.py --tools                          # tool-calling round trip
    python predict.py --stream "..."                   # token stream
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

SYSTEM_EN = "You are a concise, professional assistant. Do not invent facts."
SYSTEM_UK = "Ти помічник працівника установи. Відповідай українською, стисло, діловим стилем, без вигаданих фактів."
PROMPT_EN = (
    "A customer writes: 'I was charged twice for invoice 1042.' Draft a short, polite reply: the duplicate charge will be refunded within 3 business days."
)
PROMPT_UK = (
    "Підготуй відповідь на звернення мешканця: він просить пояснити, коли буде відремонтовано освітлення на вулиці Шевченка. Ремонт запланований на наступний тиждень, виконавець: КП «Благоустрій»."
)
DEFAULT_IMAGE = HERE / "sample_data" / "invoice.png"
DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"
IMAGE_PROMPT = (
    "Extract from this invoice: vendor, invoice_number, date, currency, line_items (description, qty, unit_price, amount) and total. Reply with JSON only."
)
ASR_PROMPT = (
    "Transcribe the following speech segment in English into English text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits."
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
AUDIO_SUPPORTED = False
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Return the status of an order by its number.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order number"}},
            "required": ["order_id"],
        },
    },
}]


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _openai_base_url(raw: str) -> str:
    """Normalize the platform endpoint to an OpenAI base_url (.../v1)."""
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def _client() -> OpenAI:
    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    return OpenAI(
        base_url=url,
        api_key=key,                             # Authorization: Bearer (vLLM layer)
        default_headers={"X-API-Key": key},      # platform serve-proxy layer
        timeout=float(os.getenv("PREDICT_TIMEOUT", "300")),
    )


def _list_models(client: OpenAI) -> list[str]:
    url = str(client.base_url)
    try:
        return [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {url}: deployment not RUNNING yet, or URL wrong: {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(
                f"[predict] {exc.status_code} from {url}: check RESON_INFERENCE_API_KEY "
                "(the job-scoped predict key from submit.py, not the platform rsk_ key)."
            )
        if exc.status_code == 404:
            sys.exit(
                f"[predict] 404 at {url}/models: endpoint is up but has no OpenAI "
                "surface. Was the job submitted with engine: vllm_openai?"
            )
        raise


def _image_part(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _take_path_flag(argv: list[str], flag: str, suffixes: set[str], default: Path) -> tuple[str | None, list[str]]:
    """Pop `--flag [path]` from argv. Returns (path or None, remaining argv)."""
    if flag not in argv:
        return None, argv
    i = argv.index(flag)
    nxt = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
    is_path = bool(nxt) and (nxt.startswith("http") or Path(nxt).suffix.lower() in suffixes)
    value = nxt if is_path else str(default)
    rest = [a for j, a in enumerate(argv) if j != i and not (is_path and j == i + 1)]
    return value, rest


def _audio_part(source: str) -> dict:
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] audio file not found: {path}")
    fmt = path.suffix.lower().lstrip(".") or "wav"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}


def _extra(think: bool) -> dict:
    """Per-request knobs. Card sampling: temperature 1.0, top_p 0.95, top_k 64.

    Thinking is switched per request through the chat template; the server sets
    no default (Ray 2.55.1 breaks `default_chat_template_kwargs`, and the newer
    image this job waits for should be checked before relying on it).
    """
    return {"extra_body": {"top_k": 64, "chat_template_kwargs": {"enable_thinking": think}}}


def _sampling() -> dict:
    return {
        "temperature": float(os.getenv("TEMPERATURE", "1.0")),
        "top_p": float(os.getenv("TOP_P", "0.95")),
    }


def _print(resp, started: float) -> None:
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        print(f"[assistant] reasoning: {len(reasoning)} chars (thinking was on)")
    print(msg.content)
    if resp.usage:
        print("-" * 72)
        print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion, {time.perf_counter() - started:.1f} s")


def _chat(client: OpenAI, model: str, messages: list, think: bool, stream: bool, max_tokens: int) -> None:
    started = time.perf_counter()
    if stream:
        chunks = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            **_sampling(), **_extra(think),
        )
        for chunk in chunks:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                print(delta, end="", flush=True)
        print()
        return
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, **_sampling(), **_extra(think),
    )
    _print(resp, started)


def run_tools(client: OpenAI, model: str, think: bool) -> None:
    messages = [{"role": "system", "content": SYSTEM_EN},
                {"role": "user", "content": "What is the status of order 10452?"}]
    first = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto",
        max_tokens=512, **_sampling(), **_extra(think),
    )
    msg = first.choices[0].message
    call = (msg.tool_calls or [None])[0]
    if call is None:
        sys.exit(f"[tools] no tool call: the model answered with text: {msg.content!r}")
    print(f"[tools] call: {call.function.name}({call.function.arguments})")
    result = '{"order_id": "10452", "status": "in transit", "eta": "tomorrow before 14:00"}'
    messages += [
        {"role": "assistant", "content": msg.content or "",
         "tool_calls": [{"id": call.id, "type": "function",
                         "function": {"name": call.function.name,
                                      "arguments": call.function.arguments}}]},
        {"role": "tool", "tool_call_id": call.id, "content": result},
    ]
    second = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, max_tokens=512, **_sampling(), **_extra(think),
    )
    print(f"[tools] answer: {second.choices[0].message.content}")


def main() -> None:
    argv = sys.argv[1:]
    stream = "--stream" in argv
    think = "--think" in argv
    ukrainian = "--uk" in argv
    tools = "--tools" in argv
    image, argv = _take_path_flag(argv, "--image", IMAGE_SUFFIXES, DEFAULT_IMAGE)
    audio, argv = _take_path_flag(argv, "--audio", AUDIO_SUFFIXES, DEFAULT_AUDIO)
    positional = [a for a in argv if not a.startswith("--")]
    if audio and not AUDIO_SUPPORTED:
        sys.exit("[predict] this Gemma 4 size takes no audio (audio: E2B, E4B and 12B only)")

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list: replica still loading weights?")
    model = os.getenv("MODEL_ID", "assistant" if "assistant" in models else models[0])
    print(f"[predict] models: {models} -> using {model!r}, thinking {'on' if think else 'off'}")
    print("-" * 72)

    if tools:
        run_tools(client, model, think)
    elif image:
        print(f"[predict] image: {image}")
        prompt = positional[0] if positional else IMAGE_PROMPT
        # Gemma 4 card: image content before the text.
        messages = [{"role": "user", "content": [_image_part(image), {"type": "text", "text": prompt}]}]
        _chat(client, model, messages, think, stream, int(os.getenv("MAX_TOKENS", "1024")))
    elif audio:
        print(f"[predict] audio: {audio} (30 s maximum per clip)")
        prompt = positional[0] if positional else ASR_PROMPT
        # Gemma 4 card: audio content after the text.
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, _audio_part(audio)]}]
        _chat(client, model, messages, think, stream, int(os.getenv("MAX_TOKENS", "512")))
    else:
        system = SYSTEM_UK if ukrainian else SYSTEM_EN
        prompt = positional[0] if positional else (PROMPT_UK if ukrainian else PROMPT_EN)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        default_max = "4096" if think else "512"
        _chat(client, model, messages, think, stream, int(os.getenv("MAX_TOKENS", default_max)))


if __name__ == "__main__":
    main()
