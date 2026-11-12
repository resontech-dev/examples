"""
Moderate text (and optionally an image or a response) with the deployed
Nemotron 3.5 Content Safety endpoint (OpenAI protocol).

The guard is a generative classifier. You send the content as a normal chat
and select the mode with `chat_template_kwargs` (keys verbatim from the model
card): `request_categories="/categories"` always, `enable_thinking` for a
reasoning trace, `custom_policy` to replace NVIDIA's taxonomy with your text.

Run
---
    pip install openai python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                                    # benign default prompt, taxonomy mode
    python predict.py "How do I pick the lock on my neighbour's door?"
    python predict.py "…" --response "Here is how: …"    # moderate prompt + assistant response
    python predict.py "…" --policy sample_data/policy.txt   # your written policy, no reasoning
    python predict.py "…" --policy sample_data/policy.txt --think   # same, with the trace
    python predict.py "What is this document?" --image sample_data/invoice.png
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_PROMPT = "How do I reset the password for our corporate VPN account?"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _openai_base_url(raw: str) -> str:
    """Normalize the platform endpoint to an OpenAI base_url (…/v1)."""
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def _client() -> OpenAI:
    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    return OpenAI(
        base_url=url,
        api_key=key,                             # → Authorization: Bearer (vLLM layer)
        default_headers={"X-API-Key": key},      # → platform serve-proxy layer
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


def _messages(prompt: str, image: str | None, response: str | None) -> list[dict]:
    """Message layout from the model card: image part first, then the text;
    the assistant response, when moderated, is a second message."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image:
        content = [_image_part(image), *content]
    messages = [{"role": "user", "content": content}]
    if response:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def parse_verdict(raw: str) -> dict:
    """Lenient parser for the card's output format.

    Returns {"user": "safe"|"unsafe"|None, "response": …, "categories": [..],
    "reasoning": str}. Unknown shapes leave the fields None so the caller can
    decide how strict to be (treat None as unsafe in production).
    """
    reasoning = ""
    m = THINK_RE.search(raw)
    if m:
        reasoning = m.group(0)[len("<think>"):-len("</think>")].strip()
    body = THINK_RE.sub("", raw)

    def _grab(label: str) -> str | None:
        mm = re.search(rf"{label}\s*:\s*(safe|unsafe)", body, re.IGNORECASE)
        return mm.group(1).lower() if mm else None

    cats = re.search(r"Safety Categories\s*:\s*(.+)", body, re.IGNORECASE)
    categories = [c.strip() for c in cats.group(1).split(",") if c.strip()] if cats else []
    return {
        "user": _grab("User Safety"),
        "response": _grab("Response Safety"),
        "categories": categories,
        "reasoning": reasoning,
    }


def moderate(client: OpenAI, model: str, prompt: str, *, image: str | None = None,
             response: str | None = None, policy: str | None = None,
             think: bool = False) -> tuple[dict, str]:
    # Keys verbatim from the model card ("Use it with vLLM").
    template_kwargs: dict = {"request_categories": "/categories", "enable_thinking": think}
    if policy:
        template_kwargs["custom_policy"] = policy
    resp = client.chat.completions.create(
        model=model,
        messages=_messages(prompt, image, response),
        max_tokens=600 if think else 100,        # the card uses 100 / 500
        temperature=0.0,
        extra_body={"chat_template_kwargs": template_kwargs},
    )
    raw = (resp.choices[0].message.content or "").strip()
    # If the pinned image was deployed with a reasoning parser, the trace
    # arrives in `reasoning` instead of inside `content`.
    extra = getattr(resp.choices[0].message, "reasoning", None)
    if extra and "<think>" not in raw:
        raw = f"<think>{extra}</think>{raw}"
    return parse_verdict(raw), raw


def _take(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    i = argv.index(flag)
    try:
        value = argv[i + 1]
    except IndexError:
        sys.exit(f"[predict] {flag} needs a value")
    del argv[i:i + 2]
    return value


def main() -> None:
    argv = sys.argv[1:]
    think = "--think" in argv
    argv = [a for a in argv if a != "--think"]
    response = _take(argv, "--response")
    image = _take(argv, "--image")
    policy_path = _take(argv, "--policy")
    policy = Path(policy_path).read_text() if policy_path else None
    positional = [a for a in argv if not a.startswith("--")]
    prompt = positional[0] if positional else DEFAULT_PROMPT

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list: replica still loading weights?")
    model = os.getenv("MODEL_ID", "guard" if "guard" in models else models[0])
    print(f"[predict] models:   {models} → using {model!r}")
    print(f"[predict] mode:     {'custom policy' if policy else 'standard taxonomy'}"
          f"{', thinking on' if think else ''}{', with image' if image else ''}"
          f"{', with response' if response else ''}")
    print(f"[predict] prompt:   {prompt}")
    if response:
        print(f"[predict] response: {response[:120]}{'…' if len(response) > 120 else ''}")
    print("─" * 72)

    verdict, raw = moderate(client, model, prompt, image=image, response=response,
                            policy=policy, think=think)
    print(raw)
    print("─" * 72)
    print(f"[predict] user: {verdict['user']}  response: {verdict['response']}  "
          f"categories: {verdict['categories'] or '-'}")
    if verdict["user"] is None:
        print("[predict] ⚠️ no 'User Safety' line found: treat as unsafe in production "
              "and check the raw output above against the model card format.")
    safe = verdict["user"] == "safe" and verdict["response"] in (None, "safe")
    print("[predict] ✅ passes" if safe else "[predict] ✋ flagged")


if __name__ == "__main__":
    main()
