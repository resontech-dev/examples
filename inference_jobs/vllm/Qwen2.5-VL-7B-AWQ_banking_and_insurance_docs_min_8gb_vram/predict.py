"""
Smoke-test the BFSI vision endpoint (Stack A) — image in, JSON out.

The real client is sectors/banking_and_insurance/bfsi_pilot/ingest.py;
this script just proves the endpoint extracts a document.

Run
---
    pip install openai python-dotenv pillow pillow-heif
    python predict.py                            # sample invoice → JSON
    python predict.py ./scan.jpg
    python predict.py ./IMG_1234.HEIC           # iPhone photos auto-convert to JPEG
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_IMAGE = HERE / "sample_data" / "invoice.png"
PROMPT = (
    "Це рахунок за ремонт. Поверни ЛИШЕ JSON: {vendor_name, invoice_number, "
    "invoice_date, positions:[{name, qty, unit_price_uah}], total_uah}. "
    "Нечитабельне поле → null. Не вигадуй значення."
)
# PROMPT = (
#     "Що ти бачиш на цьому документі?"
# )

# Formats the server can decode (vLLM loads images via PIL inside the ray-llm
# image, which has no HEIC/HEIF support — raw HEIC 400s with "cannot identify
# image file"). Anything not listed here is converted to JPEG client-side.
_SERVER_OK = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_MAX_SIDE = 2048  # plenty for document VL models; keeps the payload small


def _prepare_image(path: Path) -> tuple[bytes, str]:
    """Bytes + mime ready for the data URL; converts HEIC/etc. to JPEG."""
    if path.suffix.lower() in _SERVER_OK:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return path.read_bytes(), mime

    try:
        from PIL import Image
    except ImportError:
        sys.exit(f"[predict] {path.suffix} input needs conversion: pip install pillow pillow-heif")
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            sys.exit("[predict] HEIC input needs: pip install pillow-heif")

    import io
    img = Image.open(path).convert("RGB")
    img.thumbnail((_MAX_SIDE, _MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    data = buf.getvalue()
    print(f"[predict] converted {path.suffix} → JPEG {img.size[0]}x{img.size[1]} ({len(data) // 1024} KB)")
    return data, "image/jpeg"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _openai_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def main() -> None:
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image.is_file():
        sys.exit(f"[predict] image not found: {image}")

    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    client = OpenAI(base_url=url, api_key=key,
                    default_headers={"X-API-Key": key},
                    timeout=float(os.getenv("PREDICT_TIMEOUT", "300")))

    try:
        models = [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {url} — deployment not RUNNING yet? {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(f"[predict] {exc.status_code} — check RESON_INFERENCE_API_KEY")
        raise
    model = os.getenv("MODEL_ID", models[0] if models else "vision")
    print(f"[predict] models: {models} → using {model!r}")
    print(f"[predict] image:  {image}")
    print("─" * 72)

    data, mime = _prepare_image(image)
    b64 = base64.b64encode(data).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        max_tokens=768,
        temperature=0.0,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
