"""BFSI pilot — Module 3: ingest worker (one file: worker + schemas + parser).

Drains `documents` with status='pending' one at a time
(FOR UPDATE SKIP LOCKED), branching by doc_type:

  * scans (repair_invoice / damage_photo_act / police_report) → Stack A
    (VL_BASE_URL): page image + schema prompt → JSON → pydantic validation
    (one retry with the validation errors) → extracted/failed + linking;
  * policy_pdf → no model needed: text extraction (pypdfium2) + section
    chunking by the `^\\d+(\\.\\d+)*` numbering regex → doc_chunks
    (embedding NULL).

Separate pass, needs Stack B up:

    python process_documents.py --embed     # batches of 32 → /v1/embeddings → pgvector

Runs under the ingest_rw role (PG_DSN_INGEST). Every action → audit_log.

    python process_documents.py             # drain the queue once, exit
    python process_documents.py --loop      # keep polling every 3 s
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import psycopg
import pypdfium2 as pdfium
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

PG_DSN = os.getenv("PG_DSN_INGEST", "postgresql://ingest_rw:ingest_rw@localhost:5432/bfsi")
S3_BUCKET = os.getenv("S3_BUCKET", "bfsi")

VL_BASE_URL = os.getenv("VL_BASE_URL", "")
VL_API_KEY = os.getenv("VL_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL") or os.getenv("CHAT_BASE_URL", "")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or os.getenv("CHAT_API_KEY", "")

CHUNK_MAX_CHARS = 3200               # ≈ 800 tokens
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+")


# ── Extraction schemas (pydantic) — one per doc_type ─────────────────────────

class Position(BaseModel):
    name: str
    qty: float | None = None
    unit_price_uah: float | None = None


class RepairInvoice(BaseModel):
    vendor_name: str | None
    invoice_number: str | None
    invoice_date: str | None          # YYYY-MM-DD
    vehicle_plate: str | None
    policy_number: str | None
    positions: list[Position] = []
    total_uah: float | None
    confidence_notes: list[str] = []


class DamagePhotoAct(BaseModel):
    act_number: str | None
    act_date: str | None
    address: str | None
    damage_description: str | None
    confidence_notes: list[str] = []


class PoliceReport(BaseModel):
    report_number: str | None
    report_date: str | None
    vehicle_plate: str | None
    summary: str | None
    confidence_notes: list[str] = []


SCHEMAS: dict[str, type[BaseModel]] = {
    "repair_invoice": RepairInvoice,
    "damage_photo_act": DamagePhotoAct,
    "police_report": PoliceReport,
}
REQUIRED: dict[str, list[str]] = {    # nulls here cost 0.15 confidence each
    "repair_invoice": ["vendor_name", "invoice_number", "invoice_date", "total_uah"],
    "damage_photo_act": ["act_number", "act_date", "damage_description"],
    "police_report": ["report_number", "report_date"],
}


# Explicit JSON shapes, written by hand ON PURPOSE. The first pilot run used
# model_json_schema()["properties"], which shows nested models only as a
# $ref — the model never saw Position's field names and invented its own
# ('description', 'total_price'): 7 of 8 failures + empty positions
# everywhere. The model must SEE the exact keys, not a schema reference.
SHAPES = {
    "repair_invoice": (
        '{"vendor_name": "…", "invoice_number": "…", "invoice_date": "YYYY-MM-DD", '
        '"vehicle_plate": "…", "policy_number": "…", '
        '"positions": [{"name": "текст з колонки Найменування", "qty": 1, '
        '"unit_price_uah": 12345}], '
        '"total_uah": 12345, "confidence_notes": []}'
    ),
    "damage_photo_act": (
        '{"act_number": "…", "act_date": "YYYY-MM-DD", "address": "…", '
        '"damage_description": "…", "confidence_notes": []}'
    ),
    "police_report": (
        '{"report_number": "…", "report_date": "YYYY-MM-DD", "vehicle_plate": "…", '
        '"summary": "…", "confidence_notes": []}'
    ),
}


def schema_prompt(doc_type: str) -> str:
    return (
        f"Це документ типу {doc_type}. Поверни ЛИШЕ JSON (без markdown) точно "
        f"такої форми: {SHAPES[doc_type]} Ключі — саме ці, іншими не заміняй. "
        f"У positions перепиши КОЖЕН рядок таблиці документа. Числа — числами, "
        f"без пробілів і валюти; дати — YYYY-MM-DD. Нечитабельне чи відсутнє "
        f"поле → null і причина в confidence_notes. Не вигадуй і не обчислюй "
        f"значення: якщо підсумок нечитабельний — total_uah = null, навіть "
        f"якщо його можна порахувати з позицій."
    )


# ── Clients ──────────────────────────────────────────────────────────────────

def s3_client():
    return boto3.client(
        "s3", endpoint_url=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"), region_name="us-east-1")


def openai_client(base_url: str, key: str) -> OpenAI:
    base = base_url.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return OpenAI(base_url=f"{base}/v1", api_key=key,
                  default_headers={"X-API-Key": key}, timeout=300.0)


def fetch_bytes(s3, file_ref: str) -> bytes:
    key = file_ref.split(f"s3://{S3_BUCKET}/", 1)[1]
    return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def audit(conn, action: str, detail: dict) -> None:
    conn.execute("INSERT INTO audit_log (actor, action, detail) VALUES ('ingest', %s, %s)",
                 (action, json.dumps(detail, ensure_ascii=False, default=str)))


# ── Scan branch (Stack A) ────────────────────────────────────────────────────

def page_png(data: bytes, file_ref: str) -> bytes:
    if file_ref.endswith(".pdf"):
        page = pdfium.PdfDocument(data)[0]
        img = page.render(scale=2.0).to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return data                                  # already an image


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def vl_extract(vl: OpenAI, png: bytes, doc_type: str) -> tuple[dict | None, str | None]:
    """Call the vision model; one retry with validation errors. → (data, error)."""
    b64 = base64.b64encode(png).decode("ascii")
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": [
            {"type": "text", "text": schema_prompt(doc_type)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }]
    last_err = ""
    for attempt in (1, 2):
        resp = vl.chat.completions.create(model=os.getenv("VL_MODEL_ID", "vision"),
                                          messages=messages, max_tokens=1024,
                                          temperature=0.0)
        raw = strip_fences(resp.choices[0].message.content or "")
        try:
            data = SCHEMAS[doc_type].model_validate(json.loads(raw))
            return data.model_dump(), None
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = str(e)[:500]
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"Попередня відповідь не пройшла валідацію: {last_err}. "
                             f"Поверни виправлений ЛИШЕ JSON за тією ж схемою."})
    return None, last_err


def confidence_of(extracted: dict, doc_type: str) -> float:
    nulls = sum(1 for f in REQUIRED[doc_type] if extracted.get(f) in (None, ""))
    return round(max(0.0, 1.0 - 0.15 * nulls), 2)


# Ukrainian plates print in Latin, but VLMs regularly emit Cyrillic
# lookalikes (the first run produced 'AA1234ВК' with a Cyrillic В).
# Normalize both sides before matching.
_CYR2LAT = str.maketrans("АВСЕІКМНОРТХУ", "ABCEIKMHOPTXY")


def norm_plate(s: str | None) -> str:
    return (s or "").upper().replace(" ", "").translate(_CYR2LAT)


def link_claim(conn, extracted: dict) -> tuple[str | None, str | None]:
    """policy_number or vehicle_plate → open claim. → (claim_id, note)."""
    pol_id = None
    pn = (extracted.get("policy_number") or "").strip().upper()
    if re.fullmatch(r"POL-\d{4}", pn):
        pol_id = pn
    elif extracted.get("vehicle_plate"):
        target = norm_plate(extracted["vehicle_plate"])
        for pid, plate in conn.execute(
                "SELECT id, vehicle_plate FROM policies WHERE vehicle_plate IS NOT NULL"):
            if norm_plate(plate) == target:
                pol_id = pid
                break
    if not pol_id:
        return None, "no policy_number/vehicle_plate to link by"
    row = conn.execute(
        "SELECT id FROM claims WHERE policy_id = %s "
        "ORDER BY (status IN ('fnol','docs_pending','review')) DESC, fnol_date DESC "
        "LIMIT 1", (pol_id,)).fetchone()
    return (row[0], None) if row else (None, f"no claim for {pol_id}")


def process_scan(conn, s3, vl: OpenAI, doc) -> None:
    doc_id, doc_type, file_ref, entity_id = doc
    png = page_png(fetch_bytes(s3, file_ref), file_ref)
    extracted, err = vl_extract(vl, png, doc_type)
    if extracted is None:
        conn.execute("UPDATE documents SET status='failed', error=%s WHERE id=%s",
                     (err, doc_id))
        audit(conn, "extract_failed", {"document_id": doc_id, "error": err})
        return
    conf = confidence_of(extracted, doc_type)
    link_err = None
    if doc_type == "repair_invoice":
        claim_id, note = link_claim(conn, extracted)
        if claim_id and not entity_id:
            conn.execute("UPDATE documents SET entity_id=%s WHERE id=%s",
                         (claim_id, doc_id))
            audit(conn, "link", {"document_id": doc_id, "claim_id": claim_id})
        elif claim_id and entity_id and claim_id != entity_id:
            link_err = f"link mismatch: extracted→{claim_id}, seeded→{entity_id}"
        elif not claim_id and not entity_id:
            link_err = note
    conn.execute(
        "UPDATE documents SET status='extracted', extracted=%s, confidence=%s, error=%s "
        "WHERE id=%s",
        (json.dumps(extracted, ensure_ascii=False), conf, link_err, doc_id))
    audit(conn, "extract", {"document_id": doc_id, "confidence": conf,
                            "nulls": [f for f in REQUIRED[doc_type]
                                      if extracted.get(f) in (None, "")]})


# ── Policy branch (no model) ─────────────────────────────────────────────────

def parse_policy(data: bytes) -> list[tuple[int, str | None, str]]:
    """PDF text → [(seq, section_ref, chunk_text)] — split on section numbers.

    This replaces the spec's Docling adapter on purpose: the pilot's policy
    PDFs are generated by create_demo_data.py with stable `N.` / `N.K.` numbering, so a
    regex chunker is exact and dependency-free. Swap point for real customer
    PDFs: replace THIS function (Docling/Unstructured), same return shape.
    """
    doc = pdfium.PdfDocument(data)
    text = "\n".join(page.get_textpage().get_text_range() for page in doc)
    chunks, cur_lines, cur_ref, seq = [], [], None, 0

    def flush():
        nonlocal seq, cur_lines
        body = "\n".join(cur_lines).strip()
        while body:
            head, body = body[:CHUNK_MAX_CHARS], body[CHUNK_MAX_CHARS:]
            chunks.append((seq, cur_ref, head))
            seq += 1
        cur_lines = []

    for line in text.splitlines():
        m = SECTION_RE.match(line.strip())
        if m and "." not in m.group(1):          # new TOP-level section → new chunk
            flush()
            cur_ref = m.group(1)
        cur_lines.append(line)
    flush()
    return chunks


def process_policy(conn, s3, doc) -> None:
    doc_id, _doc_type, file_ref, _entity_id = doc
    chunks = parse_policy(fetch_bytes(s3, file_ref))
    for seq, ref, text in chunks:
        conn.execute(
            "INSERT INTO doc_chunks (document_id, seq, section_ref, chunk_text) "
            "VALUES (%s,%s,%s,%s)", (doc_id, seq, ref, text))
    conn.execute("UPDATE documents SET status='extracted' WHERE id=%s", (doc_id,))
    audit(conn, "chunk", {"document_id": doc_id, "chunks": len(chunks)})


# ── Main loops ───────────────────────────────────────────────────────────────

CLAIM_SQL = """
SELECT id, doc_type, file_ref, entity_id FROM documents
WHERE status='pending' ORDER BY created_at LIMIT 1
FOR UPDATE SKIP LOCKED
"""


def drain(loop: bool) -> None:
    if not VL_BASE_URL:
        print("[ingest] WARNING: VL_BASE_URL empty — scans will fail; policy PDFs still chunk")
    s3 = s3_client()
    vl = openai_client(VL_BASE_URL, VL_API_KEY) if VL_BASE_URL else None
    conn = psycopg.connect(PG_DSN, autocommit=False)
    done = 0
    while True:
        with conn.transaction():
            row = conn.execute(CLAIM_SQL).fetchone()
            if row is None:
                if loop:
                    time.sleep(3)
                    continue
                break
            doc_id, doc_type = row[0], row[1]
            t0 = time.time()
            try:
                if doc_type == "policy_pdf":
                    process_policy(conn, s3, row)
                elif vl is None:
                    raise RuntimeError("VL_BASE_URL not set (Stack A not deployed?)")
                else:
                    process_scan(conn, s3, vl, row)
                print(f"[ingest] {doc_id} ({doc_type}) → done in {time.time()-t0:.1f}s")
            except Exception as e:                       # noqa: BLE001 — worker must survive
                conn.execute("UPDATE documents SET status='failed', error=%s WHERE id=%s",
                             (str(e)[:500], doc_id))
                audit(conn, "extract_failed", {"document_id": doc_id, "error": str(e)[:500]})
                print(f"[ingest] {doc_id} FAILED: {e}")
            done += 1
    counts = dict(conn.execute(
        "SELECT status, count(*) FROM documents GROUP BY status").fetchall())
    print(f"[ingest] processed {done}; documents by status: {counts}")
    conn.close()


def embed_pass() -> None:
    if not EMBED_BASE_URL:
        sys.exit("[ingest] EMBED_BASE_URL/CHAT_BASE_URL not set (Stack B not deployed?)")
    client = openai_client(EMBED_BASE_URL, EMBED_API_KEY)
    conn = psycopg.connect(PG_DSN, autocommit=True)
    total = 0
    while True:
        rows = conn.execute(
            "SELECT id, chunk_text FROM doc_chunks WHERE embedding IS NULL "
            "ORDER BY id LIMIT 32").fetchall()
        if not rows:
            break
        emb = client.embeddings.create(model=os.getenv("EMBED_MODEL_ID", "embed"),
                                       input=[r[1] for r in rows])
        for (chunk_id, _), item in zip(rows, emb.data):
            vec = "[" + ",".join(f"{x:.6f}" for x in item.embedding) + "]"
            conn.execute("UPDATE doc_chunks SET embedding=%s::vector WHERE id=%s",
                         (vec, chunk_id))
        total += len(rows)
        print(f"[ingest] embedded {total} chunks…")
    audit(conn, "embed", {"chunks": total})
    left = conn.execute("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL").fetchone()[0]
    print(f"[ingest] done: {total} embedded, {left} left NULL")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="fill doc_chunks.embedding (Stack B)")
    ap.add_argument("--loop", action="store_true", help="poll forever instead of drain-once")
    args = ap.parse_args()
    embed_pass() if args.embed else drain(loop=args.loop)
