"""BFSI pilot — Module 4: agent backend (one file: FastAPI + tools + prompts).

POST /chat {session_id, message} (JSON), or multipart with an optional
`file` — the upload lands in MinIO as a pending document ("прийнято,
обробка після інджест-вікна" on the 1-GPU pilot).

Agent loop: session history (in-memory) → chat/completions with tools →
execute tool_calls → repeat (max 6 rounds) → final text + trace.

Tools (all under the agent_ro role — the DB itself blocks writes):
  run_sql       SELECT-only guard + forced LIMIT 200
  search_docs   query embedding → pgvector top-8 cosine
  get_document  metadata + extracted + file_ref
  red_flags     5 FIXED queries in code — anti-fraud logic stays
                deterministic and auditable, never model-written SQL

Run:
    python agent_server.py                 # uvicorn on :8000
    python agent_server.py --selftest      # proves the read-only wall holds
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import psycopg
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

PG_DSN = os.getenv("PG_DSN_AGENT", "postgresql://agent_ro:agent_ro@localhost:5432/bfsi")
S3_BUCKET = os.getenv("S3_BUCKET", "bfsi")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL") or CHAT_BASE_URL
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or CHAT_API_KEY
MAX_ROUNDS = 6

SESSIONS: dict[str, list[dict]] = {}

DDL = """
clients(id, name, type person|company, risk_level)
policies(id, client_id→clients, product KASKO|PROPERTY|HEALTH, valid_from, valid_to,
         deductible_uah, limit_uah, vehicle_plate, file_ref)
claims(id, policy_id→policies, status fnol|docs_pending|review|approved|paid|denied,
       fnol_date, loss_description, amount_claimed_uah, assignee)
documents(id, entity_type claim|policy|client, entity_id, doc_type
          repair_invoice|damage_photo_act|police_report|policy_pdf,
          file_ref, status pending|extracted|failed|verified,
          extracted jsonb, confidence, error, created_at)
doc_chunks(id, document_id→documents, seq, chunk_text, section_ref, embedding)
audit_log(id, actor, action, detail, ts)
""".strip()

SYSTEM_PROMPT = f"""Ти — асистент фахівця з врегулювання збитків страхової компанії.
Рішення ухвалює людина; ти готуєш факти, перевірки та чернетки.

Схема БД (Postgres, лише SELECT):
{DDL}

Приклади запитів:
  SELECT status, count(*) FROM claims GROUP BY status;
  SELECT c.id, c.status, c.fnol_date FROM claims c
    JOIN policies p ON p.id=c.policy_id WHERE p.client_id='CLT-0003';
  SELECT d.id, d.doc_type, d.status FROM documents d
    WHERE d.entity_type='claim' AND d.entity_id='CLM-0007';

Правила:
1. Факти — ЛИШЕ з результатів інструментів. Жодних вигаданих CLM-/POL-/DOC- ідентифікаторів.
2. Твердження про покриття чи виключення — ЛИШЕ з посиланням (document_id, пункт section_ref)
   з search_docs/get_document. Якщо пошук не дав пункту — скажи, що не знайдено.
3. Підозри на шахрайство — лише через red_flags (детермінований інструмент), не власні SQL-евристики.
4. Чернетка листа клієнту — формат:
   Тема: ...
   Шановний(а) {{ім'я}},
   {{суть: що отримано, чого бракує (перелік doc_type українською), дедлайн з полісу п.6.2}}
   З повагою, відділ врегулювання.
5. Відповідай мовою запитання (українська за замовчуванням). Коротко, по суті, з ідентифікаторами.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Read-only SQL SELECT над БД страхової. До 200 рядків.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Семантичний пошук по чанках документів (поліси). "
                       "Повертає document_id, section_ref, chunk_text.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "entity_id": {"type": ["string", "null"],
                                                    "description": "POL-/CLM- фільтр"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_document",
        "description": "Метадані документа + extracted JSON + file_ref за DOC-id.",
        "parameters": {"type": "object",
                       "properties": {"document_id": {"type": "string"}},
                       "required": ["document_id"]}}},
    {"type": "function", "function": {
        "name": "red_flags",
        "description": "Детермінований антифрод-скринінг: 5 фіксованих перевірок. "
                       "Повертає {claim_id, flag_type, detail}.",
        "parameters": {"type": "object", "properties": {}}}},
]

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|copy|vacuum)\b", re.I)


# ── Tool implementations (agent_ro connection) ───────────────────────────────

def db() -> psycopg.Connection:
    return psycopg.connect(PG_DSN, autocommit=True)


def tool_run_sql(conn, query: str) -> dict:
    q = query.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", q, re.I) or FORBIDDEN.search(q) or ";" in q:
        return {"error": "only a single SELECT statement is allowed"}
    try:
        cur = conn.execute(f"SELECT * FROM ({q}) _sub LIMIT 200")
        cols = [c.name for c in cur.description]
        rows = [list(r) for r in cur.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows)}
    except psycopg.Error as e:
        return {"error": f"SQL error: {str(e).splitlines()[0]}"}


_EMBED = OpenAI(
    base_url=(EMBED_BASE_URL.rstrip("/").removesuffix("/predict") + "/v1")
    if EMBED_BASE_URL else "http://unset/v1",
    api_key=EMBED_API_KEY or "unset",
    default_headers={"X-API-Key": EMBED_API_KEY or "unset"}, timeout=120.0,
)


def tool_search_docs(conn, query: str, entity_id: str | None = None) -> dict:
    emb = _EMBED.embeddings.create(model=os.getenv("EMBED_MODEL_ID", "embed"),
                                   input=[query]).data[0].embedding
    vec = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
    sql = ("SELECT dc.document_id, dc.section_ref, dc.chunk_text, d.entity_id "
           "FROM doc_chunks dc JOIN documents d ON d.id = dc.document_id "
           "WHERE dc.embedding IS NOT NULL ")
    params: list[Any] = []
    if entity_id:
        sql += "AND (d.entity_id = %s OR d.id = %s) "
        params += [entity_id, entity_id]
    sql += "ORDER BY dc.embedding <=> %s::vector LIMIT 8"
    params.append(vec)
    rows = conn.execute(sql, params).fetchall()
    return {"hits": [{"document_id": r[0], "section_ref": r[1],
                      "chunk_text": r[2][:600], "entity_id": r[3]} for r in rows]}


def tool_get_document(conn, document_id: str) -> dict:
    row = conn.execute(
        "SELECT id, entity_type, entity_id, doc_type, status, confidence, error, "
        "extracted, file_ref FROM documents WHERE id=%s", (document_id,)).fetchone()
    if row is None:
        return {"error": f"no document {document_id!r}"}
    return {"id": row[0], "entity_type": row[1], "entity_id": row[2],
            "doc_type": row[3], "status": row[4],
            "confidence": float(row[5]) if row[5] is not None else None,
            "error": row[6], "extracted": row[7], "file_ref": row[8]}


RED_FLAG_QUERIES = {
    # (а) same client, ≥2 claims within 60 days
    "repeat_client_60d": """
        SELECT c1.id AS claim_id, p1.client_id,
               c2.id AS other_claim, abs(c1.fnol_date - c2.fnol_date) AS days_apart
        FROM claims c1 JOIN policies p1 ON p1.id = c1.policy_id
        JOIN claims c2 ON c2.id <> c1.id
        JOIN policies p2 ON p2.id = c2.policy_id AND p2.client_id = p1.client_id
        WHERE abs(c1.fnol_date - c2.fnol_date) <= 60""",
    # (б) amount > 150% of the product average
    "amount_above_150pct_avg": """
        SELECT c.id AS claim_id, c.amount_claimed_uah, p.product,
               round(avg_amt.avg_a) AS product_avg
        FROM claims c JOIN policies p ON p.id = c.policy_id
        JOIN (SELECT p2.product, avg(c2.amount_claimed_uah) avg_a
              FROM claims c2 JOIN policies p2 ON p2.id = c2.policy_id
              GROUP BY p2.product) avg_amt ON avg_amt.product = p.product
        WHERE c.amount_claimed_uah > 1.5 * avg_amt.avg_a""",
    # (в) claim on an expired policy
    "expired_policy": """
        SELECT c.id AS claim_id, c.fnol_date, p.id AS policy_id, p.valid_to
        FROM claims c JOIN policies p ON p.id = c.policy_id
        WHERE c.fnol_date > p.valid_to""",
    # (г) the same repair vendor across claims of DIFFERENT clients
    "shared_vendor": """
        SELECT d.entity_id AS claim_id, d.extracted->>'vendor_name' AS vendor
        FROM documents d
        WHERE d.doc_type='repair_invoice' AND d.status='extracted'
          AND d.extracted->>'vendor_name' IN (
            SELECT dd.extracted->>'vendor_name'
            FROM documents dd
            JOIN claims cc ON cc.id = dd.entity_id
            JOIN policies pp ON pp.id = cc.policy_id
            WHERE dd.doc_type='repair_invoice' AND dd.status='extracted'
              AND dd.extracted->>'vendor_name' IS NOT NULL
            GROUP BY dd.extracted->>'vendor_name'
            HAVING count(DISTINCT pp.client_id) > 1)""",
    # (д) review status with zero attached documents
    "review_without_documents": """
        SELECT c.id AS claim_id, c.status, c.amount_claimed_uah
        FROM claims c
        WHERE c.status='review'
          AND NOT EXISTS (SELECT 1 FROM documents d
                          WHERE d.entity_type='claim' AND d.entity_id = c.id)""",
}


def tool_red_flags(conn) -> dict:
    out = []
    for flag_type, sql in RED_FLAG_QUERIES.items():
        cur = conn.execute(sql)
        cols = [c.name for c in cur.description]
        for row in cur.fetchall():
            out.append({"claim_id": row[0], "flag_type": flag_type,
                        "detail": dict(zip(cols[1:], map(str, row[1:])))})
    # de-dup: scenario (а) yields mirrored pairs
    seen, uniq = set(), []
    for f in out:
        k = (f["claim_id"], f["flag_type"])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return {"flags": uniq, "checked": list(RED_FLAG_QUERIES)}


def audit(conn, actor: str, action: str, detail: dict) -> None:
    conn.execute("INSERT INTO audit_log (actor, action, detail) VALUES (%s,%s,%s)",
                 (actor, action, json.dumps(detail, ensure_ascii=False, default=str)))


# ── Agent loop ───────────────────────────────────────────────────────────────

_CHAT = OpenAI(
    base_url=(CHAT_BASE_URL.rstrip("/").removesuffix("/predict") + "/v1") if CHAT_BASE_URL else "http://unset/v1",
    api_key=CHAT_API_KEY or "unset",
    default_headers={"X-API-Key": CHAT_API_KEY or "unset"}, timeout=300.0,
)


def run_agent(session_id: str, message: str) -> dict:
    history = SESSIONS.setdefault(session_id,
                                  [{"role": "system", "content": SYSTEM_PROMPT}])
    history.append({"role": "user", "content": message})
    trace: list[dict] = []
    conn = db()
    actor = f"agent:{session_id}"
    try:
        for _ in range(MAX_ROUNDS):
            resp = _CHAT.chat.completions.create(
                model=os.getenv("CHAT_MODEL_ID", "assistant"),
                messages=history, tools=TOOLS,
                max_tokens=1024, temperature=0.1)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                history.append({"role": "assistant", "content": msg.content})
                return {"reply": msg.content, "trace": trace}
            history.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                t0 = time.time()
                fn = call.function.name
                if fn == "run_sql":
                    result = tool_run_sql(conn, args.get("query", ""))
                elif fn == "search_docs":
                    result = tool_search_docs(conn, args.get("query", ""),
                                              args.get("entity_id"))
                elif fn == "get_document":
                    result = tool_get_document(conn, args.get("document_id", ""))
                elif fn == "red_flags":
                    result = tool_red_flags(conn)
                else:
                    result = {"error": f"unknown tool {fn!r}"}
                ms = round((time.time() - t0) * 1000)
                rows = result.get("row_count") or len(result.get("hits", []) or
                                                     result.get("flags", []) or [])
                trace.append({"tool": fn, "args": args, "ms": ms, "rows": rows})
                audit(conn, actor, fn, {"args": args, "rows_returned": rows})
                history.append({"role": "tool", "tool_call_id": call.id,
                                "content": json.dumps(result, ensure_ascii=False,
                                                      default=str)[:8000]})
        return {"reply": "Ліміт ітерацій інструментів вичерпано — уточніть запит.",
                "trace": trace}
    finally:
        conn.close()


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="bfsi-agent")


@app.post("/chat")
async def chat(request: Request):
    """JSON {session_id, message} — plain chat; multipart adds a `file` part."""
    ctype = request.headers.get("content-type", "")
    file_name, file_data = None, None
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        session_id = form.get("session_id")
        message = form.get("message", "")
        up = form.get("file")
        if up is not None and hasattr(up, "read"):
            file_name, file_data = up.filename, await up.read()
    else:
        body = await request.json()
        session_id, message = body.get("session_id"), body.get("message", "")
    session_id = session_id or uuid.uuid4().hex[:8]

    upload_note = ""
    if file_data is not None:
        doc_id = f"DOC-U{uuid.uuid4().hex[:6].upper()}"
        key = f"uploads/{doc_id}_{file_name}"
        s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
                          aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
                          aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
                          config=BotoConfig(signature_version="s3v4"), region_name="us-east-1")
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_data)
        # agent_ro cannot INSERT documents — by design. Use the ingest role
        # for the enqueue only (the one narrowly-scoped write the API does).
        with psycopg.connect(os.getenv("PG_DSN_INGEST",
                                       "postgresql://ingest_rw:ingest_rw@localhost:5432/bfsi"),
                             autocommit=True) as ic:
            ic.execute("INSERT INTO documents (id, entity_type, entity_id, doc_type, file_ref) "
                       "VALUES (%s,'claim','', 'repair_invoice', %s)",
                       (doc_id, f"s3://{S3_BUCKET}/{key}"))
        upload_note = (f"[Документ {doc_id} прийнято; на 1-GPU пілоті обробка "
                       f"відбудеться в наступне інджест-вікно (Стек A).]\n\n")

    result = run_agent(session_id, message or "")
    return {"session_id": session_id, "reply": upload_note + (result["reply"] or ""),
            "trace": result["trace"]}


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


# ── Selftest: the read-only wall ─────────────────────────────────────────────

def selftest() -> None:
    conn = db()
    r = tool_run_sql(conn, "UPDATE claims SET status='paid'")
    assert "error" in r, "guard let UPDATE through"
    r = tool_run_sql(conn, "SELECT count(*) FROM claims; DROP TABLE claims")
    assert "error" in r, "guard let a second statement through"
    try:
        conn.execute("UPDATE claims SET status='paid' WHERE id='CLM-0001'")
        raise SystemExit("FAIL: agent_ro role allowed UPDATE — check db/001_core.sql grants")
    except psycopg.errors.InsufficientPrivilege:
        pass
    conn.execute("INSERT INTO audit_log (actor, action, detail) VALUES "
                 "('selftest','selftest','{}')")
    n = conn.execute("SELECT count(*) FROM claims").fetchone()[0]
    print(f"[selftest] guard ✔  role wall ✔  audit insert ✔  (claims={n})")
    conn.close()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AGENT_PORT", "8000")))
