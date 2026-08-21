"""BFSI pilot — integration eval (Definition of Done) + field accuracy.

    python run_eval.py               # 20 golden questions → eval/report.md
    python run_eval.py --fields      # М3: extracted vs ground_truth.json per field

Grading per question type:
  fact/aggregate    canonical values from expected_sql must appear in the reply
  citation          reply must cite the policy's document + the exact section
                    from ground_truth.citations (and the section must exist
                    in doc_chunks)
  invoice_coverage  the non-covered positions from ground truth are named
  red_flags         ≥ min_flag_claims of the expected claim ids are mentioned
  letter            the letter lists exactly the missing doc_types
Global metric on every reply: hallucinated CLM-/POL-/DOC- ids (target 0).
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

import httpx
import psycopg
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

PG_DSN = os.getenv("PG_DSN", "postgresql://bfsi:bfsi@localhost:5432/bfsi")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000")
GT = json.loads((HERE / "eval" / "ground_truth.json").read_text())
ID_RE = re.compile(r"\b(?:CLM|POL|DOC)-[A-Z0-9]{4,}\b")

DOC_TYPE_UA = {  # how a letter may name a doc type (any variant counts)
    "repair_invoice": ["рахунок", "рахунк"],
    "damage_photo_act": ["акт огляду", "акт", "фотофіксац"],
    "police_report": ["поліц", "єрдр", "протокол", "витяг"],
}


def sql_values(conn, sql: str) -> list[str]:
    """Flatten the expected_sql result into comparable strings."""
    out = []
    for row in conn.execute(sql).fetchall():
        for v in row:
            if v is None:
                continue
            s = str(v)
            out.append(s[:-2] if s.endswith(".0") else s)
    return out


def known_ids(conn) -> set[str]:
    ids = set()
    for table in ("clients", "policies", "claims", "documents"):
        ids |= {r[0] for r in conn.execute(f"SELECT id FROM {table}").fetchall()}
    return ids


def section_exists(conn, policy_doc_id: str, section: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM doc_chunks WHERE document_id=%s AND "
        "(section_ref=%s OR chunk_text LIKE %s) LIMIT 1",
        (policy_doc_id, section.split(".")[0], f"%{section}.%")).fetchone()
    return row is not None


def grade(conn, q: dict, reply: str, valid_ids: set[str]) -> tuple[bool, str]:
    low = reply.lower()

    if q["type"] in ("fact", "aggregate"):
        missing = [v for v in sql_values(conn, q["expected_sql"]) if v not in reply]
        for eid in q.get("expect_ids", []):
            if eid not in reply:
                missing.append(eid)
        if q.get("expect_contains_any") and not any(s.lower() in low for s in q["expect_contains_any"]):
            missing.append(f"any-of{q['expect_contains_any']}")
        return (not missing, f"missing: {missing}" if missing else "ok")

    if q["type"] == "citation":
        pol = q["citation"]["policy"]
        cite = GT["citations"][pol]
        section = cite[q["citation"]["expected_key"]]
        doc_id = cite["document_id"]
        cited = section in reply and (doc_id in reply or pol in reply)
        exists = section_exists(conn, doc_id, section)
        tone_ok = (not q.get("expect_contains_any")
                   or any(s.lower() in low for s in q["expect_contains_any"]))
        detail = f"expected §{section} of {doc_id}; cited={cited} exists_in_db={exists}"
        return (cited and exists and tone_ok, detail)

    if q["type"] == "invoice_coverage":
        gt = GT["invoice_coverage"][q["coverage_doc"]]
        named = [p for p in gt["non_covered_positions"]
                 if p.lower().split(":")[0] in low]
        sec_ok = gt["cite"]["tuning_section"] in reply
        detail = f"named {len(named)}/{len(gt['non_covered_positions'])} positions, §tuning cited={sec_ok}"
        return (len(named) == len(gt["non_covered_positions"]) and sec_ok, detail)

    if q["type"] == "red_flags":
        hits = [c for c in q["expect_ids_any"] if c in reply]
        flagged_claims = {c for c in hits}
        need = q.get("min_flag_claims", 1)
        return (len(flagged_claims) >= need, f"{len(flagged_claims)}/{need} flagged claims mentioned")

    if q["type"] == "letter":
        expected = GT["missing_docs"][q["letter_claim"]]
        ok_types = [t for t in expected if any(w in low for w in DOC_TYPE_UA[t])]
        extra = [t for t, words in DOC_TYPE_UA.items()
                 if t not in expected and any(w in low for w in words)
                 and t != "repair_invoice"]     # invoices are often mentioned as received
        detail = f"listed {len(ok_types)}/{len(expected)} missing types; spurious={extra}"
        return (len(ok_types) == len(expected) and not extra, detail)

    return (False, f"unknown type {q['type']}")


def run_eval() -> None:
    questions = yaml.safe_load((HERE / "eval" / "golden_questions.yaml").read_text())
    conn = psycopg.connect(PG_DSN, autocommit=True)
    valid_ids = known_ids(conn)
    client = httpx.Client(timeout=600.0)

    rows, halluc_total = [], 0
    for q in questions:
        r = client.post(f"{AGENT_URL}/chat",
                        json={"session_id": f"eval-{q['id']}-{uuid.uuid4().hex[:4]}",
                              "message": q["question"]})
        reply = r.json().get("reply", "") if r.status_code == 200 else f"HTTP {r.status_code}"
        passed, detail = grade(conn, q, reply, valid_ids)
        halluc = sorted({m for m in ID_RE.findall(reply)
                         if m not in valid_ids and not m.startswith("DOC-U")})
        halluc_total += len(halluc)
        rows.append({"id": q["id"], "type": q["type"], "passed": passed,
                     "detail": detail, "hallucinated": halluc,
                     "n_tools": len(r.json().get("trace", [])) if r.status_code == 200 else 0})
        mark = "✅" if passed and not halluc else ("⚠️" if passed else "❌")
        print(f"{mark} {q['id']} [{q['type']}] {detail}"
              + (f"  HALLUCINATED: {halluc}" if halluc else ""))

    passed_n = sum(1 for r in rows if r["passed"])
    report = ["# BFSI pilot — eval report", "",
              f"- questions passed: **{passed_n}/{len(rows)}**",
              f"- hallucinated ids total: **{halluc_total}** (target 0)", "",
              "| id | type | pass | tools | detail | hallucinated |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        report.append(f"| {r['id']} | {r['type']} | {'✅' if r['passed'] else '❌'} | "
                      f"{r['n_tools']} | {r['detail']} | {', '.join(r['hallucinated']) or '—'} |")
    out = HERE / "eval" / "report.md"
    out.write_text("\n".join(report) + "\n")
    print(f"\n[eval] {passed_n}/{len(rows)} passed, {halluc_total} hallucinated ids → {out}")


# Cyrillic→Latin lookalikes: VLMs emit 'AA1234ВК' with a Cyrillic В for a
# Latin-printed plate. Both sides get the same transform, so it only removes
# encoding noise — a genuinely different letter still counts as a miss.
_CYR2LAT = str.maketrans("АВСЕІКМНОРТХУ", "ABCEIKMHOPTXY")


def _match(want, got) -> bool:
    """Field comparison: numeric-aware (27500 == 27500.0), homoglyph-tolerant."""
    if want is None or got is None:
        return want is None and got is None
    try:
        return float(want) == float(got)
    except (TypeError, ValueError):
        pass
    norm = lambda v: re.sub(r"\s+", "", str(v)).upper().translate(_CYR2LAT)
    return norm(want) == norm(got)


def run_fields() -> None:
    """М3 criterion: per-field accuracy of extraction vs ground truth."""
    conn = psycopg.connect(PG_DSN, autocommit=True)
    docs = {r[0]: r for r in conn.execute(
        "SELECT id, status, confidence, error, extracted, doc_type, file_ref "
        "FROM documents")}
    stats: dict[str, list[int]] = {}
    misses, cards = [], []
    for doc_id, truth in GT["invoices"].items():
        db = docs.get(doc_id)
        extracted = (db[4] or {}) if db else {}
        fields = []
        for field, want in truth.items():
            if field == "positions_count":
                got = len(extracted.get("positions") or [])
            else:
                got = extracted.get(field)
            hit = _match(want, got)
            s = stats.setdefault(field, [0, 0])
            s[0] += hit
            s[1] += 1
            if not hit:
                misses.append(f"  {doc_id}.{field}: want={want!r} got={got!r}")
            fields.append((field, want, got, hit))
        cards.append((doc_id, db, fields))

    print("field accuracy (correct/total):")
    for field, (ok, total) in sorted(stats.items()):
        print(f"  {field:18s} {ok}/{total}  ({ok/total:.0%})")
    if misses:
        print("misses:")
        print("\n".join(misses))
    failed = sorted((k, v[3]) for k, v in docs.items() if v[1] == "failed")
    if failed:
        print("failed documents (error, перший рядок):")
        for doc_id, err in failed:
            print(f"  {doc_id}: {(err or '').splitlines()[0][:140]}")
    _write_html_report(cards)


def _write_html_report(cards) -> None:
    """eval/fields_report.html — документ поруч із таблицею еталон/модель."""
    import base64
    import html as _html
    import io as _io

    import boto3
    import pypdfium2 as pdfium
    from botocore.client import Config as BotoConfig

    bucket = os.getenv("S3_BUCKET", "bfsi")
    s3 = boto3.client(
        "s3", endpoint_url=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"), region_name="us-east-1")

    esc = lambda v: _html.escape("∅" if v is None else str(v))
    parts = ["""<meta charset="utf-8"><title>BFSI — field extraction report</title><style>
body{font:14px/1.45 system-ui;margin:24px;background:#f6f7f9;color:#1c2733}
section{background:#fff;border:1px solid #dde3ea;border-radius:10px;padding:16px;margin:0 0 20px}
h2{margin:0 0 8px;font-size:16px} h2 small{color:#68778a;font-weight:400}
.row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
img{max-width:460px;width:100%;border:1px solid #dde3ea;border-radius:6px}
table{border-collapse:collapse;flex:1;min-width:320px}
td,th{border:1px solid #dde3ea;padding:4px 9px;text-align:left;vertical-align:top}
th{background:#eef1f5} tr.ok td:last-child{color:#188038;font-weight:700}
tr.miss td{background:#fdecec} tr.miss td:last-child{color:#c5221f;font-weight:700}
.err{background:#fff4e5;border:1px solid #f0c36d;border-radius:6px;padding:6px 10px;
     margin:0 0 10px;font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap}
</style><h1>Екстракція проти еталону — по документах</h1>"""]

    for doc_id, db, fields in cards:
        img_b64 = ""
        if db is not None:
            try:
                key = db[6].split(f"s3://{bucket}/", 1)[1]
                data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                if key.endswith(".pdf"):
                    buf = _io.BytesIO()
                    pdfium.PdfDocument(data)[0].render(scale=1.6).to_pil().save(
                        buf, format="PNG")
                    data = buf.getvalue()
                img_b64 = base64.b64encode(data).decode("ascii")
            except Exception as exc:                       # noqa: BLE001 — report anyway
                print(f"[eval] no image for {doc_id}: {exc}")
        status = f"{db[1]}, confidence={db[2]}" if db else "нема в БД"
        err = (f"<div class='err'>{esc((db[3] or '')[:500])}</div>"
               if db and db[3] else "")
        rows = "".join(
            f"<tr class='{'ok' if hit else 'miss'}'><td>{esc(f)}</td>"
            f"<td>{esc(want)}</td><td>{esc(got)}</td><td>{'✓' if hit else '✗'}</td></tr>"
            for f, want, got, hit in fields)
        img_tag = (f"<img src='data:image/png;base64,{img_b64}'>" if img_b64 else "")
        parts.append(
            f"<section><h2>{doc_id} <small>{esc(db[5]) if db else ''} — {esc(status)}"
            f"</small></h2>{err}<div class='row'>{img_tag}"
            f"<table><tr><th>поле</th><th>еталон</th><th>модель</th><th></th></tr>"
            f"{rows}</table></div></section>")

    out = HERE / "eval" / "fields_report.html"
    out.write_text("\n".join(parts))
    print(f"[eval] візуальний звіт → {out}  (відкрийте в браузері)")


if __name__ == "__main__":
    run_fields() if "--fields" in sys.argv else run_eval()
