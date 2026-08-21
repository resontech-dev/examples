"""Закинь свої файли в inputs_docs/ — цей скрипт поставить їх у чергу обробки.

Що він робить: файл → MinIO + рядок у `documents` зі status='pending'.
Що він НЕ робить: сам прогін через VLM. Обробку черги (екстракцію полів)
виконує process_documents.py, коли задеплоєний Стек A — черга спільна
з демо-документами:

    # 1. поклади .pdf/.png/.jpg у inputs_docs/
    python add_documents.py                        # всі файли, тип repair_invoice
    python add_documents.py --type damage_photo_act
    python add_documents.py --claim CLM-0007       # одразу привʼязати до справи
    # 2. коли активний Стек A:
    python process_documents.py                    # → status extracted/failed

Повторний запуск безпечний: уже зареєстровані файли (за іменем)
пропускаються. Для вимірюваної точності допиши правильні значення полів у
eval/ground_truth.json → "invoices" — скрипт кладе готові заготовки в
eval/ground_truth_todo.json (лишається заповнити й перенести).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

import boto3
import psycopg
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

PG_DSN = os.getenv("PG_DSN_INGEST", "postgresql://ingest_rw:ingest_rw@localhost:5432/bfsi")
S3_BUCKET = os.getenv("S3_BUCKET", "bfsi")
INPUTS = HERE / "inputs_docs"
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg"}
DOC_TYPES = ("repair_invoice", "damage_photo_act", "police_report", "policy_pdf")

# Поля-заготовки для ground truth за типом документа (як у create_demo_data.py).
GT_STUB = {
    "repair_invoice": {"vendor_name": None, "invoice_number": None,
                       "invoice_date": None, "vehicle_plate": None,
                       "total_uah": None, "positions_count": None},
    "damage_photo_act": {"act_number": None, "act_date": None},
    "police_report": {"report_number": None, "report_date": None, "vehicle_plate": None},
}


def s3_client():
    return boto3.client(
        "s3", endpoint_url=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"), region_name="us-east-1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--type", choices=DOC_TYPES, default="repair_invoice",
                    help="тип для ВСІХ файлів цього запуску (default: repair_invoice)")
    ap.add_argument("--claim", default="",
                    help="CLM-… привʼязати одразу (інакше лінкує воркер за номером "
                         "авто/поліса з extracted)")
    ap.add_argument("--folder", type=Path, default=INPUTS)
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir()
                   if p.is_file() and p.suffix.lower() in ALLOWED) if args.folder.is_dir() else []
    if not files:
        sys.exit(f"[add] немає файлів у {args.folder} (підтримка: {sorted(ALLOWED)})")

    s3 = s3_client()
    conn = psycopg.connect(PG_DSN, autocommit=True)
    if args.claim:
        row = conn.execute("SELECT 1 FROM claims WHERE id=%s", (args.claim,)).fetchone()
        if row is None:
            sys.exit(f"[add] справи {args.claim!r} немає в БД")

    entity_type = "policy" if args.type == "policy_pdf" else "claim"
    # DOC-L0001, DOC-L0002… — продовжуємо нумерацію локально доданих.
    n_existing = conn.execute(
        "SELECT count(*) FROM documents WHERE id LIKE 'DOC-L%'").fetchone()[0]

    added, skipped, todo = 0, 0, {}
    for path in files:
        key = f"inputs/{path.name}"
        file_ref = f"s3://{S3_BUCKET}/{key}"
        if conn.execute("SELECT 1 FROM documents WHERE file_ref=%s",
                        (file_ref,)).fetchone():
            skipped += 1
            continue
        n_existing += 1
        doc_id = f"DOC-L{n_existing:04d}"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=path.read_bytes(),
                      ContentType=mimetypes.guess_type(path.name)[0]
                      or "application/octet-stream")
        conn.execute(
            "INSERT INTO documents (id, entity_type, entity_id, doc_type, file_ref) "
            "VALUES (%s,%s,%s,%s,%s)",
            (doc_id, entity_type, args.claim, args.type, file_ref))
        conn.execute(
            "INSERT INTO audit_log (actor, action, detail) VALUES ('add_documents','enqueue',%s)",
            (json.dumps({"document_id": doc_id, "file": path.name,
                         "doc_type": args.type}),))
        if args.type in GT_STUB:
            todo[doc_id] = {"_file": path.name, **GT_STUB[args.type]}
        added += 1
        print(f"[add] {doc_id} ← {path.name}  ({args.type}"
              + (f", claim={args.claim}" if args.claim else ", без привʼязки — злінкує воркер")
              + ")")

    if todo:
        todo_path = HERE / "eval" / "ground_truth_todo.json"
        merged = json.loads(todo_path.read_text()) if todo_path.is_file() else {}
        merged.update(todo)
        todo_path.parent.mkdir(exist_ok=True)
        todo_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
        print(f"[add] заготовки ground truth → {todo_path.name} "
              f"(заповни і перенеси в ground_truth.json → invoices)")

    pending = conn.execute(
        "SELECT count(*) FROM documents WHERE status='pending'").fetchone()[0]
    print(f"[add] додано {added}, пропущено (вже в базі) {skipped}; "
          f"у черзі pending={pending}")
    print("[add] далі: python process_documents.py  (потрібен активний Стек A)")
    conn.close()


if __name__ == "__main__":
    main()
