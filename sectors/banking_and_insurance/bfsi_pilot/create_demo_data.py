"""BFSI pilot — Module 1: synthetic seed (data + documents + ground truth).

One deterministic pass (seeded RNG):
  * 12 clients, 10 policies (4 KASKO / 3 PROPERTY / 3 HEALTH, 2 expired),
    50 claims with 5 engineered red-flag scenarios;
  * 10 policy PDFs (numbered sections — citations are tested against them),
    20 invoice/act/report files: 12 clean PDFs, 5 "phone photos"
    (perspective + noise via PIL, no OpenCV), 3 deliberately broken;
  * everything → MinIO + `documents` rows (status='pending');
  * eval/ground_truth.json — correct field values per scan, red-flag list,
    citation section numbers, expected doc→claim links, missing-docs lists.

Run (after `docker compose up -d`):

    pip install -r requirements.txt
    python create_demo_data.py            # idempotent: truncates and re-seeds

A Cyrillic-capable TTF is required for the PDFs (base PDF fonts have no
Ukrainian glyphs). The script probes common OS locations and ./fonts/*.ttf;
set SEED_FONT=/path/to/font.ttf to override.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import random
import sys
from pathlib import Path

import boto3
import numpy as np
import psycopg
import pypdfium2 as pdfium
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

RNG = random.Random(7)
TODAY = dt.date(2026, 8, 1)          # frozen "now" so date-based flags are stable

PG_DSN = os.getenv("PG_DSN", "postgresql://bfsi:bfsi@localhost:5432/bfsi")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_BUCKET = os.getenv("S3_BUCKET", "bfsi")

# ── Font (Cyrillic) ──────────────────────────────────────────────────────────

FONT_CANDIDATES = [
    os.getenv("SEED_FONT", ""),
    str(HERE / "fonts" / "DejaVuSans.ttf"),
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",     # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # debian/ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def register_font() -> str:
    for path in FONT_CANDIDATES:
        if path and Path(path).is_file():
            pdfmetrics.registerFont(TTFont("Body", path))
            return path
    sys.exit("[seed] No Cyrillic TTF found. Put one at fonts/DejaVuSans.ttf "
             "or set SEED_FONT=/path/to/font.ttf")


# ── Reference data ───────────────────────────────────────────────────────────

PERSON_NAMES = [
    "Іван Мельник", "Оксана Шевченко", "Петро Коваль", "Марія Бондаренко",
    "Андрій Ткаченко", "Наталія Кравець", "Сергій Поліщук", "Юлія Савченко",
]
COMPANY_NAMES = ["ТОВ «Будсервіс»", "ФОП Романюк О.В.", "ТОВ «Логістик-Захід»", "ПП «Аграрій»"]
ASSIGNEES = ["О. Гнатюк", "І. Литвин", "М. Демчук"]
VENDORS = ["СТО «Автомайстер-Плюс»", "СТО «Мотор-Сервіс»", "ТОВ «РемБуд-Сервіс»",
           "СТО «Драйв-Авто»", "ФОП Захарчук (сантехроботи)"]
SHARED_VENDOR = VENDORS[0]           # red flag (г): same СТО across different clients
PLATES = ["AA1234BK", "KA7781EX", "BC4412HT", "AI9056OP"]

LOSSES = {
    "KASKO": ["ДТП на перехресті, пошкоджено передній бампер",
              "Падіння гілки на дах авто", "Пошкодження на парковці, зникла оптика"],
    "PROPERTY": ["Залиття квартири через аварію водопроводу",
                 "Пожежа у підсобному приміщенні", "Крадіжка зі зломом"],
    "HEALTH": ["Стаціонарне лікування після травми", "Планова операція"],
}

P = ParagraphStyle("body", fontName="Body", fontSize=10, leading=14)
H1 = ParagraphStyle("h1", fontName="Body", fontSize=15, leading=19, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Body", fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)


# ── Entity generation ────────────────────────────────────────────────────────

def make_clients() -> list[dict]:
    out = []
    for i, name in enumerate(PERSON_NAMES, 1):
        out.append({"id": f"CLT-{i:04d}", "name": name, "type": "person",
                    "risk_level": "normal"})
    for j, name in enumerate(COMPANY_NAMES, len(PERSON_NAMES) + 1):
        out.append({"id": f"CLT-{j:04d}", "name": name, "type": "company",
                    "risk_level": "normal"})
    return out


def make_policies(clients: list[dict]) -> list[dict]:
    """10 policies: 4 KASKO (with plates), 3 PROPERTY, 3 HEALTH; 2 expired."""
    # Order matters: fixtures in make_claims() reference these ids.
    # KASKO = POL-0001/0004/0005/0006 (plates!), PROPERTY = 0002/0003/0008,
    # HEALTH = 0007/0009/0010; POL-0006 & POL-0009 are the expired pair.
    spec = [("KASKO", 0), ("PROPERTY", 5000), ("PROPERTY", 0), ("KASKO", 10000),
            ("KASKO", 5000), ("KASKO", 5000), ("HEALTH", 0),
            ("PROPERTY", 10000), ("HEALTH", 0), ("HEALTH", 5000)]
    limits = {"KASKO": [150_000, 300_000, 500_000, 250_000],
              "PROPERTY": [400_000, 600_000, 1_000_000],
              "HEALTH": [100_000, 200_000, 150_000]}
    out, counters = [], {"KASKO": 0, "PROPERTY": 0, "HEALTH": 0}
    plate_iter = iter(PLATES)
    for i, (product, deductible) in enumerate(spec, 1):
        k = counters[product]; counters[product] += 1
        # POL-0006 (KASKO) and POL-0009 (HEALTH) are EXPIRED — red flag (в).
        expired = i in (6, 9)
        valid_from = TODAY - dt.timedelta(days=RNG.randint(200, 330))
        valid_to = (TODAY - dt.timedelta(days=RNG.randint(20, 60)) if expired
                    else valid_from + dt.timedelta(days=365))
        out.append({
            "id": f"POL-{i:04d}",
            # Injective policy→client mapping: repeat-claims-per-client is an
            # ENGINEERED red flag (CLM-0021/22 on POL-0005), not seed noise.
            "client_id": clients[i - 1]["id"],
            "product": product, "valid_from": valid_from, "valid_to": valid_to,
            "deductible_uah": deductible, "limit_uah": limits[product][k % len(limits[product])],
            "vehicle_plate": next(plate_iter) if product == "KASKO" else None,
            "file_ref": f"s3://{S3_BUCKET}/policies/POL-{i:04d}.pdf",
        })
    return out


def make_claims(policies: list[dict]) -> tuple[list[dict], list[dict]]:
    """50 claims with engineered red flags + fixtures for golden questions."""
    by_id = {p["id"]: p for p in policies}
    statuses = ["fnol", "docs_pending", "review", "approved", "paid", "denied"]
    claims = []

    def add(cid, policy_id, status, days_ago, amount, loss=None, assignee=None):
        p = by_id[policy_id]
        claims.append({
            "id": cid, "policy_id": policy_id, "status": status,
            "fnol_date": TODAY - dt.timedelta(days=days_ago),
            "loss_description": loss or RNG.choice(LOSSES[p["product"]]),
            "amount_claimed_uah": amount,
            "assignee": assignee or RNG.choice(ASSIGNEES),
        })

    # Fixtures the golden questions rely on (stable ids). Same-policy
    # fixtures sit ≥61 days apart so red flag (а) fires ONLY for the
    # engineered CLM-0021/22 pair on POL-0005.
    add("CLM-0004", "POL-0004", "docs_pending", 12, 48_000)        # letter: missing docs
    add("CLM-0007", "POL-0001", "docs_pending", 25, 61_500)        # fact question
    add("CLM-0012", "POL-0003", "review", 18, 85_000,
        loss="Залиття квартири через аварію водопроводу")          # coverage+citation
    add("CLM-0019", "POL-0001", "review", 90, 187_000)             # invoice DOC-0011 > limit 150k
    # Red flag (а): same client (POL-0005's holder), 2 claims 40 days apart.
    add("CLM-0021", "POL-0005", "review", 55, 52_000)
    add("CLM-0022", "POL-0005", "docs_pending", 15, 44_000)
    # Red flag (б): KASKO amount ≫ 150% of product average.
    add("CLM-0030", "POL-0004", "review", 145, 420_000)
    # Red flag (в): claim on EXPIRED policy (fnol after valid_to).
    add("CLM-0033", "POL-0006", "fnol", 10, 66_000)
    # Red flag (г): shared СТО — the vendor appears in these two claims'
    # invoices (different clients: POL-0001 vs POL-0004 holders).
    add("CLM-0005", "POL-0001", "review", 160, 39_000)
    add("CLM-0011", "POL-0004", "docs_pending", 80, 57_000)
    # Red flag (д): review with zero documents.
    add("CLM-0040", "POL-0008", "review", 45, 120_000)

    fixture_ids = {c["id"] for c in claims}
    # Filler claims must not create accidental red flags:
    #   * no 'review' status — scenario (д) stays unique to CLM-0040
    #     (fillers carry no documents);
    #   * per-policy fnol dates ≥ 61 days from EVERY other claim of that
    #     policy (incl. fixtures) — scenario (а) stays unique to CLM-0021/22
    #     (policy→client is injective, so policy == client here).
    filler_statuses = [s for s in statuses if s != "review"]
    taken_days: dict[str, int] = {}          # policy_id → oldest taken days_ago
    for c in claims:
        d = (TODAY - c["fnol_date"]).days
        taken_days[c["policy_id"]] = max(taken_days.get(c["policy_id"], 0), d)
    i = 1
    while len(claims) < 50:
        cid = f"CLM-{i:04d}"
        i += 1
        if cid in fixture_ids:
            continue
        pol = RNG.choice([p for p in policies if p["id"] != "POL-0006"])
        days_ago = taken_days.get(pol["id"], 0) + 61 + RNG.randint(0, 8)
        taken_days[pol["id"]] = days_ago
        avg = {"KASKO": 55_000, "PROPERTY": 90_000, "HEALTH": 30_000}[pol["product"]]
        add(cid, pol["id"], RNG.choice(filler_statuses), days_ago,
            int(RNG.uniform(0.4, 1.4) * avg))
    claims.sort(key=lambda c: c["id"])

    flags = [
        {"claim_id": "CLM-0021", "flag_type": "repeat_client_60d"},
        {"claim_id": "CLM-0022", "flag_type": "repeat_client_60d"},
        {"claim_id": "CLM-0030", "flag_type": "amount_above_150pct_avg"},
        {"claim_id": "CLM-0033", "flag_type": "expired_policy"},
        {"claim_id": "CLM-0005", "flag_type": "shared_vendor"},
        {"claim_id": "CLM-0011", "flag_type": "shared_vendor"},
        {"claim_id": "CLM-0040", "flag_type": "review_without_documents"},
    ]
    return claims, flags


# ── Policy PDF rendering (numbered sections → citations) ─────────────────────

COVERED = {
    "KASKO": ["ДТП за участю застрахованого ТЗ", "протиправні дії третіх осіб",
              "падіння предметів на ТЗ", "пожежа внаслідок ДТП"],
    "PROPERTY": ["пожежа", "вибух побутового газу",
                 "залиття внаслідок аварії систем водопостачання чи опалення",
                 "протиправні дії третіх осіб (крадіжка зі зломом)"],
    "HEALTH": ["стаціонарне лікування", "невідкладна хірургія",
               "діагностичні обстеження за направленням лікаря"],
}
EXCLUSIONS = {
    "KASKO": ["умисні дії страхувальника або членів його родини",
              "керування у стані алкогольного чи наркотичного сп'яніння",
              "використання ТЗ у змаганнях чи навчальній їзді",
              "тюнінг, додаткове обладнання та роботи з покращення ТЗ",
              "миття, полірування та косметичні послуги",
              "стихійне лихо (для базового пакета)"],
    "PROPERTY": ["умисні дії страхувальника",
                 "поступове проникнення вологи (грибок, пліснява)",
                 "стихійне лихо (для базового пакета)",
                 "ремонтні роботи без письмового погодження страховика",
                 "знос інженерних мереж понад 70%"],
    "HEALTH": ["лікування, не призначене лікарем", "косметичні процедури",
               "травми у стані алкогольного сп'яніння",
               "екстремальні види спорту без окремої програми",
               "хронічні стани, відомі до укладення договору"],
}


def render_policy_pdf(policy: dict, holder: str) -> tuple[bytes, dict]:
    """Build the PDF and return (bytes, citation map for ground truth)."""
    prod = policy["product"]
    story = [Paragraph(f"ДОГОВІР СТРАХУВАННЯ № {policy['id']}", H1),
             Paragraph(f"Продукт: {prod}. Страхувальник: {holder}.", P),
             Paragraph(f"Строк дії: {policy['valid_from']} — {policy['valid_to']}.", P),
             Spacer(1, 6 * mm)]

    def section(num: str, title: str, items: list[str] | None = None, text: str | None = None):
        story.append(Paragraph(f"{num}. {title}", H2))
        if text:
            story.append(Paragraph(f"{num}.1. {text}", P))
        for k, it in enumerate(items or [], 1):
            story.append(Paragraph(f"{num}.{k}. {it}", P))

    obj = {"KASKO": f"транспортний засіб, держ. номер {policy['vehicle_plate']}",
           "PROPERTY": "нерухоме майно страхувальника за адресою, зазначеною в заяві",
           "HEALTH": "життя та здоров'я застрахованої особи"}[prod]
    section("1", "ПРЕДМЕТ ДОГОВОРУ", text=f"Предметом договору є {obj}.")
    section("2", "СТРАХОВА СУМА ТА ЛІМІТИ",
            text=f"Страхова сума (ліміт відповідальності) — "
                 f"{policy['limit_uah']:,.0f} грн.".replace(",", " "))
    section("3", "ФРАНШИЗА",
            text=f"Безумовна франшиза — {policy['deductible_uah']:,.0f} грн "
                 f"за кожним страховим випадком.".replace(",", " "))
    section("4", "ПОКРИТІ РИЗИКИ", items=COVERED[prod])
    section("5", "ВИКЛЮЧЕННЯ ЗІ СТРАХОВОГО ПОКРИТТЯ", items=EXCLUSIONS[prod])
    section("6", "ПОРЯДОК ВИПЛАТ", items=[
        "повідомлення про подію протягом 3 робочих днів",
        "подання повного пакета документів протягом 30 днів",
        "виплата протягом 15 робочих днів після рішення",
    ])
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Документ згенеровано синтетично для тестування — не є договором.", P))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=policy["id"]).build(story)

    cites = {"exclusions_section": "5",
             "payout_docs_deadline": "6.2"}
    if prod == "PROPERTY":
        cites["water_damage_covered"] = f"4.{COVERED[prod].index('залиття внаслідок аварії систем водопостачання чи опалення') + 1}"
        cites["natural_disaster_excluded"] = f"5.{EXCLUSIONS[prod].index('стихійне лихо (для базового пакета)') + 1}"
    if prod == "KASKO":
        ex = EXCLUSIONS[prod]
        cites["tuning_excluded"] = f"5.{ex.index('тюнінг, додаткове обладнання та роботи з покращення ТЗ') + 1}"
        cites["washing_excluded"] = f"5.{ex.index('миття, полірування та косметичні послуги') + 1}"
        cites["alcohol_excluded"] = f"5.{next(i for i, e in enumerate(ex, 1) if 'сп' in e and 'яніння' in e)}"
    return buf.getvalue(), cites


# ── Scan rendering (invoices / acts / reports) ───────────────────────────────

def render_invoice_pdf(spec: dict) -> bytes:
    story = [Paragraph(f"РАХУНОК-ФАКТУРА № {spec['invoice_number']}", H1),
             Paragraph(f"Виконавець: {spec['vendor_name']}", P),
             Paragraph(f"Дата: {spec['invoice_date']}", P)]
    if spec.get("vehicle_plate"):
        story.append(Paragraph(f"Транспортний засіб, держ. номер: {spec['vehicle_plate']}", P))
    if spec.get("policy_number"):
        story.append(Paragraph(f"Поліс: {spec['policy_number']}", P))
    story.append(Spacer(1, 5 * mm))
    rows = [["Найменування", "К-сть", "Ціна, грн", "Сума, грн"]]
    for pos in spec["positions"]:
        rows.append([pos["name"], str(pos["qty"]), f"{pos['unit_price_uah']:,.0f}".replace(",", " "),
                     f"{pos['qty'] * pos['unit_price_uah']:,.0f}".replace(",", " ")])
    total_txt = ("нерозбірливо" if spec.get("unreadable_total")
                 else f"{spec['total_uah']:,.0f} грн".replace(",", " "))
    rows.append(["РАЗОМ", "", "", total_txt])
    t = Table(rows, colWidths=[80 * mm, 20 * mm, 30 * mm, 35 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Body"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, (0.5, 0.5, 0.5)),
        ("BACKGROUND", (0, 0), (-1, 0), (0.88, 0.91, 0.95)),
    ]))
    story.append(t)
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=spec["invoice_number"]).build(story)
    return buf.getvalue()


def render_act_pdf(title: str, lines: list[str]) -> bytes:
    story = [Paragraph(title, H1)] + [Paragraph(l, P) for l in lines]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=title).build(story)
    return buf.getvalue()


def pdf_to_photo(pdf_bytes: bytes, heavy_noise: bool = False) -> bytes:
    """Rasterize page 1 and fake a phone photo: perspective + noise + gray bg."""
    page = pdfium.PdfDocument(pdf_bytes)[0]
    img = page.render(scale=2.0).to_pil().convert("RGB")
    w, h = img.size
    j = int(w * 0.06)
    quad = (RNG.randint(0, j), RNG.randint(0, j),
            RNG.randint(0, j), h - RNG.randint(0, j),
            w - RNG.randint(0, j), h - RNG.randint(0, j),
            w - RNG.randint(0, j), RNG.randint(0, j))
    img = img.transform((w, h), Image.QUAD, quad, resample=Image.BILINEAR,
                        fillcolor=(118, 112, 106))
    arr = np.asarray(img).astype(np.int16)
    arr += RNG.randint(-14, -6)                                   # dim
    arr += np.random.default_rng(7).normal(0, 22 if heavy_noise else 9,
                                           arr.shape).astype(np.int16)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_scan_specs(policies: list[dict]) -> list[dict]:
    """20 scan documents: doc_type, format, content, target claim, quirks."""
    pol = {p["id"]: p for p in policies}

    def inv(doc_id, claim_id, policy_id, positions, fmt="pdf", vendor=None,
            linked=True, unreadable_total=False, no_plate=False):
        p = pol[policy_id]
        total = sum(x["qty"] * x["unit_price_uah"] for x in positions)
        return {
            "id": doc_id, "doc_type": "repair_invoice", "format": fmt,
            "claim_id": claim_id, "linked": linked,
            "vendor_name": vendor or RNG.choice(VENDORS[1:]),
            "invoice_number": f"РФ-{doc_id[-4:]}",
            "invoice_date": str(TODAY - dt.timedelta(days=RNG.randint(3, 40))),
            "vehicle_plate": None if no_plate else p["vehicle_plate"],
            "policy_number": policy_id, "positions": positions,
            "total_uah": total, "unreadable_total": unreadable_total,
        }

    def pos(name, qty, price):
        return {"name": name, "qty": qty, "unit_price_uah": price}

    specs = [
        # 12 clean PDFs ────────────────────────────────────────────────────────
        inv("DOC-0001", "CLM-0007", "POL-0001",
            [pos("Бампер передній (заміна)", 1, 18_500), pos("Фарбування", 1, 9_000)]),
        inv("DOC-0002", "CLM-0030", "POL-0004",
            [pos("Кузовний ремонт після ДТП", 1, 380_000), pos("Розбір/дефектовка", 1, 40_000)]),
        inv("DOC-0003", "CLM-0005", "POL-0001",
            [pos("Заміна лобового скла", 1, 24_000), pos("Герметизація", 1, 3_500)],
            vendor=SHARED_VENDOR),                                  # red flag (г) №1
        inv("DOC-0004", "CLM-0021", "POL-0005",
            [pos("Ремонт підвіски", 1, 31_000)], linked=False),     # linking test
        inv("DOC-0005", "CLM-0012", "POL-0003",
            [pos("Заміна стояка водопостачання", 1, 12_000),
             pos("Відновлення стелі та стін", 1, 46_000)]),
        inv("DOC-0006", "CLM-0022", "POL-0005",
            [pos("Заміна фари лівої", 1, 14_500)]),
        inv("DOC-0007", "CLM-0011", "POL-0004",
            [pos("Рихтування дверей", 2, 11_000)], vendor=SHARED_VENDOR),  # red flag (г) №2
        inv("DOC-0008", "CLM-0019", "POL-0001", [pos("Ремонт бампера", 1, 22_000)]),
        inv("DOC-0009", "CLM-0007", "POL-0001",
            [pos("Балансування та розвал", 1, 2_400)], linked=False),      # linking test
        inv("DOC-0010", "CLM-0033", "POL-0006", [pos("Ремонт капота", 1, 19_000)]),
        # DOC-0011 — THE invoice-vs-coverage fixture: tuning + washing +
        # total 260 500 грн > POL-0001's 150k limit.
        inv("DOC-0011", "CLM-0019", "POL-0001",
            [pos("Кузовний ремонт", 1, 96_000),
             pos("Тюнінг: спойлер та обвіс", 1, 120_000),
             pos("Миття та полірування", 1, 2_500),
             pos("Фарбування елементів", 2, 21_000)]),
        inv("DOC-0012", "CLM-0004", "POL-0004", [pos("Діагностика ходової", 1, 1_800)]),
        # 5 phone photos ──────────────────────────────────────────────────────
        inv("DOC-0013", "CLM-0021", "POL-0005",
            [pos("Заміна амортизаторів", 2, 8_900)], fmt="photo"),
        inv("DOC-0014", "CLM-0012", "POL-0003",
            [pos("Просушка приміщення", 1, 7_000)], fmt="photo", linked=False),
        {"id": "DOC-0015", "doc_type": "damage_photo_act", "format": "photo",
         "claim_id": "CLM-0012", "linked": True,
         "act_number": "АКТ-0015", "act_date": str(TODAY - dt.timedelta(days=16)),
         "damage_description": "Залиття стелі та стін кухні, площа ~14 м²",
         "address": "м. Київ, вул. Симиренка 12, кв. 8"},
        {"id": "DOC-0016", "doc_type": "damage_photo_act", "format": "photo",
         "claim_id": "CLM-0040", "linked": False,                  # (д) stays doc-less
         "act_number": "АКТ-0016", "act_date": str(TODAY - dt.timedelta(days=41)),
         "damage_description": "Пошкодження фасаду складу після пожежі",
         "address": "смт Гостомель, вул. Промислова 3"},
        {"id": "DOC-0017", "doc_type": "police_report", "format": "photo",
         "claim_id": "CLM-0030", "linked": True,
         "report_number": "ЄО-118-0442", "report_date": str(TODAY - dt.timedelta(days=29)),
         "vehicle_plate": "AA1234BK",
         "summary": "ДТП за участю двох ТЗ, пошкодження значні"},
        # 3 deliberately broken ───────────────────────────────────────────────
        inv("DOC-0018", "CLM-0022", "POL-0005",
            [pos("Ремонт порогів", 1, 27_500)], fmt="photo_bad",
            unreadable_total=True),                                 # нечитабельна сума
        inv("DOC-0019", "CLM-0011", "POL-0004",
            [pos("Заміна дзеркала", 1, 6_400)], no_plate=True),     # без номера авто
        {"id": "DOC-0020", "doc_type": "damage_photo_act", "format": "photo_bad",
         "claim_id": "CLM-0004", "linked": True,
         "act_number": "АКТ-0020", "act_date": str(TODAY - dt.timedelta(days=10)),
         "damage_description": "Подряпини та вм'ятини по правому борту",
         "address": "—"},
    ]
    assert len(specs) == 20, f"scan spec count drifted: {len(specs)}"
    return specs


def render_scan(spec: dict) -> tuple[bytes, str]:
    """Render one scan spec → (bytes, s3 extension)."""
    if spec["doc_type"] == "repair_invoice":
        pdf = render_invoice_pdf(spec)
    elif spec["doc_type"] == "damage_photo_act":
        pdf = render_act_pdf(
            f"АКТ ОГЛЯДУ ПОШКОДЖЕНЬ № {spec['act_number']}",
            [f"Дата огляду: {spec['act_date']}", f"Адреса/об'єкт: {spec['address']}",
             f"Опис пошкоджень: {spec['damage_description']}",
             "Фотофіксація: додається (аркуш 2)."])
    else:  # police_report
        pdf = render_act_pdf(
            f"ВИТЯГ З ЄРДР / ПРОТОКОЛ № {spec['report_number']}",
            [f"Дата: {spec['report_date']}",
             f"Держ. номер ТЗ: {spec.get('vehicle_plate', '—')}",
             f"Обставини: {spec['summary']}"])
    fmt = spec["format"]
    if fmt == "pdf":
        return pdf, "pdf"
    return pdf_to_photo(pdf, heavy_noise=(fmt == "photo_bad")), "png"


# ── Persistence ──────────────────────────────────────────────────────────────

def s3_client():
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"), region_name="us-east-1")


def main() -> None:
    register_font()
    clients = make_clients()
    policies = make_policies(clients)
    claims, red_flags = make_claims(policies)
    scans = make_scan_specs(policies)
    client_of_policy = {p["id"]: p["client_id"] for p in policies}
    name_of_client = {c["id"]: c["name"] for c in clients}

    s3 = s3_client()
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=S3_BUCKET)

    conn = psycopg.connect(PG_DSN, autocommit=False)
    cur = conn.cursor()
    cur.execute("TRUNCATE audit_log, doc_chunks, documents, claims, policies, clients CASCADE")

    for c in clients:
        cur.execute("INSERT INTO clients VALUES (%s,%s,%s,%s)",
                    (c["id"], c["name"], c["type"], c["risk_level"]))
    for p in policies:
        cur.execute("INSERT INTO policies VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (p["id"], p["client_id"], p["product"], p["valid_from"], p["valid_to"],
                     p["deductible_uah"], p["limit_uah"], p["vehicle_plate"], p["file_ref"]))
    for c in claims:
        cur.execute("INSERT INTO claims VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (c["id"], c["policy_id"], c["status"], c["fnol_date"],
                     c["loss_description"], c["amount_claimed_uah"], c["assignee"]))

    # Policy PDFs → MinIO + documents(pending) + citation ground truth.
    citations = {}
    for i, p in enumerate(policies, 1):
        pdf, cites = render_policy_pdf(p, name_of_client[p["client_id"]])
        key = f"policies/{p['id']}.pdf"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=pdf, ContentType="application/pdf")
        doc_id = f"DOC-{100 + i:04d}"                    # DOC-0101..0110 = policies
        cur.execute(
            "INSERT INTO documents (id, entity_type, entity_id, doc_type, file_ref) "
            "VALUES (%s,'policy',%s,'policy_pdf',%s)",
            (doc_id, p["id"], f"s3://{S3_BUCKET}/{key}"))
        citations[p["id"]] = {"document_id": doc_id, **cites}

    # Scans → MinIO + documents(pending) + field ground truth.
    gt_fields, expected_links = {}, {}
    for spec in scans:
        blob, ext = render_scan(spec)
        key = f"docs/{spec['id']}.{ext}"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=blob,
                      ContentType="application/pdf" if ext == "pdf" else "image/png")
        entity_id = spec["claim_id"] if spec["linked"] else ""
        cur.execute(
            "INSERT INTO documents (id, entity_type, entity_id, doc_type, file_ref) "
            "VALUES (%s,'claim',%s,%s,%s)",
            (spec["id"], entity_id, spec["doc_type"], f"s3://{S3_BUCKET}/{key}"))
        # Only invoices carry a plate/policy number the worker can link by;
        # an unlinked act (DOC-0016) is unlinkable by design — red flag (д).
        # The link heuristic picks the newest OPEN claim of the policy, so
        # for POL-0005 (two open claims) both are acceptable answers.
        if not spec["linked"] and spec["doc_type"] == "repair_invoice":
            acceptable = {"DOC-0004": ["CLM-0021", "CLM-0022"]}.get(
                spec["id"], [spec["claim_id"]])
            expected_links[spec["id"]] = acceptable
        if spec["doc_type"] == "repair_invoice":
            gt_fields[spec["id"]] = {
                "vendor_name": spec["vendor_name"],
                "invoice_number": spec["invoice_number"],
                "invoice_date": spec["invoice_date"],
                "vehicle_plate": spec["vehicle_plate"],
                "total_uah": None if spec["unreadable_total"] else spec["total_uah"],
                "positions_count": len(spec["positions"]),
            }
        elif spec["doc_type"] == "damage_photo_act":
            gt_fields[spec["id"]] = {"act_number": spec["act_number"],
                                     "act_date": spec["act_date"]}
        else:
            gt_fields[spec["id"]] = {"report_number": spec["report_number"],
                                     "report_date": spec["report_date"],
                                     "vehicle_plate": spec.get("vehicle_plate")}

    cur.execute("INSERT INTO audit_log (actor, action, detail) VALUES "
                "('seed','seed',%s)", (json.dumps({"claims": len(claims),
                                                   "documents": len(scans) + len(policies)}),))
    conn.commit()

    ground_truth = {
        "generated_at": str(TODAY),
        "invoices": gt_fields,
        "red_flags": red_flags,
        "citations": citations,
        "expected_links": expected_links,
        "missing_docs": {
            # KASKO claim needs invoice + damage act + police report.
            "CLM-0004": ["police_report"],           # has invoice DOC-0012 + act DOC-0020
            "CLM-0007": ["damage_photo_act", "police_report"],   # has invoices only
        },
        "invoice_coverage": {
            "DOC-0011": {"non_covered_positions": ["Тюнінг: спойлер та обвіс",
                                                   "Миття та полірування"],
                         "over_limit": True,     # 260 500 грн > POL-0001 limit 150k
                         "cite": {"policy": "POL-0001",
                                  "tuning_section": citations["POL-0001"]["tuning_excluded"],
                                  "washing_section": citations["POL-0001"]["washing_excluded"]}},
        },
    }
    out = HERE / "eval" / "ground_truth.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=1))

    cur.execute("SELECT count(*) FROM documents WHERE status='pending'")
    pending = cur.fetchone()[0]
    print(f"[seed] clients={len(clients)} policies={len(policies)} claims={len(claims)}")
    print(f"[seed] documents pending={pending} (20 scans + 10 policy PDFs)")
    print(f"[seed] ground truth → {out}")
    conn.close()


if __name__ == "__main__":
    main()
