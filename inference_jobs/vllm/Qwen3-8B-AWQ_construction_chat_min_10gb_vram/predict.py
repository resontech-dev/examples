"""
Talk to the deployed Qwen3-8B endpoint — plain chat, or the BIM experiment.

Three modes:

    python predict.py "any prompt"              # plain chat (streamed)
    python predict.py --bim ["question"]        # Level A: BIM digest in context
    python predict.py --bim-tools ["question"]  # Level B: run_sql/get_element agent loop

Level A stuffs sample_data/bim_digest.json (an IfcOpenShell-style extract —
see tools/extract_ifc.py for producing one from a real IFC) into the context
and asks the question against it. Level B loads the same digest into an
in-memory SQLite database and gives the model two tools:

    run_sql(query)      — read-only SELECT over elements / psets / clashes
    get_element(guid)   — full record for one element

The agent loop prints every SQL the model runs, so you can audit whether an
answer came from data or from imagination — the core metric of the
experiment (share of exact answers + share of hallucinated GUIDs).

This script works unchanged against ANY vLLM chat job in this repo — point
.env at the gpt-oss-20b or Qwen3-Coder-30B endpoint to compare models on
the same question set.

Run
---
    pip install openai python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DIGEST_PATH = Path(os.getenv("BIM_DIGEST", HERE / "sample_data" / "bim_digest.json"))
DEFAULT_PROMPT = "Поясни різницю між IFC та DWG у трьох реченнях."
DEFAULT_BIM_QUESTION = "Скільки дверей на другому поверсі (Level 2)? Наведи їхні GUID."
MAX_TOOL_ROUNDS = 8


# ── Plumbing (same shape as every job in this repo) ──────────────────────────

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


def _client() -> OpenAI:
    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    return OpenAI(
        base_url=url,
        api_key=key,                             # → Authorization: Bearer (vLLM layer)
        default_headers={"X-API-Key": key},      # → platform serve-proxy layer
        timeout=float(os.getenv("PREDICT_TIMEOUT", "300")),
    )


def _pick_model(client: OpenAI) -> str:
    try:
        models = [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach the endpoint — not RUNNING yet, or URL wrong: {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(f"[predict] {exc.status_code} — check RESON_INFERENCE_API_KEY")
        raise
    if not models:
        sys.exit("[predict] /v1/models is empty — replica still loading weights?")
    model = os.getenv("MODEL_ID", models[0])
    print(f"[predict] endpoint: {client.base_url}")
    print(f"[predict] models:   {models}  → using {model!r}")
    return model


# ── BIM digest → context text (Level A) ──────────────────────────────────────

def _load_digest() -> dict:
    if not DIGEST_PATH.is_file():
        sys.exit(f"[predict] digest not found: {DIGEST_PATH} "
                 "(generate one with tools/extract_ifc.py)")
    return json.loads(DIGEST_PATH.read_text())


def _digest_text(d: dict) -> str:
    """Compact per-element lines — small enough to sit in a 32k context."""
    lines = [f"PROJECT: {d.get('project')}  SCHEMA: {d.get('schema')}",
             f"STOREYS: {', '.join(d.get('storeys', []))}", "", "ELEMENTS:"]
    for e in d["elements"]:
        q = (e.get("psets") or {}).get("BaseQuantities") or {}
        qty = " ".join(f"{k}={v}" for k, v in q.items())
        mats = ",".join(e.get("material") or []) or "-"
        lines.append(f"{e['guid']} | {e['class']} | {e.get('name')} | "
                     f"storey={e.get('storey')} | material={mats}"
                     + (f" | {qty}" if qty else ""))
    clashes = d.get("clashes") or []
    if clashes:
        lines += ["", "CLASHES (from ifcclash):"]
        for c in clashes:
            lines.append(f"{c['a_class']} {c.get('a_name')} ({c['a_guid']}) ↔ "
                         f"{c['b_class']} {c.get('b_name')} ({c['b_guid']}): {c.get('note')}")
    return "\n".join(lines)


LEVEL_A_SYSTEM = (
    "You are a BIM assistant. Answer ONLY from the model extract provided. "
    "Cite element GUIDs for every element you mention. If the extract does "
    "not contain the answer, say so explicitly — never invent elements, "
    "quantities, or GUIDs. Answer in the language of the question."
)


def run_level_a(client: OpenAI, model: str, question: str) -> None:
    digest = _digest_text(_load_digest())
    print(f"[predict] Level A — digest {len(digest) // 1024} KB in context")
    print(f"[predict] question: {question}")
    print("─" * 72)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LEVEL_A_SYSTEM},
            {"role": "user", "content": f"MODEL EXTRACT:\n{digest}\n\nQUESTION: {question}"},
        ],
        max_tokens=int(os.getenv("MAX_TOKENS", "768")),
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print()


# ── Level B: SQLite + tools agent loop ───────────────────────────────────────

DB_SCHEMA = """\
elements(guid TEXT PRIMARY KEY, class TEXT, name TEXT, storey TEXT, material TEXT)
psets(guid TEXT, pset TEXT, prop TEXT, value TEXT)   -- BaseQuantities live here (NetVolume, Width…)
clashes(a_guid TEXT, a_class TEXT, a_name TEXT, b_guid TEXT, b_class TEXT, b_name TEXT, note TEXT)"""


def _build_db(d: dict) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE elements(guid TEXT PRIMARY KEY, class TEXT, name TEXT, storey TEXT, material TEXT);"
        "CREATE TABLE psets(guid TEXT, pset TEXT, prop TEXT, value TEXT);"
        "CREATE TABLE clashes(a_guid TEXT, a_class TEXT, a_name TEXT,"
        "                     b_guid TEXT, b_class TEXT, b_name TEXT, note TEXT);"
    )
    for e in d["elements"]:
        conn.execute("INSERT INTO elements VALUES (?,?,?,?,?)",
                     (e["guid"], e["class"], e.get("name"),
                      e.get("storey"), ",".join(e.get("material") or [])))
        for pset, props in (e.get("psets") or {}).items():
            if not isinstance(props, dict):
                continue
            for prop, value in props.items():
                conn.execute("INSERT INTO psets VALUES (?,?,?,?)",
                             (e["guid"], pset, prop, str(value)))
    for c in d.get("clashes") or []:
        conn.execute("INSERT INTO clashes VALUES (?,?,?,?,?,?,?)",
                     (c["a_guid"], c["a_class"], c.get("a_name"),
                      c["b_guid"], c["b_class"], c.get("b_name"), c.get("note")))
    conn.commit()
    return conn


TOOLS = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SQL SELECT over the BIM database. "
                       "Returns up to 50 rows as JSON.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_element",
        "description": "Full record for one element by its IFC GlobalId: "
                       "attributes, all psets, and any clashes it appears in.",
        "parameters": {"type": "object",
                       "properties": {"guid": {"type": "string"}},
                       "required": ["guid"]},
    }},
]


def _exec_tool(conn: sqlite3.Connection, name: str, args: dict) -> str:
    """Execute a tool call. Errors are RETURNED as text so the model can
    read them and fix its SQL — don't crash the loop on a bad query."""
    try:
        if name == "run_sql":
            q = args["query"].strip().rstrip(";")
            if not q.lower().startswith("select"):
                return json.dumps({"error": "only SELECT queries are allowed"})
            cur = conn.execute(q)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(50)]
            return json.dumps({"rows": rows, "truncated_at": 50}, default=str)
        if name == "get_element":
            g = args["guid"]
            el = conn.execute("SELECT * FROM elements WHERE guid=?", (g,)).fetchone()
            if el is None:
                return json.dumps({"error": f"no element with guid {g!r}"})
            cols = [c[0] for c in conn.execute("SELECT * FROM elements LIMIT 0").description]
            psets = conn.execute("SELECT pset, prop, value FROM psets WHERE guid=?", (g,)).fetchall()
            clashes = conn.execute(
                "SELECT * FROM clashes WHERE a_guid=? OR b_guid=?", (g, g)).fetchall()
            return json.dumps({"element": dict(zip(cols, el)),
                               "psets": psets, "clashes": clashes}, default=str)
        return json.dumps({"error": f"unknown tool {name!r}"})
    except sqlite3.Error as e:
        return json.dumps({"error": f"SQL error: {e}"})


LEVEL_B_SYSTEM = (
    "You are a BIM assistant with SQL access to a building model database.\n"
    f"Schema:\n{DB_SCHEMA}\n\n"
    "Rules: answer ONLY from query results — run run_sql/get_element as many "
    "times as needed, then answer citing GUIDs from the results. psets.value "
    "is TEXT — CAST(value AS REAL) before SUM/AVG. If the data doesn't "
    "contain the answer, say so. Answer in the language of the question."
)


def run_level_b(client: OpenAI, model: str, question: str) -> None:
    conn = _build_db(_load_digest())
    n = conn.execute("SELECT COUNT(*) FROM elements").fetchone()[0]
    print(f"[predict] Level B — in-memory SQLite: {n} elements, tools: run_sql, get_element")
    print(f"[predict] question: {question}")
    print("─" * 72)

    messages = [{"role": "system", "content": LEVEL_B_SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS,
            max_tokens=int(os.getenv("MAX_TOKENS", "768")),
            temperature=0.1,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(msg.content)
            return
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            shown = args.get("query") or args.get("guid") or args
            print(f"  ⚙ {call.function.name}: {shown}")
            result = _exec_tool(conn, call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    print("[predict] stopped: tool-round limit reached without a final answer "
          f"({MAX_TOOL_ROUNDS} rounds) — is tool calling enabled on the endpoint?")


# ── Plain chat ───────────────────────────────────────────────────────────────

def run_chat(client: OpenAI, model: str, prompt: str) -> None:
    print(f"[predict] prompt: {prompt}")
    print("─" * 72)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a concise, helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print()


def main() -> None:
    args = sys.argv[1:]
    mode = "chat"
    if "--bim" in args:
        mode, args = "a", [a for a in args if a != "--bim"]
    if "--bim-tools" in args:
        mode, args = "b", [a for a in args if a != "--bim-tools"]

    client = _client()
    model = _pick_model(client)

    if mode == "a":
        run_level_a(client, model, args[0] if args else DEFAULT_BIM_QUESTION)
    elif mode == "b":
        run_level_b(client, model, args[0] if args else DEFAULT_BIM_QUESTION)
    else:
        run_chat(client, model, args[0] if args else DEFAULT_PROMPT)


if __name__ == "__main__":
    main()
