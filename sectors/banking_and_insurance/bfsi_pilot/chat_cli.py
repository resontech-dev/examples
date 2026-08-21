"""BFSI pilot — Module 5: CLI chat (test interface until the playground).

    python chat_cli.py               # REPL against AGENT_URL
    python chat_cli.py --verbose     # + tool-call trace with timings
    /file <path>                 # upload a document into the ingest queue
    /quit
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000")
VERBOSE = "--verbose" in sys.argv


def main() -> None:
    session_id = uuid.uuid4().hex[:8]
    client = httpx.Client(timeout=600.0)
    print(f"[chat] session {session_id} → {AGENT_URL}  (/file <path> to upload, /quit to exit)")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/file "):
            path = Path(line.split(" ", 1)[1]).expanduser()
            if not path.is_file():
                print(f"[chat] no such file: {path}")
                continue
            r = client.post(f"{AGENT_URL}/chat",
                            data={"session_id": session_id,
                                  "message": f"Я завантажив документ {path.name}."},
                            files={"file": (path.name, path.read_bytes())})
        else:
            r = client.post(f"{AGENT_URL}/chat",
                            json={"session_id": session_id, "message": line})
        if r.status_code != 200:
            print(f"[chat] HTTP {r.status_code}: {r.text[:300]}")
            continue
        data = r.json()
        if VERBOSE:
            for t in data.get("trace", []):
                arg = t["args"].get("query") or t["args"].get("document_id") or t["args"]
                print(f"  ⚙ {t['tool']} ({t['ms']} ms, {t['rows']} rows): {str(arg)[:100]}")
        print(f"agent> {data['reply']}\n")


if __name__ == "__main__":
    main()
