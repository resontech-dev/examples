"""Minimal predictor — echoes back whatever bytes you POST.

Implements the smallest possible version of the predictor contract:

    POST /predict  (raw bytes or text)
        -> {"echo": "<your text>", "bytes_received": N}
"""
from __future__ import annotations

import time
from typing import Any


class EchoPredictor:
    def __init__(self, prefix: str = "echo: ") -> None:
        self.prefix = prefix
        self.requests_served = 0

    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        text = data.decode("utf-8", errors="replace")
        self.requests_served += 1
        return {
            "echo": f"{self.prefix}{text}",
            "bytes_received": len(data),
            "request_index": self.requests_served,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "requests_served": self.requests_served}
