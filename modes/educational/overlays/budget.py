"""
budget.py — hard rouble limits for the optional LLM analysis of a clip.

Forecast BEFORE every request (input estimate × price + the whole output cap × price, a pessimistic worst case);
over the limit → BudgetExceeded and no request. Spend is booked per video and in total in a small JSON ledger.
Prices are estimates (see modes/trailer_cinematic/budget.py: Timeweb publishes none), the rate is deliberately high.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

PRICE_USD_PER_TOKEN = {"gemini/gemini-3.1-flash-lite": (0.5e-6, 3.0e-6)}
FALLBACK_PRICE = (2.0e-6, 12.0e-6)
RUB_PER_USD = 100.0
CHARS_PER_TOKEN = 2.5        # Cyrillic + JSON: under-counting characters is the safe side
VIDEO_LIMIT_RUB = 5.0
TOTAL_LIMIT_RUB = 30.0
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def estimate_input_tokens(messages: List[Dict[str, Any]]) -> int:
    return int(sum(len(str(m.get("content", ""))) for m in messages) / CHARS_PER_TOKEN) + 50


def cost_rub(model: str, input_tokens: float, output_tokens: float) -> float:
    pin, pout = PRICE_USD_PER_TOKEN.get(model, FALLBACK_PRICE)
    return (input_tokens * pin + output_tokens * pout) * RUB_PER_USD


def forecast_rub(messages: List[Dict[str, Any]], max_output_tokens: int, model: str) -> float:
    return cost_rub(model, estimate_input_tokens(messages), max_output_tokens)


class Ledger:
    """Survives restarts when given a path (a restarted test run must not get a fresh allowance)."""

    def __init__(self, path: Optional[str] = None, *, video_limit_rub: float = VIDEO_LIMIT_RUB,
                 total_limit_rub: float = TOTAL_LIMIT_RUB):
        self.path = Path(path) if path else None
        self.video_limit_rub, self.total_limit_rub = float(video_limit_rub), float(total_limit_rub)
        self.data: Dict[str, Any] = {"total_rub": 0.0, "videos": {}, "calls": 0}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def spent(self, video_id: str) -> float:
        return float(self.data["videos"].get(video_id, 0.0))

    def check(self, video_id: str, forecast: float) -> None:
        if self.spent(video_id) + forecast > self.video_limit_rub:
            raise BudgetExceeded(f"video limit: {self.spent(video_id):.2f} + {forecast:.2f} > {self.video_limit_rub:.2f} RUB")
        if float(self.data["total_rub"]) + forecast > self.total_limit_rub:
            raise BudgetExceeded(f"total limit: {self.data['total_rub']:.2f} + {forecast:.2f} > {self.total_limit_rub:.2f} RUB")

    def book(self, video_id: str, rub: float) -> None:
        with _lock:
            self.data["videos"][video_id] = round(self.spent(video_id) + rub, 4)
            self.data["total_rub"] = round(float(self.data["total_rub"]) + rub, 4)
            self.data["calls"] = int(self.data["calls"]) + 1
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
                os.replace(tmp, self.path)
