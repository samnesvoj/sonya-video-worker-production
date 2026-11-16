"""
Gemini spend (owner 2026-11-16: «сколько денег потратилось на Gemini»). Kept APART from Vast: own ledger table, own block
in the reports, and only the separate line «Всего расходы = Vast + Gemini» adds the two. No LLM here.

Source: the worker writes one JSON per job with every answered Gemini call (scripts/shared/llm_usage.py) to
S3 <prefix><UTC day>/<job_id>-<rand>.json; import_objects() copies new files into the ledger table llm_usage
(idempotent: one row per call, keyed by the S3 object + call index).

Price: Timeweb bills Gemini in RUB, but publishes no per-token price for gemini-3.1-flash-lite (only «see the panel»),
so the default is an ESTIMATE = 2 x Google's list price ($0.25 / $1.50 per 1M) at ~80 RUB/$ -> 40 RUB in / 240 RUB out per
1M tokens - the same numbers the worker's 1.5 RUB/video budget uses (scripts/shared/llm_editor.py). Images are billed as
input tokens (already inside prompt_tokens). A call whose provider reports its own cost (OpenRouter: usage.cost, USD)
is booked at that cost x the CBR rate. Env: SONYA_GEMINI_RUB_PER_M_IN / _OUT, SONYA_GEMINI_PRICE_OFFICIAL=1 once the
numbers come from the Timeweb panel (drops the «оценка» mark).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Callable, Dict, Iterable, List, Optional

from accounting.money import D, UTC, msk_date_of

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
  usage_key TEXT PRIMARY KEY, job_id TEXT, mode TEXT, day TEXT NOT NULL, ts REAL, provider TEXT, model TEXT,
  prompt_version TEXT, input_tokens INTEGER, output_tokens INTEGER, images INTEGER NOT NULL DEFAULT 0,
  cost_usd TEXT, source TEXT, imported_at REAL);
CREATE INDEX IF NOT EXISTS ix_llm_usage_day ON llm_usage(day);
CREATE TABLE IF NOT EXISTS llm_usage_objects (obj_key TEXT PRIMARY KEY, calls INTEGER NOT NULL, imported_at REAL);
"""
DEFAULT_PREFIX = "worker-v2/llm-usage/"


@dataclass(frozen=True)
class Prices:
    rub_in_per_m: Decimal = Decimal("40")
    rub_out_per_m: Decimal = Decimal("240")
    estimate: bool = True

    @classmethod
    def from_env(cls, env=None) -> "Prices":
        env = os.environ if env is None else env
        return cls(D(env.get("SONYA_GEMINI_RUB_PER_M_IN") or "40"), D(env.get("SONYA_GEMINI_RUB_PER_M_OUT") or "240"),
                   str(env.get("SONYA_GEMINI_PRICE_OFFICIAL", "")).lower() not in ("1", "true", "yes", "on"))


def ensure_schema(ledger) -> None:
    with ledger.lock:
        ledger.c.executescript(SCHEMA)


# ------------------------------------------------------------------------------------------------ import
def rows_from_doc(obj_key: str, doc: dict) -> List[dict]:
    out = []
    for i, c in enumerate(doc.get("calls") or []):
        if not isinstance(c, dict):
            continue
        ts = float(c.get("ts") or 0)
        out.append({"usage_key": f"{obj_key}#{i}", "job_id": str(doc.get("job_id") or ""), "mode": str(doc.get("mode") or ""),
                    "day": msk_date_of(ts).isoformat() if ts else None, "ts": ts, "provider": str(c.get("provider") or ""),
                    "model": str(c.get("model") or ""), "prompt_version": str(c.get("prompt_version") or ""),
                    "input_tokens": _int(c.get("input_tokens")), "output_tokens": _int(c.get("output_tokens")),
                    "images": _int(c.get("images")) or 0,
                    "cost_usd": str(c["cost_usd"]) if c.get("cost_usd") not in (None, "") else None})
    return [r for r in out if r["day"]]


def _int(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def import_objects(ledger, keys: Iterable[str], get: Callable[[str], bytes], now: Optional[float] = None) -> int:
    """New S3 objects -> ledger rows. Already imported objects are skipped. -> calls imported."""
    ensure_schema(ledger)
    now = time.time() if now is None else now
    n = 0
    for key in keys:
        if ledger.c.execute("SELECT 1 FROM llm_usage_objects WHERE obj_key=?", (key,)).fetchone():
            continue
        try:
            doc = json.loads(get(key) or b"{}")
        except ValueError:
            doc = {}
        rows = rows_from_doc(key, doc if isinstance(doc, dict) else {})
        with ledger.lock:
            for r in rows:
                ledger.c.execute("INSERT OR IGNORE INTO llm_usage(usage_key, job_id, mode, day, ts, provider, model, prompt_version, "
                                 "input_tokens, output_tokens, images, cost_usd, source, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (r["usage_key"], r["job_id"], r["mode"], r["day"], r["ts"], r["provider"], r["model"], r["prompt_version"],
                                  r["input_tokens"], r["output_tokens"], r["images"], r["cost_usd"], "s3", now))
            ledger.c.execute("INSERT OR IGNORE INTO llm_usage_objects(obj_key, calls, imported_at) VALUES (?,?,?)", (key, len(rows), now))
        n += len(rows)
    return n


def s3_day_prefixes(prefix: str, now: float, days: int = 3) -> List[str]:
    """Worker folders are UTC days; an MSK day spans two of them -> today + the previous days."""
    today = datetime.fromtimestamp(now, UTC).date()
    return [f"{prefix}{(today - timedelta(days=i)).isoformat()}/" for i in range(days)]


def import_from_s3(ledger, client, bucket: str, prefix: str = DEFAULT_PREFIX, now: Optional[float] = None, days: int = 3) -> int:
    """boto3 client: list today's / recent day folders, import the new objects."""
    now = time.time() if now is None else now
    keys: List[str] = []
    for p in s3_day_prefixes(prefix, now, days):
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": p, **({"ContinuationToken": token} if token else {})}
            resp = client.list_objects_v2(**kw)
            keys += [o["Key"] for o in resp.get("Contents") or [] if o["Key"].endswith(".json")]
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    return import_objects(ledger, keys, lambda k: client.get_object(Bucket=bucket, Key=k)["Body"].read(), now)


# ------------------------------------------------------------------------------------------------ numbers
def call_rub(row, prices: Prices, rate: Optional[Decimal]) -> Decimal:
    """One call in RUB (exact Decimal). Provider-reported USD cost wins when a rate is known."""
    if row["cost_usd"] not in (None, "") and rate is not None:
        return D(row["cost_usd"]) * rate
    return (D(row["input_tokens"] or 0) * prices.rub_in_per_m + D(row["output_tokens"] or 0) * prices.rub_out_per_m) / Decimal(1_000_000)


def stats(ledger, first: date, last: date, prices: Prices, rate: Optional[Decimal] = None) -> dict:
    """MSK days first..last inclusive -> rub (exact), requests, jobs (distinct), tokens, images, calls without token counts."""
    ensure_schema(ledger)
    out = {"rub": Decimal(0), "requests": 0, "jobs": 0, "in": 0, "out": 0, "images": 0, "no_tokens": 0, "by_mode": {}}
    jobs = set()
    for r in ledger.c.execute("SELECT * FROM llm_usage WHERE day BETWEEN ? AND ?", (first.isoformat(), last.isoformat())):
        rub = call_rub(r, prices, rate)
        out["rub"] += rub
        out["requests"] += 1
        out["in"] += r["input_tokens"] or 0
        out["out"] += r["output_tokens"] or 0
        out["images"] += r["images"] or 0
        if r["input_tokens"] is None and r["cost_usd"] in (None, ""):
            out["no_tokens"] += 1
        if r["job_id"]:
            jobs.add(r["job_id"])
        m = r["mode"] or "—"
        out["by_mode"][m] = out["by_mode"].get(m, Decimal(0)) + rub
    out["jobs"] = len(jobs)
    out["per_video"] = out["rub"] / out["jobs"] if out["jobs"] else None
    return out


def periods(ledger, day: date, prices: Prices, rate: Optional[Decimal] = None) -> Dict[str, dict]:
    """day / the day before / month to day (MSK)."""
    return {"day": stats(ledger, day, day, prices, rate), "prev": stats(ledger, day - timedelta(days=1), day - timedelta(days=1), prices, rate),
            "month": stats(ledger, day.replace(day=1), day, prices, rate)}


def timeweb_balance_text(token: str, get=None) -> Optional[str]:
    """GET api.timeweb.cloud/api/v1/account/finances -> «1 234,56 ₽» (None on any problem). The token is never logged."""
    if not token:
        return None
    try:
        if get is None:
            import urllib.request
            req = urllib.request.Request("https://api.timeweb.cloud/api/v1/account/finances",
                                         headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:   # noqa: S310 - fixed https URL
                doc = json.loads(r.read())
        else:
            doc = get()
        fin = doc.get("finances") or {}
        bal = fin.get("balance")
        if bal is None:
            return None
        from accounting import tgformat as tf
        return tf.esc(tf.num(D(bal), 2)) + " ₽"
    except Exception:  # noqa: BLE001
        return None
