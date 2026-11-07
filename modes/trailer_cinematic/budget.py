"""
budget.py — cost forecast + hard stop around scripts/shared/llm_provider.chat_json (no new client).

Before every call: forecast = input tokens (chars / CHARS_PER_TOKEN) × price_in
                             + max_output_tokens × price_out          (worst case: the whole output cap)
If spent + forecast > budget → BudgetExceeded BEFORE the request is sent.
After the call the provider-reported cost (OpenRouter usage.cost) is booked; if the provider does
not report it, the token-based estimate is booked instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# USD per token (OpenRouter /api/v1/models, 2026-11-03). Unknown model → the conservative fallback.
PRICES: Dict[str, tuple] = {
    "google/gemini-3.8-flash": (0.825e-6, 4.125e-6),
    "google/gemini-3.7-flash": (0.75e-6, 3.75e-6),
    "google/gemini-3.5-flash": (1.5e-6, 9.0e-6),
    "google/gemini-2.5-flash": (0.3e-6, 2.5e-6),
    # Timeweb AI Gateway. ESTIMATE, not an official price: timeweb.cloud/services/ai-gateway (2026-11-04) lists no price
    # for this model (only "see the panel"). = 2 × Google list price ($0.25 in / $1.50 out per 1M). Image / video /
    # audio tokens are billed as input tokens here (assumption). Replace with the panel price when known.
    "gemini/gemini-3.1-flash-lite": (0.5e-6, 3.0e-6),
}
ESTIMATED_PRICES = {"gemini/gemini-3.1-flash-lite"}
FALLBACK_PRICE = (2.0e-6, 12.0e-6)
IMAGE_TOKENS = 1100            # conservative per-image input estimate (Gemini via OpenRouter, ≤ 768 px frames)
VIDEO_TOKENS = 500             # conservative per-clip estimate (measured 60 video tokens/s at any resolution; clips ≤ 6 s)
DEFAULT_RUN_USD = 0.50         # Timeweb path: per-run hard limit
DEFAULT_DAY_USD = 2.00         # Timeweb path: per-day hard limit (all runs, ledger file)
CHARS_PER_TOKEN = 2.5          # Cyrillic + JSON is denser than English; under-counting chars is safer
# Owner limits are in roubles. Timeweb bills in RUB; there is no official USD↔RUB rate in the panel, so the conversion
# is deliberately pessimistic (a real rate of 80–95 ₽/$ means the real spend is LOWER than what we book).
RUB_PER_USD = float(os.environ.get("SONYA_RUB_PER_USD") or 100.0)
# Owner (2026-11-04): 100 ₽/video is an EMERGENCY ceiling for debugging only. Production target: ≤ 20 ₽ of Gemini per video.
PROD_VIDEO_LIMIT_RUB = float(os.environ.get("SONYA_TRAILER_VIDEO_LIMIT_RUB") or 20.0)
DEBUG_VIDEO_LIMIT_RUB = 100.0
# Forecast calibration (measured on «Волк с Уолл-стрит», 5 runs, 74 calls): answers took 0.15–0.5 of the output cap
# (analyst map ≈ 1.2k of 8000, writer ≈ 2k of 4000, director/judge ≈ 0.7–1k of 4000). The forecast used the whole cap
# → $0.71 against ≈ $0.04–0.17 booked per run (≈ 5–15× too high). The forecast now books FORECAST_OUT_FRACTION of
# the cap (≈ 2× above the measured maximum); the per-call guard before each request still assumes the FULL cap.
FORECAST_OUT_FRACTION = 0.4


def forecast_out(cap: int) -> int:
    return int(cap * FORECAST_OUT_FRACTION)


def rub(usd: float) -> float:
    return round(float(usd) * RUB_PER_USD, 2)


def usd_from_rub(r: float) -> float:
    return float(r) / RUB_PER_USD


class LLMStop(RuntimeError):
    """Stop the whole run now: no per-window fallbacks, no retries (budget, bad key, request too large)."""


class BudgetExceeded(LLMStop):
    pass


class FatalLLMError(LLMStop):
    pass


FRIENDLY = {
    "auth": "ключ отклонён провайдером (401/403) — проверьте ключ/баланс в Timeweb; повтор бессмыслен",
    "too_large": "запрос слишком большой для провайдера — уменьшите кадры/фрагменты",
    "bad_request": "провайдер отклонил запрос (400/422) — повтор бессмыслен",
}


class SchemaError(ValueError):
    pass


def price(model: str) -> tuple:
    return PRICES.get(model, FALLBACK_PRICE)


def _is_video(part: Any) -> bool:
    url = ((part.get("image_url") or {}).get("url") or "") if isinstance(part, dict) else ""
    return str(url).startswith("data:video/")


def count_videos(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list) for p in m["content"] if _is_video(p))


def _content_cost(content: Any) -> tuple:
    """(text chars, images) of one message; base64 image payloads are NOT counted as text."""
    if isinstance(content, str):
        return len(content), 0
    chars = imgs = 0
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "image_url":
            if not _is_video(part):
                imgs += 1
        elif isinstance(part, dict) and part.get("type") == "text":
            chars += len(part.get("text") or "")
        else:
            chars += len(json.dumps(part))
    return chars, imgs


def estimate_input_tokens(messages: List[Dict[str, Any]]) -> int:
    chars = imgs = 0
    for m in messages:
        c, i = _content_cost(m.get("content"))
        chars, imgs = chars + c, imgs + i
    return int(chars / CHARS_PER_TOKEN) + imgs * IMAGE_TOKENS + count_videos(messages) * VIDEO_TOKENS + 50


def forecast_usd(messages: List[Dict[str, Any]], max_output_tokens: int, model: str) -> float:
    pin, pout = price(model)
    return estimate_input_tokens(messages) * pin + max_output_tokens * pout


def unwrap_single(data: Any) -> Any:
    """Gemini via Timeweb answers response_format=json_object with [ {...} ]: one object in an array → the object.
    Anything else (empty array, several items, array of non-objects) is left alone for the strict validator."""
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    return data


def parse_json(text: str) -> Any:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return unwrap_single(json.loads(t))
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return unwrap_single(json.loads(m.group(0)))
            except ValueError:
                return None
    return None


class DailyLedger:
    """Per-day spend across runs (json file, resets when the local date changes). Holds numbers only.
    With a fixed `today` key (e.g. "video") it never resets: a cumulative per-video ledger (VideoLedger)."""

    def __init__(self, path: str, limit_usd: float, today: Optional[str] = None):
        self.path, self.limit = Path(path), float(limit_usd)
        self.today = today or time.strftime("%Y-%m-%d")

    def _read(self) -> Dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("date") == self.today and isinstance(d.get("spent_usd"), (int, float)):
                return d
        except (OSError, ValueError):
            pass
        return {"date": self.today, "spent_usd": 0.0, "calls": 0}

    def spent(self) -> float:
        return float(self._read()["spent_usd"])

    def add(self, usd: float) -> None:
        d = self._read()
        d["spent_usd"] = round(d["spent_usd"] + float(usd or 0.0), 6)
        d["calls"] = int(d.get("calls", 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        os.replace(tmp, self.path)


class VideoLedger(DailyLedger):
    """Spend on ONE source video over all its runs (owner: "≤ 100 ₽ per video, including what was already spent").
    `seed_usd` books money spent before the ledger existed (once, when the file is created)."""
    label = "video"

    def __init__(self, path: str, limit_rub: float, seed_usd: float = 0.0):
        super().__init__(path, usd_from_rub(limit_rub), today="video")
        self.limit_rub = float(limit_rub)
        if not self.path.is_file():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"date": "video", "spent_usd": round(float(seed_usd), 6), "calls": 0,
                                             "seed_usd": round(float(seed_usd), 6), "rub_per_usd": RUB_PER_USD}),
                                 encoding="utf-8")


def token_split(raw: Any) -> Dict[str, int]:
    """text / image / video / audio input tokens from usage.prompt_tokens_details (+ output), as reported."""
    u = (raw or {}).get("usage") if isinstance(raw, dict) else None
    u = u if isinstance(u, dict) else {}
    d = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
    out = {k[:-7]: int(v) for k, v in d.items() if k.endswith("_tokens") and isinstance(v, (int, float))}
    od = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
    for k, v in od.items():
        if k.endswith("_tokens") and isinstance(v, (int, float)):
            out["out_" + k[:-7]] = int(v)
    return out


class LLM:
    """chat_json + budget + telemetry. `validate(data) -> clean` raises SchemaError on a bad answer;
    then ONE repair request is sent with the validation error (also budget-guarded)."""

    def __init__(self, chat: Callable[..., Any], budget_usd: float, model: str, cache_dir: Optional[str] = None,
                 *, ledger: Any = None, chat_kwargs: Optional[Dict[str, Any]] = None,
                 reasoning: bool = True):
        self.chat, self.budget, self.model = chat, float(budget_usd), model
        # hard limits shared by all runs, checked BEFORE every call: the day ledger, optionally + the per-video ledger
        self.ledgers: List[DailyLedger] = [x for x in (ledger if isinstance(ledger, (list, tuple)) else [ledger]) if x is not None]
        self.ledger = self.ledgers[0] if self.ledgers else None
        self.chat_kwargs = dict(chat_kwargs or {})   # e.g. providers=[timeweb only], retries=2, max_request_bytes=…
        self.reasoning = reasoning            # False for providers that do not take OpenRouter's `reasoning` field
        self.calls: List[Dict[str, Any]] = []
        self.cache_dir = cache_dir            # validated answers are cached by (step, version, messages, temperature)

    def _cache_path(self, step: str, messages, prompt_version: str, temperature: float, max_out: int) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(json.dumps([self.model, prompt_version, temperature, max_out, messages],
                                      ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
        return Path(self.cache_dir) / f"llm-{step}-{h}.json"

    @property
    def spent(self) -> float:
        return round(sum(c.get("cost_usd") or 0.0 for c in self.calls), 6)

    def _guard(self, step: str, messages, max_out: int) -> float:
        fc = forecast_usd(messages, max_out, self.model)
        for lg in self.ledgers:
            if lg.spent() + fc > lg.limit:
                video = isinstance(lg, VideoLedger)
                self.calls.append({"step": step, "ok": False, "stopped": "video_budget" if video else "daily_budget",
                                   "forecast_usd": round(fc, 5), "ledger_spent_usd": round(lg.spent(), 5)})
                if video:
                    raise BudgetExceeded(f"STOP before {step}: this video {rub(lg.spent()):.2f} ₽ + forecast {rub(fc):.2f} ₽ "
                                         f"> limit {lg.limit_rub:.0f} ₽")
                raise BudgetExceeded(f"STOP before {step}: today ${lg.spent():.4f} + forecast ${fc:.4f} "
                                     f"> daily limit ${lg.limit:.2f}")
        if self.spent + fc > self.budget:
            self.calls.append({"step": step, "ok": False, "stopped": "budget", "forecast_usd": round(fc, 5),
                               "spent_usd": self.spent})
            raise BudgetExceeded(f"STOP before {step}: spent ${self.spent:.4f} + forecast ${fc:.4f} "
                                 f"> budget ${self.budget:.2f}")
        return fc

    def ask(self, step: str, messages: List[Dict[str, Any]], prompt_version: str,
            validate: Callable[[Any], Any], *, temperature: float = 0.2, max_out: int = 8000,
            effort: str = "low", timeout: float = 240.0) -> Any:
        msgs = list(messages)
        last = None
        cp = self._cache_path(step, messages, prompt_version, temperature, max_out)
        if cp is not None and cp.is_file():
            try:
                clean = validate(json.loads(cp.read_text(encoding="utf-8")))
                self.calls.append({"step": step, "ok": True, "cached": True, "cost_usd": 0.0, "wall_sec": 0.0})
                return clean
            except (OSError, ValueError, SchemaError):
                pass
        for attempt in range(3):                 # attempt 2 = schema repair; attempt 3 only after a transient network failure
            fc = self._guard(step if attempt == 0 else step + "_repair", msgs, max_out)
            t0 = time.monotonic()
            try:
                extra = {"max_tokens": max_out}
                if self.reasoning:
                    extra["reasoning"] = {"effort": effort}
                res = self.chat(msgs, prompt_version=prompt_version, opt_in=True, timeout=timeout,
                                temperature=temperature, extra=extra, **self.chat_kwargs)
            except Exception as e:                # noqa: BLE001 — recorded, retried once
                self.calls.append({"step": step, "attempt": attempt + 1, "ok": False,
                                   "error": f"{type(e).__name__}: {str(e)[:200]}", "forecast_usd": round(fc, 5),
                                   "provider_attempts": [{k: a.get(k) for k in ("status", "error", "latency_ms")}
                                                         for a in (getattr(e, "attempts", None) or [])][:3]})
                last = e
                import sys as _sys
                print(f"[llm-error] {step} attempt {attempt + 1}: {type(e).__name__}: {str(e)[:200]} "
                      f"attempts={[{k: x.get(k) for k in ('provider', 'status', 'error', 'latency_ms')} for x in (getattr(e, 'attempts', None) or [])][:3]}",
                      file=_sys.stderr, flush=True)
                kind = getattr(e, "kind", None)
                if kind in FRIENDLY:                 # a retry cannot help and would only repeat the request
                    raise FatalLLMError(f"{step}: {FRIENDLY[kind]}") from None
                time.sleep(3.0 * (attempt + 1))
                continue
            tel = res.telemetry()
            tel.pop("raw", None)
            tel.pop("attempts", None)
            pin, pout = price(self.model)
            est = (tel.get("input_tokens") or 0) * pin + (tel.get("output_tokens") or 0) * pout
            cost = tel.get("cost_usd")
            rec = dict(tel, tokens=token_split(getattr(res, "raw", None)), step=step, attempt=attempt + 1, forecast_usd=round(fc, 5),
                       cost_usd=float(cost) if cost is not None else round(est, 6),
                       cost_source="provider" if cost is not None else "estimate",
                       wall_sec=round(time.monotonic() - t0, 2))
            for lg in self.ledgers:
                lg.add(rec["cost_usd"])
            data = parse_json(res.text)
            try:
                if data is None:
                    raise SchemaError("answer is not valid JSON")
                clean = validate(data)
            except SchemaError as e:
                rec.update(ok=False, schema_error=str(e)[:300])
                self.calls.append(rec)
                last = e
                if attempt >= 1:
                    break                          # at most ONE schema repair
                msgs = list(messages) + [{"role": "assistant", "content": (res.text or "")[:6000]},
                                         {"role": "user", "content": f"Ответ не прошёл проверку схемы: {e}. "
                                                                     "Верни исправленный JSON целиком."}]
                continue
            rec["ok"] = True
            self.calls.append(rec)
            if cp is not None:
                try:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
            return clean
        raise RuntimeError(f"{step} failed after retries: {last}")

    def by_step(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in self.calls:
            if "cost_usd" in c:
                k = c["step"].split("_repair")[0]
                out[k] = round(out.get(k, 0.0) + (c["cost_usd"] or 0.0), 5)
        return out

    def tokens_by_modality(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.calls:
            for k, v in (c.get("tokens") or {}).items():
                out[k] = out.get(k, 0) + v
        return out

    def summary(self) -> Dict[str, Any]:
        ok = [c for c in self.calls if "input_tokens" in c]
        return {"calls": len(ok), "cached_calls": sum(1 for c in self.calls if c.get("cached")),
                "by_step": self.by_step(), "spent_usd": self.spent, "budget_usd": self.budget,
                "tokens_by_modality": self.tokens_by_modality(),
                "price_per_mtok_in_out": [round(price(self.model)[0] * 1e6, 3), round(price(self.model)[1] * 1e6, 3)],
                "price_is_estimate": self.model in ESTIMATED_PRICES,
                "input_tokens": sum(c.get("input_tokens") or 0 for c in ok),
                "output_tokens": sum(c.get("output_tokens") or 0 for c in ok),
                "providers": sorted({f"{c.get('provider')}/{c.get('model')}" for c in ok})}
