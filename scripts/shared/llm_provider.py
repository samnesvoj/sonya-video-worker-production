"""
llm_provider.py
===============
The ONE place SONYA modes call a hosted LLM (Gemini) for semantic work.

    chat_json(messages, prompt_version=..., opt_in=...) -> LLMResult

Providers, tried in order (each OpenAI-compatible /chat/completions):
  1. timeweb     TIMEWEB_GEMINI_BASE_URL + TIMEWEB_GEMINI_API_KEY
                 [+ TIMEWEB_GEMINI_MODEL, default gemini-2.5-flash]
  2. openrouter  OPENROUTER_API_KEY [+ SONYA_LLM_OPENROUTER_MODEL,
                 default google/gemini-3.8-flash]
  (Google direct can be added as a third entry with the same contract.)

No hidden paid calls: nothing is sent unless the caller passes
opt_in=True (modes map that from params["llm"] == "on") AND at least one
provider is configured. Every result carries provider, model,
prompt_version, tokens, cost (if the provider reports it) and latency;
failed attempts are listed too. Keys never appear in results or errors.

Timeweb: SONYA_LLM_TIMEWEB_ENABLED=false switches it off everywhere;
timeweb_provider() builds an explicit single provider (env, else macOS
Keychain group.sonya.timeweb-ai / api_key) for callers that must not fall
back to another provider. chat_json(retries=, max_request_bytes=) are
opt-in; failed attempts carry `kind` (auth/rate_limit/server/timeout/...).

STATUS: tested with a fake transport; the Timeweb endpoint
(gemini/gemini-3.1-flash-lite, OpenAI-compatible) was verified live on
2026-11-04 — see docs/TRAILER_TIMEWEB_REPORT.md.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

Transport = Callable[[str, Dict[str, str], bytes, float], Tuple[int, bytes]]


class LLMDisabled(RuntimeError):
    """Not opted in, or no provider configured. Not an error for callers."""


class LLMError(RuntimeError):
    def __init__(self, message: str, attempts: List[Dict[str, Any]], kind: str = "unknown"):
        super().__init__(message)
        self.attempts = attempts
        self.kind = kind        # last failure class: auth|rate_limit|server|timeout|network|bad_request|too_large|empty


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_version: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    latency_ms: int
    attempts: List[Dict[str, Any]]
    raw: Dict[str, Any]

    def telemetry(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        return d


TIMEWEB_BASE_URL = "https://api.timeweb.ai/v1"
TIMEWEB_MODEL = "gemini/gemini-3.1-flash-lite"
TIMEWEB_KEYCHAIN = ("group.sonya.timeweb-ai", "api_key")     # macOS Keychain (service, account)
_OFF = ("0", "false", "no", "off")


def timeweb_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Kill switch: SONYA_LLM_TIMEWEB_ENABLED=false|0|no|off disables the Timeweb provider everywhere.
    Unset → enabled (the previous behaviour)."""
    env = os.environ if env is None else env
    return str(env.get("SONYA_LLM_TIMEWEB_ENABLED", "")).strip().lower() not in _OFF


def keychain_secret(service: str, account: str) -> Optional[str]:
    """Read a generic password from the macOS Keychain at call time. The secret travels over stdout only —
    never argv, never logged; any failure returns None (stderr is not propagated)."""
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", service, "-a", account, "-w"],
                           capture_output=True, text=True, timeout=15)
    except Exception:  # noqa: BLE001 — no `security` binary, timeout…
        return None
    v = r.stdout.strip() if r.returncode == 0 else ""
    return v or None


def timeweb_provider(env: Optional[Dict[str, str]] = None, *, keychain: bool = True) -> Optional[ProviderConfig]:
    """Timeweb AI Gateway (Gemini) as ONE explicit provider: env TIMEWEB_GEMINI_* first, then the Keychain.
    Defaults: Base URL https://api.timeweb.ai/v1, model gemini/gemini-3.1-flash-lite. None if disabled or no key."""
    env = dict(os.environ) if env is None else env
    if not timeweb_enabled(env):
        return None
    key = env.get("TIMEWEB_GEMINI_API_KEY") or (keychain_secret(*TIMEWEB_KEYCHAIN) if keychain else None)
    if not key:
        return None
    return ProviderConfig("timeweb", (env.get("TIMEWEB_GEMINI_BASE_URL") or TIMEWEB_BASE_URL).rstrip("/"), key,
                          env.get("TIMEWEB_GEMINI_MODEL") or TIMEWEB_MODEL)


def configured_providers(env: Optional[Dict[str, str]] = None) -> List[ProviderConfig]:
    env = dict(os.environ) if env is None else env
    out: List[ProviderConfig] = []
    if timeweb_enabled(env) and env.get("TIMEWEB_GEMINI_BASE_URL") and env.get("TIMEWEB_GEMINI_API_KEY"):
        out.append(ProviderConfig("timeweb", env["TIMEWEB_GEMINI_BASE_URL"].rstrip("/"),
                                  env["TIMEWEB_GEMINI_API_KEY"],
                                  env.get("TIMEWEB_GEMINI_MODEL", "gemini-2.5-flash")))
    if env.get("OPENROUTER_API_KEY"):
        out.append(ProviderConfig("openrouter", "https://openrouter.ai/api/v1", env["OPENROUTER_API_KEY"],
                                  env.get("SONYA_LLM_OPENROUTER_MODEL", "google/gemini-3.8-flash"),
                                  {"usage": {"include": True}}))
    return out


def _urllib_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def certifi_transport() -> Transport:
    """Same as the default transport, but verifies TLS against the certifi bundle. Needed for api.timeweb.ai from
    python.org/venv builds whose OpenSSL store lacks the GlobalSign R46 root (CERTIFICATE_VERIFY_FAILED). Still full
    verification. Falls back to the default transport if certifi is not installed."""
    try:
        import ssl

        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return _urllib_transport

    def transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    return transport


def _content(raw: Dict[str, Any]) -> Optional[str]:
    """choices[0].message.content if the response has that shape, else None."""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    msg = choices[0].get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    return content if isinstance(content, str) else None


def _redact(raw: Dict[str, Any]) -> Dict[str, Any]:
    keep = {k: raw.get(k) for k in ("id", "model", "usage", "provider") if k in raw}
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        keep["finish_reason"] = choices[0].get("finish_reason")
    keep["content"] = (_content(raw) or "")[:20000]
    return keep


RETRYABLE = ("rate_limit", "server", "timeout", "network")


def classify_failure(status: Optional[int] = None, exc: Optional[BaseException] = None) -> str:
    """Failure class → decides retry and the message the user sees."""
    if exc is not None:
        e = exc.reason if isinstance(exc, urllib.error.URLError) and exc.reason is not None else exc
        if isinstance(e, (socket.timeout, TimeoutError)) or "timeout" in type(e).__name__.lower():
            return "timeout"
        return "network"
    if status in (401, 403):
        return "auth"
    if status == 413:
        return "too_large"
    if status == 429:
        return "rate_limit"
    if status is not None and status >= 500:
        return "server"
    if status == 200:
        return "empty"
    return "bad_request"


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret and secret in text else text


def chat_json(messages: List[Dict[str, str]], *, prompt_version: str, opt_in: bool,
              providers: Optional[List[ProviderConfig]] = None, temperature: float = 0.0,
              timeout: float = 90.0, transport: Transport = _urllib_transport,
              extra: Optional[Dict[str, Any]] = None, retries: int = 0, backoff: float = 2.0,
              max_request_bytes: Optional[int] = None,
              sleep: Callable[[float], None] = time.sleep) -> LLMResult:
    """Ask for a JSON answer. Tries providers in order; returns the first success.

    Additive options (defaults keep the old behaviour): `retries` — extra tries on the SAME provider for
    429 / 5xx / timeout / network errors (401/403/400 are never retried); `max_request_bytes` — refuse to send a
    larger body (no request is made). Every failed attempt carries `kind` (see classify_failure)."""
    if not opt_in:
        raise LLMDisabled("LLM not opted in for this job")
    providers = configured_providers() if providers is None else providers
    if not providers:
        raise LLMDisabled("no LLM provider configured")
    attempts: List[Dict[str, Any]] = []
    last_kind = "unknown"
    for p in providers:
        body = dict({"model": p.model, "messages": messages, "temperature": temperature,
                     "response_format": {"type": "json_object"}}, **p.extra_body, **(extra or {}))
        data = json.dumps(body).encode("utf-8")
        if max_request_bytes and len(data) > max_request_bytes:
            last_kind = "too_large"
            attempts.append({"provider": p.name, "model": p.model, "ok": False, "kind": "too_large",
                             "error": f"request {len(data)} bytes > limit {max_request_bytes}; not sent", "latency_ms": 0})
            continue
        for n in range(max(0, int(retries)) + 1):
            t0 = time.monotonic()
            try:
                status, payload = transport(f"{p.base_url}/chat/completions",
                                            {"Authorization": f"Bearer {p.api_key}",
                                             "Content-Type": "application/json"}, data, timeout)
            except Exception as e:  # network: message may embed URLs, never keys (key is a header)
                last_kind = classify_failure(exc=e)
                attempts.append({"provider": p.name, "model": p.model, "ok": False,
                                 "error": f"{type(e).__name__}", "kind": last_kind,
                                 "latency_ms": int((time.monotonic() - t0) * 1000)})
            else:
                latency = int((time.monotonic() - t0) * 1000)
                try:
                    raw = json.loads(payload or b"{}")
                except ValueError:
                    raw = {}
                if not isinstance(raw, dict):
                    raw = {}
                text = _content(raw) if status == 200 else None
                if status == 200 and text:
                    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
                    attempts.append({"provider": p.name, "model": p.model, "ok": True, "status": status,
                                     "latency_ms": latency})
                    return LLMResult(text=text, provider=p.name, model=raw.get("model") or p.model,
                                     prompt_version=prompt_version, input_tokens=usage.get("prompt_tokens"),
                                     output_tokens=usage.get("completion_tokens"),
                                     cost_usd=usage.get("cost"), latency_ms=latency, attempts=attempts,
                                     raw=_redact(raw))
                err = (raw.get("error") or {}).get("message") if isinstance(raw.get("error"), dict) else None
                last_kind = classify_failure(status=status)
                attempts.append({"provider": p.name, "model": p.model, "ok": False, "status": status,
                                 "kind": last_kind, "error": _scrub((err or "empty response")[:300], p.api_key),
                                 "latency_ms": latency})
            if last_kind not in RETRYABLE or n >= retries:
                break
            sleep(min(30.0, backoff * (2 ** n)))
    raise LLMError("all LLM providers failed" + (f" ({last_kind})" if last_kind != "unknown" else ""),
                   attempts, last_kind)
