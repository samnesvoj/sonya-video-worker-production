"""
llm_usage.py - every paid LLM (Gemini) call of a job -> one small JSON in S3 -> the accountant (dispatcher ledger).

Owner 2026-11-16: «пусть бухгалтер ещё пишет, сколько денег потратилось на Gemini».

  worker  process_job: begin(job_id, mode, work_dir)  -> SONYA_LLM_USAGE_FILE=<work_dir>/llm_usage.jsonl (inherited by
          subprocesses); llm_provider.chat_json appends one line per answered call (record()); finally flush() uploads
          <SONYA_LLM_USAGE_PREFIX>/<UTC day>/<job_id>-<rand>.json (also for failed jobs: the tokens were spent anyway).
  VPS     accounting/gemini.py imports those files into the ledger (idempotent) and prices them.

Only counts go out: provider, model, prompt version, tokens in/out, images, the provider's own cost if it reports one
(OpenRouter). Never prompts, answers or keys. Best effort: an accounting problem never fails or slows a job.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
ENV_FILE = "SONYA_LLM_USAGE_FILE"
ENV_JOB = "SONYA_LLM_USAGE_JOB"
ENV_MODE = "SONYA_LLM_USAGE_MODE"
DEFAULT_PREFIX = "worker-v2/llm-usage/"
_LOCK = threading.Lock()


def prefix() -> str:
    p = os.environ.get("SONYA_LLM_USAGE_PREFIX", DEFAULT_PREFIX).strip() or DEFAULT_PREFIX
    return p if p.endswith("/") else p + "/"


def count_images(messages: Any) -> int:
    """image_url parts in OpenAI-style messages (collages count as one image each - that is how they are billed)."""
    n = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            n += sum(1 for part in content if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"))
    return n


def record(*, provider: str, model: str, prompt_version: str, input_tokens: Optional[int], output_tokens: Optional[int],
           images: int = 0, cost_usd: Optional[float] = None, latency_ms: int = 0) -> None:
    """Append one answered call to the job's usage file. No file configured (tests, local runs) -> nothing."""
    path = os.environ.get(ENV_FILE)
    if not path:
        return
    line = {"ts": round(time.time(), 3), "provider": provider, "model": model, "prompt_version": prompt_version,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "images": int(images or 0),
            "cost_usd": cost_usd, "latency_ms": latency_ms}
    try:
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - accounting never breaks a job
        logger.warning("[llm_usage] record failed: %s", type(exc).__name__)


def begin(job_id: str, mode: str, work_dir: Path) -> None:
    os.environ[ENV_FILE] = str(Path(work_dir) / "llm_usage.jsonl")
    os.environ[ENV_JOB] = str(job_id)
    os.environ[ENV_MODE] = str(mode)


def read_calls(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    except FileNotFoundError:
        pass
    return out


def build_doc(job_id: str, mode: str, calls: List[Dict[str, Any]], worker: str = "") -> Dict[str, Any]:
    return {"v": 1, "job_id": str(job_id), "mode": str(mode), "worker": worker, "calls": calls}


def object_key(job_id: str, now: Optional[float] = None) -> str:
    day = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).strftime("%Y-%m-%d")
    return f"{prefix()}{day}/{job_id}-{secrets.token_hex(4)}.json"


def flush(upload=None) -> Optional[str]:
    """Upload the job's calls (if any) and forget the job. -> S3 key or None. upload(key, bytes) for tests."""
    path, job, mode = os.environ.pop(ENV_FILE, None), os.environ.pop(ENV_JOB, ""), os.environ.pop(ENV_MODE, "")
    if not path:
        return None
    calls = read_calls(path)
    if not calls:
        return None
    key = object_key(job)
    body = json.dumps(build_doc(job, mode, calls, os.environ.get("WORKER_ID", "")), ensure_ascii=False).encode("utf-8")
    try:
        if upload is None:
            from scripts.prod_s3_storage import upload_bytes
            upload_bytes(body, key, content_type="application/json")
        else:
            upload(key, body)
        logger.info("[llm_usage] job_id=%s calls=%d -> %s", job, len(calls), key)
        return key
    except Exception as exc:  # noqa: BLE001
        logger.warning("[llm_usage] upload failed job_id=%s: %s", job, type(exc).__name__)
        return None
