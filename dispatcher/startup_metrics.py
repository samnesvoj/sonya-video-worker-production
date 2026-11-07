"""
Start-up breakdown of one instance: from "dispatcher asked Vast" to "first user video starts processing".
Sources (all central): the dispatcher's own lifecycle journal (store.events) + the worker's timeline uploaded to S3
(scripts/ops/timeline.py, key fleet-metrics/<label>/<epoch>.json) + the first job's creation time from the queue.

Stages (seconds, None if a mark is missing):
  vast_provision_and_pull  created|resume_requested -> Vast reports running      (offer start, Docker image pull/extract, container create)
  container_boot           Vast running -> entrypoint_start                       (container runtime start; includes clock skew!)
  bootstrap                entrypoint_start -> bootstrap_done                     (verified release download by pinned SHA)
  preflight                bootstrap_done/run_worker_start -> preflight_done
  models                   models_start -> models_done                            (parallel verified fetch; 0-ish when cached)
  cuda_init                cuda_init_start -> cuda_init_done                      (torch import + CUDA context + matmul)
  readiness_total          preflight_done -> readiness_ok
  worker_prewarm           worker_started -> prewarm_done                         (imports + CUDA context in the worker process)
  wait_for_job             prewarm_done -> first_job_claimed
  claim_to_processing      first_job_claimed -> first_job_processing              (input download, status updates)
  user_wait                job queued -> first_job_processing                     (what the user feels)
CAVEAT: stages that mix the dispatcher clock and the instance clock (container_boot, user_wait) include clock skew; negative values are
clamped to 0 and flagged. These are MEASUREMENTS of real runs only once real instances exist - nothing here is a forecast.
"""
from __future__ import annotations

import json
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


def _first(marks: List[dict], name: str) -> Optional[float]:
    return next((m["t"] for m in marks if m.get("name") == name), None)


def _d(a: Optional[float], b: Optional[float], flags: List[str], label: str) -> Optional[float]:
    if a is None or b is None:
        return None
    v = b - a
    if v < 0:
        flags.append(f"{label}: negative ({v:.1f}s) - clock skew?")
        return 0.0
    return round(v, 2)


def breakdown(events: List[dict], marks: List[dict], job_queued_at: Optional[float] = None) -> Dict[str, Any]:
    ev = lambda *names: next((e["ts"] for e in events if e["event"] in names), None)
    flags: List[str] = []
    start = ev("created", "resume_requested")
    running = next((e["ts"] for e in events if e["event"] == "running" and (start is None or e["ts"] >= start)), None)
    t = lambda n: _first(marks, n)
    st = {
        "vast_provision_and_pull": _d(start, running, flags, "vast_provision_and_pull"),
        "container_boot": _d(running, t("entrypoint_start"), flags, "container_boot"),
        "bootstrap": _d(t("entrypoint_start"), t("bootstrap_done"), flags, "bootstrap"),
        "preflight": _d(t("run_worker_start"), t("preflight_done"), flags, "preflight"),
        "models": _d(t("models_start"), t("models_done"), flags, "models"),
        "cuda_init": _d(t("cuda_init_start"), t("cuda_init_done"), flags, "cuda_init"),
        "readiness_total": _d(t("preflight_done"), t("readiness_ok"), flags, "readiness_total"),
        "worker_prewarm": _d(t("worker_started"), t("prewarm_done"), flags, "worker_prewarm"),
        "wait_for_job": _d(t("prewarm_done"), t("first_job_claimed"), flags, "wait_for_job"),
        "claim_to_processing": _d(t("first_job_claimed"), t("first_job_processing"), flags, "claim_to_processing"),
        "user_wait": _d(job_queued_at, t("first_job_processing"), flags, "user_wait"),
    }
    ready = t("prewarm_done") or t("readiness_ok")
    st["request_to_ready"] = _d(start, ready, flags, "request_to_ready")
    first = next((m for m in marks if m.get("name") == "entrypoint_start"), {})
    return {"stages": st, "release_cached": str(first.get("release_cached", "")) == "yes", "flags": flags,
            "models_bytes": next((m.get("bytes_downloaded") for m in marks if m.get("name") == "models_done"), None),
            "cuda_ms": next((m.get("ms") for m in marks if m.get("name") == "cuda_init_done"), None),
            "gpu": next((m.get("gpu") for m in marks if m.get("name") == "cuda_init_done"), None)}


def summarize(rows: Iterable[Dict[str, Any]], kind: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Per-stage p50 / max over instances. kind='cold' (release not cached) or 'resume' (cached) filters the rows."""
    rows = [r for r in rows if kind is None or (r["release_cached"] == (kind == "resume"))]
    out: Dict[str, Dict[str, float]] = {}
    for stage in (rows[0]["stages"] if rows else {}):
        vals = sorted(r["stages"][stage] for r in rows if r["stages"].get(stage) is not None)
        if vals:
            out[stage] = {"n": len(vals), "p50": round(median(vals), 2), "max": vals[-1]}
    return out


def read_timeline_from_s3(client, bucket: str, worker_label: str) -> List[dict]:
    """Latest uploaded timeline for an instance label (read-only)."""
    resp = client.list_objects_v2(Bucket=bucket, Prefix=f"fleet-metrics/{worker_label}/")
    keys = sorted(o["Key"] for o in resp.get("Contents", []))
    if not keys:
        return []
    return json.loads(client.get_object(Bucket=bucket, Key=keys[-1])["Body"].read()).get("marks", [])
