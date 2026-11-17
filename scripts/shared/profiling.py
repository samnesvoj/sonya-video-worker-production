"""
profiling.py -- per-job stage timings (2026-11-17, owner: «ускорь режимы, особенно Образовательный»).

Always on and cheap: `with stage("asr"):` records wall-clock seconds per named stage of the current job (one job per
worker process); gpu_worker logs one `[timing] job_id=... {...}` line at job_done so every production job shows where
its minutes went (download / normalize / asr / editor / crop analysis / render / previews / upload ...).

SONYA_PROFILE=1 (tests on Vast only): additionally every subprocess.run (ffmpeg / ffprobe ...) is timed by program,
and gpu_worker writes the stages + a cProfile of the main thread to S3 worker-v2/profile/<job_id>/.
Never raises, never changes a job.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from typing import Any, Dict, Iterator, List

_LOCK = threading.Lock()
_T0 = [time.monotonic()]
_STAGES: List[Dict[str, Any]] = []
_PROCS: Dict[str, List[float]] = {}
_HOOKED = [False]


def enabled_full() -> bool:
    return os.environ.get("SONYA_PROFILE", "").strip().lower() in ("1", "on", "true", "yes")


def begin() -> None:
    """Start of a job: forget the previous one."""
    with _LOCK:
        _T0[0] = time.monotonic()
        _STAGES.clear()
        _PROCS.clear()
    if enabled_full():
        _hook_subprocess()


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    t = time.monotonic()
    try:
        yield
    finally:
        add(name, time.monotonic() - t, start=t)


def add(name: str, seconds: float, start: float | None = None) -> None:
    try:
        with _LOCK:
            _STAGES.append({"name": str(name), "sec": round(float(seconds), 2),
                            "at": round((start if start is not None else time.monotonic() - seconds) - _T0[0], 2),
                            "thread": threading.current_thread().name})
    except Exception:  # noqa: BLE001
        pass


def summary() -> Dict[str, Any]:
    """{stage: total seconds} (a stage hit several times, e.g. per clip, is summed; `n` = how many) + subprocesses."""
    with _LOCK:
        tot: Dict[str, Dict[str, float]] = {}
        for s in _STAGES:
            d = tot.setdefault(s["name"], {"sec": 0.0, "n": 0})
            d["sec"] = round(d["sec"] + s["sec"], 2)
            d["n"] += 1
        procs = {k: {"sec": round(sum(v), 2), "n": len(v)} for k, v in _PROCS.items()}
        return {"wall": round(time.monotonic() - _T0[0], 2), "stages": tot, "procs": procs}


def events() -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_STAGES)


def _prog(args: Any) -> str:
    try:
        a = args if isinstance(args, (list, tuple)) else str(args).split()
        p = os.path.basename(str(a[0]))
        if p in ("ffmpeg", "ffprobe") and any("nvenc" in str(x) for x in a):
            p += ":nvenc"
        return p
    except Exception:  # noqa: BLE001
        return "?"


def _hook_subprocess() -> None:
    if _HOOKED[0]:
        return
    _HOOKED[0] = True
    orig = subprocess.run

    def run(*a, **kw):                # noqa: ANN001
        t = time.monotonic()
        try:
            return orig(*a, **kw)
        finally:
            try:
                with _LOCK:
                    _PROCS.setdefault(_prog(a[0] if a else kw.get("args")), []).append(time.monotonic() - t)
            except Exception:  # noqa: BLE001
                pass
    subprocess.run = run
