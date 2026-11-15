"""
clip_parallel.py - how many clips of ONE job are analysed + rendered at the same time.

Until 2026-11-16 always 2 (SONYA_CLIP_WORKERS): one clip analyses on the CPU while the other renders on NVENC. The clip phase
does not depend on the source length (10 clips of 30-60 s from 5 min or from 3 h), so for long sources it is most of the
job, and on a server that runs ONE job (a dedicated server, the warm with one user) 2 threads leave most cores idle.

auto (default): the CPU threads of OUR container slice (cgroup quota / affinity, scripts/ops/gpu_plan.cpu_threads) are
shared by the jobs running on this server right now (job folders sonya_<job>_p<pid>_* of live worker processes, see
gpu_worker._job_tmp_enter); every clip in flight gets SONYA_CLIP_THREADS (4) of them; never fewer than 2 (the old
behaviour), never more than SONYA_CLIP_WORKERS_MAX (6) or the number of clips. A number in SONYA_CLIP_WORKERS = fixed.
The clips themselves are identical: the same analysis, track and render per clip, only more of them at once.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_TAG = re.compile(r"^sonya_.+_p(\d+)_")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def jobs_root() -> Optional[Path]:
    """The folder holding every job folder of this server (TMPDIR of a job is <root>/sonya_<job>_p<pid>_xxx/t)."""
    t = os.environ.get("TMPDIR")
    if not t:
        return None
    p = Path(t)
    for cand in (p.parent, p):
        if _TAG.match(cand.name):
            return cand.parent
    return None


def active_jobs(root: Optional[Path] = None) -> int:
    root = root or jobs_root()
    if root is None:
        return 1
    pids = set()
    try:
        for d in root.iterdir():
            m = _TAG.match(d.name)
            if m and d.is_dir() and _alive(int(m.group(1))):
                pids.add(int(m.group(1)))
    except OSError:
        return 1
    return max(1, len(pids))


def cpu_threads() -> float:
    try:
        from scripts.ops.gpu_plan import cpu_threads as _ct
        n = float(_ct())
        if n > 0:
            return n
    except Exception:  # noqa: BLE001 - never block a render over this
        pass
    return float(os.cpu_count() or 2)


def clip_workers(n_clips: int, threads: Optional[float] = None, jobs: Optional[int] = None) -> int:
    v = os.environ.get("SONYA_CLIP_WORKERS", "auto").strip().lower()
    if v not in ("", "auto"):
        try:
            return max(1, int(v))
        except ValueError:
            pass
    per_clip = max(1.0, float(os.environ.get("SONYA_CLIP_THREADS", "4")))
    cap = max(2, int(os.environ.get("SONYA_CLIP_WORKERS_MAX", "6")))
    threads = cpu_threads() if threads is None else threads
    jobs = active_jobs() if jobs is None else max(1, jobs)
    n = int((threads / jobs) // per_clip)
    out = max(2, min(cap, n, max(1, n_clips)))
    logger.info("[clip_parallel] %d clip(s) at once (cpu %.1f threads / %d job(s) / %.0f per clip, %d clips)",
                out, threads, jobs, per_clip, n_clips)
    return out
