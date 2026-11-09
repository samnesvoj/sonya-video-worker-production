"""
gpu_plan.py - which GPUs this server has, how many videos it may process at once and which worker process runs on which card.
Stdlib only (the dispatcher imports `capacity` / `Budget` too, so the worker and the fleet use ONE formula).

  * detect()          nvidia-smi -> [Gpu(index, uuid, name, vram_total_mb, vram_used_mb, driver)]
  * capacity()        parallel videos allowed by VRAM of EACH card, CPU threads and RAM (measured per video, load test
                      2026-11-09, SPEED_REPORT.md: ~1.5 GB VRAM, ~3.5 GB RAM, ~3-4 CPU threads) with a safety margin
  * assign()          N workers -> cards, balanced by each card's capacity (equal cards: round robin, difference <= 1)
  * build_plan()      the whole decision, written by readiness to $SONYA_STATE_DIR/gpu_plan.json
  * nvenc_session_limit()  GeForce drivers cap simultaneous NVENC sessions; data-center cards (A40, L40S, A6000 ...) do not

Every worker process is pinned to ONE card with CUDA_VISIBLE_DEVICES=<GPU uuid> (+ CUDA_DEVICE_ORDER=PCI_BUS_ID): torch,
ctranslate2 AND ffmpeg's h264_nvenc (it opens its CUDA device through the same driver API) then see only that card as device 0,
so no code inside a job has to know about multi-GPU. With exactly ONE visible card nothing is pinned (byte-identical to before).

CLI (deploy/vast/run_worker.sh, docs, dry runs):
  python -m scripts.ops.gpu_plan probe              CUDA context + matmul on the (pinned) card -> one JSON line   (readiness)
  python -m scripts.ops.gpu_plan workers            one line per worker: "<slot> KEY=VAL ..." from the plan file  (run_worker)
  python -m scripts.ops.gpu_plan simulate --gpus 2xA40:46068 --threads 32 --ram-gb 128 [--concurrency auto] [--broken 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

MAX_WORKERS = 32                      # run_worker.sh accepts 1..32 worker processes (5..32: big-server load tests only)
AUTO_MAX = 16                         # 'auto' never plans more than this (VRAM-free cases, no GPU check)


# ── per-video resource budget ──────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Budget:
    """What ONE video (one worker process) needs, with margin. Measured (RTX 3060, kishlak_15m virality, 1..4 at once):
    VRAM 2.2 / 3.6 / 5.6 / 6.3 GB, container RAM 4.9 / 7.9 / 12.6 / 13.9 GB, CPU threads busy 8 / 10 / 11.1 / 11.5 of 12."""
    vram_per_job_mb: int = 2000       # measured ~1.5 GB per extra video (GigaAM + YOLO + NVENC, own CUDA context) + 33 %
    vram_reserve_mb: int = 1024       # per card: driver, fragmentation, a peak of the first video (2.2 GB alone)
    ram_per_job_mb: int = 4096        # measured ~3.5 GB per extra video + 17 % (= the dispatcher's 4 GB per video rule)
    ram_reserve_mb: int = 4096        # system, health daemon, page cache for the input video
    threads_per_job: float = 4.0      # ~3-4 threads per video; the CPU is what limits (4 at once on 12 threads: +60 % each)
    jobs_per_gpu_max: int = 8         # NVENC engine + SM time per card; the L40S load test (phase G) measures 1..12
    max_total: int = AUTO_MAX

    ENV = {"vram_per_job_mb": "SONYA_VRAM_PER_JOB_MB", "vram_reserve_mb": "SONYA_VRAM_RESERVE_MB",
           "ram_per_job_mb": "SONYA_RAM_PER_JOB_MB", "ram_reserve_mb": "SONYA_RAM_RESERVE_MB",
           "threads_per_job": "SONYA_CPU_THREADS_PER_JOB", "jobs_per_gpu_max": "SONYA_JOBS_PER_GPU_MAX"}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Budget":
        e = os.environ if env is None else env
        kw: Dict[str, Any] = {}
        for f in fields(cls):
            key = cls.ENV.get(f.name)
            if key and e.get(key, "").strip():
                try:
                    v = float(e[key]) if f.type in ("float", float) else int(e[key])
                except ValueError as exc:
                    raise ValueError(f"{key} must be a number") from exc
                if v <= 0:
                    raise ValueError(f"{key} must be > 0")
                kw[f.name] = v
        return cls(**kw)


def capacity(vram_mb_per_gpu: Sequence[int], cpu_threads: float, ram_mb: int, budget: Budget = Budget()) -> Dict[str, Any]:
    """Parallel videos this hardware carries. vram_mb_per_gpu: usable (free) VRAM of every HEALTHY card; [] = no GPU
    (CPU-only / GPU check skipped: VRAM does not limit). -> {"total", "per_gpu", "by_vram", "by_cpu", "by_ram", "limit"}.
    total is never below 1 (one video at a time is the old, always-allowed behaviour) and never above max_total."""
    per_gpu = [max(0, min(budget.jobs_per_gpu_max, (int(v) - budget.vram_reserve_mb) // budget.vram_per_job_mb))
               for v in vram_mb_per_gpu]
    by_vram = sum(per_gpu) if vram_mb_per_gpu else budget.max_total
    by_cpu = int(math.floor(cpu_threads / budget.threads_per_job + 1e-9)) if cpu_threads else budget.max_total
    by_ram = max(0, (int(ram_mb) - budget.ram_reserve_mb) // budget.ram_per_job_mb) if ram_mb else budget.max_total
    limits = {"vram": by_vram, "cpu": by_cpu, "ram": by_ram, "max": budget.max_total}
    limit = min(limits, key=lambda k: (limits[k], ("vram", "cpu", "ram", "max").index(k)))
    return {"total": max(1, min(limits.values())), "per_gpu": per_gpu, "by_vram": by_vram, "by_cpu": by_cpu,
            "by_ram": by_ram, "limit": limit}


def assign(n_workers: int, gpu_caps: Sequence[int]) -> List[int]:
    """Worker k (0-based) -> position in gpu_caps. Each next worker goes to the card with the lowest load ratio
    (assigned / capacity), ties -> fewer assigned, then the lower index. Equal cards -> round robin, so no card idles while
    another carries two more; a card with twice the VRAM gets twice the workers. Zero capacities count as 1."""
    if not gpu_caps:
        return []
    caps = [max(1, int(c)) for c in gpu_caps]
    load = [0] * len(caps)
    out: List[int] = []
    for _ in range(n_workers):
        g = min(range(len(caps)), key=lambda i: (load[i] / caps[i], load[i], i))
        load[g] += 1
        out.append(g)
    return out


# ── hardware detection ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Gpu:
    index: int
    uuid: str
    name: str
    vram_total_mb: int
    vram_used_mb: int = 0
    driver: str = ""
    cuda_ok: Optional[bool] = None       # set by readiness (probe in a child process pinned to this card)
    nvenc_ok: Optional[bool] = None      # real 0.5 s h264_nvenc encode on this card; None = not probed (no h264_nvenc in ffmpeg)
    cuda_ms: Optional[int] = None
    nvenc_error: str = ""                # last ffmpeg line of a failed NVENC probe (the card still works, clips on the CPU)
    error: str = ""                      # why the card is not used (short, no env values)

    @property
    def consumer(self) -> bool:
        return is_consumer(self.name)

    @property
    def vram_free_mb(self) -> int:
        return max(0, self.vram_total_mb - self.vram_used_mb)

    @property
    def usable(self) -> bool:
        return self.cuda_ok is not False and not self.error

    def short(self) -> str:
        return f"GPU {self.index} {self.name} {self.vram_total_mb} MB ({self.uuid[:12]}...)"


QUERY = "index,uuid,name,memory.total,memory.used,driver_version"


def parse_smi(out: str) -> List[Gpu]:
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        num = lambda s: int(float(re.sub(r"[^0-9.]", "", s) or 0))   # noqa: E731  "46068 MiB" / "46068"
        try:
            gpus.append(Gpu(index=int(parts[0]), uuid=parts[1], name=parts[2], vram_total_mb=num(parts[3]),
                            vram_used_mb=num(parts[4]), driver=parts[5]))
        except ValueError:
            continue
    return gpus


def detect(nvidia_smi: str = "nvidia-smi", run: Callable[..., Any] = subprocess.run) -> List[Gpu]:
    """Every card the container sees (Vast already limits the container to the rented cards). [] on any error."""
    try:
        r = run([nvidia_smi, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_smi(r.stdout) if r.returncode == 0 else []


def is_consumer(name: str) -> bool:
    """GeForce / TITAN = consumer driver limits (NVENC sessions). RTX A4000/A6000, A40, L40S, L4, Tesla, Quadro: data-center."""
    n = name.lower()
    return "geforce" in n or "titan" in n


def nvenc_session_limit(gpus: Sequence[Gpu], env: Mapping[str, str] | None = None) -> int:
    """Simultaneous NVENC sessions allowed (0 = no limit). SONYA_NVENC_MAX_SESSIONS overrides (0 = unlimited).
    GeForce drivers cap it PER SYSTEM (not per card): 3 before R530, 5 from R530, 8 from R550 (NVIDIA video codec support
    matrix). Data-center / professional cards have no cap. Conservative on purpose: an encode beyond the cap is not an
    error - scripts/shared/nvenc_guard.py renders it with libx264 on the CPU (and retries a refused session on the CPU)."""
    e = os.environ if env is None else env
    raw = (e.get("SONYA_NVENC_MAX_SESSIONS") or "").strip()
    if raw:
        v = int(raw)
        if v < 0:
            raise ValueError("SONYA_NVENC_MAX_SESSIONS must be >= 0")
        return v
    consumer = [g for g in gpus if g.consumer]
    if not consumer:
        return 0
    try:
        major = min(int(str(g.driver).split(".")[0]) for g in consumer)
    except ValueError:
        major = 0
    return 8 if major >= 550 else 5 if major >= 530 else 3


def cpu_threads(root: str = "/sys/fs/cgroup") -> float:
    """CPU threads of OUR slice: cgroup quota (cpu.max / cfs_quota) or the affinity mask, whichever is smaller."""
    n = float(len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else float(os.cpu_count() or 1)
    try:
        p = Path(root, "cpu.max")
        if p.is_file():
            q, per = p.read_text().split()[:2]
            if q != "max":
                n = min(n, int(q) / int(per))
        elif Path(root, "cpu", "cpu.cfs_quota_us").is_file():
            q = int(Path(root, "cpu", "cpu.cfs_quota_us").read_text())
            if q > 0:
                n = min(n, q / int(Path(root, "cpu", "cpu.cfs_period_us").read_text()))
    except (OSError, ValueError):
        pass
    return round(n, 2)


def ram_mb(root: str = "/sys/fs/cgroup") -> int:
    """RAM of OUR container: min(physical, cgroup memory limit)."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 0
    for f in (Path(root, "memory.max"), Path(root, "memory", "memory.limit_in_bytes")):
        try:
            v = f.read_text().strip()
            if v.isdigit() and 0 < int(v) < (total or 1 << 62):
                total = int(v)
            break
        except OSError:
            continue
    return int(total / 2**20)


# ── the plan ───────────────────────────────────────────────────────────────────────────────────────────────────────
def resolve_concurrency(env: Mapping[str, str], cap_total: int) -> tuple:
    """SONYA_WORKER_CONCURRENCY: unset -> 1 (old behaviour); 1..32 -> exactly that (operator / dispatcher override);
    'auto' -> what the hardware carries (capacity), capped by SONYA_WORKER_CONCURRENCY_MAX. -> (n, source)."""
    raw = (env.get("SONYA_WORKER_CONCURRENCY") or "1").strip().lower()
    if raw == "auto":
        cap_raw = (env.get("SONYA_WORKER_CONCURRENCY_MAX") or str(AUTO_MAX)).strip()
        if not cap_raw.isdigit() or not 1 <= int(cap_raw) <= MAX_WORKERS:
            raise ValueError(f"SONYA_WORKER_CONCURRENCY_MAX must be 1..{MAX_WORKERS} (got {cap_raw})")
        return max(1, min(cap_total, int(cap_raw))), "auto"
    if not raw.isdigit() or not 1 <= int(raw) <= MAX_WORKERS:
        raise ValueError(f"SONYA_WORKER_CONCURRENCY must be 1..{MAX_WORKERS} or auto (got {raw})")
    return int(raw), "env"


# ── CPU window per worker (2x A40 load test 2026-11-09) ──────────────────────────────────────────────────────────────
# ffmpeg (decoder, filters, swscale, libx264), OpenCV and torch size their thread pools by the CPUs they may run on. On a
# 128-thread EPYC every ffmpeg starts hundreds of threads; with 16+ workers the container ran out of threads ("Resource
# temporarily unavailable", Python "can't start new thread"): clips were lost (4090 and A40 at 16), uploads and whole jobs
# failed (A40 at 24). Each worker therefore gets its own window of CPUs (sched affinity, inherited by every child), so its
# pools are sized like on a normal 16-32 thread server. Windows overlap (a worker in a heavy phase can still use idle CPUs
# of its neighbours). Small servers (< CPU_WINDOW_MIN_HOST CPUs) and one worker: nothing changes.
CPU_WINDOW_MIN = 16                   # never fewer CPUs per worker than a 3060 box has (12-16 threads)
CPU_WINDOW_MIN_HOST = 32


def cpu_windows(cpus: Sequence[int], n_workers: int, env: Mapping[str, str] | None = None) -> List[Optional[List[int]]]:
    """Worker k -> its CPU list (None = not pinned). SONYA_CPU_WINDOW: 0 = off, N = N CPUs per worker, unset = automatic:
    max(16, 2 x CPUs / workers), i.e. every CPU is in about two windows."""
    e = os.environ if env is None else env
    cpus = sorted(int(c) for c in cpus)
    raw = (e.get("SONYA_CPU_WINDOW") or "").strip()
    if raw == "0" or n_workers <= 1 or len(cpus) < CPU_WINDOW_MIN_HOST:
        return [None] * max(0, n_workers)
    w = int(raw) if raw.isdigit() else max(CPU_WINDOW_MIN, math.ceil(2 * len(cpus) / n_workers))
    if w >= len(cpus):
        return [None] * n_workers
    out: List[Optional[List[int]]] = []
    for k in range(n_workers):
        start = (k * len(cpus)) // n_workers
        out.append(sorted(cpus[(start + i) % len(cpus)] for i in range(w)))
    return out


def cpu_list(cpus: Sequence[int]) -> str:
    """[0,1,2,3,8,9] -> "0-3,8-9" (taskset / cpuset syntax)."""
    parts: List[str] = []
    xs = sorted(set(int(c) for c in cpus))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        parts.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(parts)


def parse_cpu_list(text: str) -> List[int]:
    out: List[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def allowed_cpus() -> List[int]:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))


def build_plan(gpus: Sequence[Gpu], env: Mapping[str, str] | None = None, *, threads: Optional[float] = None,
               ram: Optional[int] = None, gpu_check: bool = True, cpus: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """gpus: every card nvidia-smi listed, with readiness probe results filled in. gpu_check=False: GPU check skipped
    (local smoke / CPU tests): workers are not pinned and VRAM does not limit."""
    e = dict(os.environ if env is None else env)
    budget = Budget.from_env(e)
    threads = float(e.get("SONYA_CPU_THREADS") or (threads if threads is not None else cpu_threads()))
    ram = int(e.get("SONYA_RAM_MB") or (ram if ram is not None else ram_mb()))
    usable = [g for g in gpus if g.usable] if gpu_check else []
    cap = capacity([g.vram_free_mb for g in usable], threads, ram, budget)
    n, source = resolve_concurrency(e, cap["total"])
    warnings: List[str] = []
    for g in gpus:
        if gpu_check and not g.usable:
            warnings.append(f"{g.short()} NOT used: {g.error or 'CUDA probe failed'}")
    if usable and len(usable) < len(gpus):
        warnings.append(f"working on {len(usable)} of {len(gpus)} GPUs")
    if source == "env" and n > cap["total"]:
        warnings.append(f"SONYA_WORKER_CONCURRENCY={n} is above what the hardware carries ({cap['total']}, limit: {cap['limit']})")
    if usable and n < len(usable):
        warnings.append(f"{n} worker(s) for {len(usable)} usable GPUs: {len(usable) - n} GPU(s) will idle")
    pin = gpu_check and len(gpus) > 1          # one visible card: nothing pinned = exactly the old environment
    slots = assign(n, cap["per_gpu"]) if usable else []
    nv_limit = nvenc_session_limit(usable, e) if usable else 0
    workers = []
    windows = cpu_windows(allowed_cpus() if cpus is None else cpus, n, e)
    for k in range(n):
        w: Dict[str, Any] = {"slot": k + 1, "env": {}}
        if windows[k]:
            w["env"]["SONYA_CPU_AFFINITY"] = cpu_list(windows[k])
        if usable:
            g = usable[slots[k]]
            w.update(gpu_index=g.index, gpu_uuid=g.uuid, gpu_name=g.name, nvenc=g.nvenc_ok)
            w["env"]["SONYA_GPU_INDEX"] = str(g.index)
            if g.nvenc_ok is not None:     # readiness already encoded on THIS card: the worker does not probe again (on a
                w["env"]["SONYA_NVENC_PROBED"] = "ok" if g.nvenc_ok else "fail"   # GeForce the probe could hit the cap)
            if pin:
                w["env"].update(CUDA_VISIBLE_DEVICES=g.uuid or str(g.index), CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if nv_limit:
            w["env"]["SONYA_NVENC_MAX_SESSIONS"] = str(nv_limit)
        workers.append(w)
    per_card = {g.index: sum(1 for w in workers if w.get("gpu_index") == g.index) for g in usable}
    return {"schema": 1, "gpus": [asdict(g) | {"consumer": g.consumer} for g in gpus], "usable": [g.index for g in usable],
            "gpu_check": gpu_check, "cpu_threads": threads, "ram_mb": ram, "budget": asdict(budget), "capacity": cap,
            "concurrency": n, "source": source, "workers": workers, "per_gpu_workers": per_card,
            "cpu_window": len(windows[0]) if windows and windows[0] else 0,
            "nvenc_max_sessions": nv_limit, "warnings": warnings}


def describe(plan: Mapping[str, Any]) -> str:
    cap = plan["capacity"]
    spread = ", ".join(f"GPU{i} x{c}" for i, c in plan.get("per_gpu_workers", {}).items()) or "no GPU pinning"
    nv = plan.get("nvenc_max_sessions") or 0
    return (f"{plan['concurrency']} worker(s) ({plan['source']}) on {len(plan['usable'])} GPU(s): {spread}; hardware carries "
            f"{cap['total']} (vram {cap['by_vram']}, cpu {cap['by_cpu']} of {plan['cpu_threads']} threads, ram {cap['by_ram']} of "
            f"{plan['ram_mb']} MB; limit: {cap['limit']}); nvenc sessions: {nv if nv else 'unlimited'}; "
            f"cpu window per worker: {plan.get('cpu_window') or 'off'}")


def plan_path(env: Mapping[str, str] | None = None) -> Path:
    e = os.environ if env is None else env
    return Path(e.get("SONYA_GPU_PLAN_FILE") or Path(e.get("SONYA_STATE_DIR", "/run/sonya")) / "gpu_plan.json")


def worker_lines(plan: Mapping[str, Any], state_dir: str) -> List[str]:
    out = []
    for w in plan["workers"]:
        env = dict(w["env"])
        if env.get("SONYA_NVENC_MAX_SESSIONS"):
            env["SONYA_NVENC_SLOT_DIR"] = str(Path(state_dir) / "nvenc")   # shared by all workers of this server
        bad = [v for v in env.values() if not re.fullmatch(r"[A-Za-z0-9_./:,-]+", v)]
        if bad:
            raise ValueError("unsafe value in the worker plan")
        out.append(" ".join([str(w["slot"])] + [f"{k}={v}" for k, v in sorted(env.items())]))
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _probe() -> int:  # pragma: no cover - needs a GPU; run by readiness in a child process pinned to one card
    import time
    t0 = time.time()
    try:
        import torch
        if not torch.cuda.is_available():
            print(json.dumps({"ok": False, "error": "torch.cuda.is_available() is False"}))
            return 1
        x = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")   # real context + cuBLAS init
        torch.cuda.synchronize()
        assert float(x[0, 0]) == 1024.0
        props = torch.cuda.get_device_properties(0)
        print(json.dumps({"ok": True, "ms": int((time.time() - t0) * 1000), "count": torch.cuda.device_count(),
                          "name": props.name, "uuid": str(getattr(props, "uuid", "") or ""), "torch": torch.__version__,
                          "cuda": torch.version.cuda}))
        return 0
    except ImportError:
        print(json.dumps({"ok": False, "error": "torch is not installed"}))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"CUDA smoke test failed ({type(exc).__name__})"}))
    return 1


def _fake_gpus(spec: str) -> List[Gpu]:
    """'2xA40:46068,1xGeForce RTX 3060:12288@550' -> Gpu list (simulate / tests)."""
    out: List[Gpu] = []
    for part in [p for p in spec.split(",") if p.strip()]:
        m = re.fullmatch(r"\s*(?:(\d+)x)?([^:@]+):(\d+)(?:@(\d+))?\s*", part)
        if not m:
            raise ValueError(f"bad --gpus item {part!r} (want e.g. 2xA40:46068)")
        for _ in range(int(m.group(1) or 1)):
            i = len(out)
            out.append(Gpu(index=i, uuid=f"GPU-fake{i:04d}-0000-0000-0000-000000000000", name=m.group(2).strip(),
                           vram_total_mb=int(m.group(3)), driver=f"{m.group(4) or 570}.00", cuda_ok=True, nvenc_ok=True))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu_plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("workers")
    sim = sub.add_parser("simulate")
    sim.add_argument("--gpus", default="", help="e.g. 2xA40:46068  or  1xGeForce RTX 3060:12288@550  ('' = no GPU)")
    sim.add_argument("--threads", type=float, required=True)
    sim.add_argument("--ram-gb", type=float, required=True)
    sim.add_argument("--concurrency", default="auto")
    sim.add_argument("--max", default="")
    sim.add_argument("--broken", default="", help="comma-separated card indexes whose probe fails")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        return _probe()
    if a.cmd == "workers":
        p = plan_path()
        try:
            if p.is_file():
                plan = json.loads(p.read_text())
            else:     # readiness skipped the GPU check (local smoke): unpinned workers, count from the env only
                plan = build_plan([], gpu_check=False)
            for line in worker_lines(plan, os.environ.get("SONYA_STATE_DIR", "/run/sonya")):
                print(line)
        except ValueError as exc:
            print(f"[gpu_plan] {exc}", file=sys.stderr)
            return 3
        return 0
    gpus = _fake_gpus(a.gpus)
    for i in [int(x) for x in a.broken.split(",") if x.strip()]:
        gpus[i].cuda_ok, gpus[i].error = False, "simulated broken card"
    env = {"SONYA_WORKER_CONCURRENCY": a.concurrency, **({"SONYA_WORKER_CONCURRENCY_MAX": a.max} if a.max else {})}
    env.update({k: v for k, v in os.environ.items() if k in Budget.ENV.values() or k == "SONYA_NVENC_MAX_SESSIONS"})
    plan = build_plan(gpus, env, threads=a.threads, ram=int(a.ram_gb * 1024), gpu_check=bool(gpus), cpus=range(int(a.threads)))
    print(describe(plan))
    for w in plan["warnings"]:
        print(f"WARNING {w}")
    for w in plan["workers"]:
        print(f"  worker {w['slot']}: " + (f"GPU {w['gpu_index']} {w['gpu_name']}" if "gpu_index" in w else "no GPU pinning")
              + (f"  [{' '.join(f'{k}={v}' for k, v in sorted(w['env'].items()))}]" if w["env"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
