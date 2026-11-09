"""
res_sampler.py - resource samples of ONE worker server for load tests (off unless SONYA_RES_SAMPLER=1).

Every SONYA_RES_SAMPLER_SEC (5) seconds: GPU memory used / total, GPU + NVENC utilisation (server totals AND, on a multi-GPU
server, every card: "gpus" -> summary "per_gpu" peaks + average utilisation), RAM used, CPU %, load average, number of ffmpegs. One JSON object per minute is uploaded to S3 as a NEW key
    fast-start-test/res/<WORKER_ID>/<boot-epoch>/<minute-index>.json
(never overwrites anything; no env values, no file names, no user data). Best-effort: never raises, never blocks the worker.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List


def _nvml():
    """(pynvml, [handle of EVERY card]) or (None, None). NVML indexes cards in PCI order = nvidia-smi index."""
    try:
        import pynvml  # nvidia-ml-py (in the lock)
        pynvml.nvmlInit()
        return pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:  # noqa: BLE001
        return None, None


def gpu_samples(nv, handles) -> List[Dict[str, Any]]:
    """Per card: VRAM used/total, SM + NVENC utilisation, number of our processes on it (= workers pinned there)."""
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(handles if isinstance(handles, (list, tuple)) else [handles] if handles else []):
        g: Dict[str, Any] = {"i": i}
        try:
            m = nv.nvmlDeviceGetMemoryInfo(h)
            g.update(vram_used_mb=int(m.used / 2**20), vram_total_mb=int(m.total / 2**20),
                     gpu_util=nv.nvmlDeviceGetUtilizationRates(h).gpu, enc_util=nv.nvmlDeviceGetEncoderUtilization(h)[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            g["procs"] = len(nv.nvmlDeviceGetComputeRunningProcesses(h))
        except Exception:  # noqa: BLE001
            pass
        out.append(g)
    return out


_CG: Dict[str, float] = {}


def cgroup() -> Dict[str, Any]:
    """The CONTAINER's own memory and CPU (psutil inside Docker reports the whole host): cgroup v2, else v1."""
    out: Dict[str, Any] = {}
    root = "/sys/fs/cgroup"
    try:
        for f in (f"{root}/memory.current", f"{root}/memory/memory.usage_in_bytes"):
            if os.path.exists(f):
                out["cg_mem_mb"] = int(int(open(f).read().strip()) / 2**20)
                break
        usec = None
        if os.path.exists(f"{root}/cpu.stat"):
            usec = next(int(l.split()[1]) for l in open(f"{root}/cpu.stat") if l.startswith("usage_usec"))
        elif os.path.exists(f"{root}/cpuacct/cpuacct.usage"):
            usec = int(open(f"{root}/cpuacct/cpuacct.usage").read()) / 1000
        if usec is not None:
            now = time.time()
            if "u" in _CG:
                out["cg_cpu_cores"] = round((usec - _CG["u"]) / 1e6 / max(now - _CG["t"], 1e-3), 2)   # cores busy on average
            _CG.update(u=usec, t=now)
    except Exception:  # noqa: BLE001
        pass
    return out


def sample(nv, h) -> Dict[str, Any]:
    import psutil
    s: Dict[str, Any] = {"t": round(time.time(), 1), "cpu_pct": psutil.cpu_percent(interval=None),
                         "ram_used_mb": int((psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20),
                         "load1": round(os.getloadavg()[0], 2), "cpus": os.cpu_count(), **cgroup()}
    try:                                   # how big the Vast disk must be: image + release + models + temp of parallel jobs
        s["disk_used_gb"] = round(psutil.disk_usage("/").used / 1e9, 2)
        s["tmp_gb"] = round(sum(f.stat().st_size for f in __import__("pathlib").Path(os.environ.get("TMPDIR", "/work/tmp")).rglob("*")
                                if f.is_file()) / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    if nv and h:
        gs = [g for g in gpu_samples(nv, h) if "vram_used_mb" in g]
        if gs:
            # server totals (one card: exactly the old fields) + every card on its own (multi-GPU: is any card idle / full?)
            s.update(vram_used_mb=sum(g["vram_used_mb"] for g in gs), vram_total_mb=sum(g["vram_total_mb"] for g in gs),
                     gpu_util=max(g["gpu_util"] for g in gs), enc_util=max(g["enc_util"] for g in gs))
            if len(gs) > 1:
                s["gpus"] = gs
    try:
        ff = [p.info.get("cmdline") or [] for p in psutil.process_iter(["name", "cmdline"]) if (p.info.get("name") or "").startswith("ffmpeg")]
        s["ffmpeg"] = len(ff)
        s["ff_nvenc"] = sum(1 for c in ff if "h264_nvenc" in c)      # NVENC sessions in use (GeForce: max 8)
        s["ff_x264"] = sum(1 for c in ff if "libx264" in c)
    except Exception:  # noqa: BLE001
        pass
    return s


def summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(samples)}
    for k in ("vram_used_mb", "ram_used_mb", "cpu_pct", "gpu_util", "enc_util", "load1", "ffmpeg", "ff_nvenc", "ff_x264", "disk_used_gb", "tmp_gb", "cg_mem_mb",
              "cg_cpu_cores"):
        vals = [x[k] for x in samples if k in x]
        if vals:
            out[k + "_max"] = max(vals)
    per: Dict[str, Dict[str, Any]] = {}
    for x in samples:
        for g in x.get("gpus") or []:
            per.setdefault(str(g.get("i")), {}).setdefault("_rows", []).append(g)
    for i, d in per.items():
        rows = d.pop("_rows")
        for k in ("vram_used_mb", "gpu_util", "enc_util", "procs"):
            vals = [r[k] for r in rows if k in r]
            if vals:
                d[k + "_max"] = max(vals)
        util = [r["gpu_util"] for r in rows if "gpu_util" in r]
        if util:
            d["gpu_util_avg"] = round(sum(util) / len(util), 1)
        d["vram_total_mb"] = max((r.get("vram_total_mb", 0) for r in rows), default=0)
    if per:
        out["per_gpu"] = per
    return out


def run() -> None:  # pragma: no cover - needs a GPU server; exercised in the Vast load test
    if os.environ.get("SONYA_RES_SAMPLER") != "1":
        return
    every = float(os.environ.get("SONYA_RES_SAMPLER_SEC", "5"))
    wid, boot = os.environ.get("WORKER_ID", "worker"), int(time.time())
    nv, h = _nvml()
    try:
        import psutil
        psutil.cpu_percent(interval=None)
        from scripts.prod_s3_storage import upload_bytes
    except Exception:  # noqa: BLE001
        return
    buf: List[Dict[str, Any]] = []
    minute, t_flush = 0, time.time() + 60
    while True:
        try:
            buf.append(sample(nv, h))
            if time.time() >= t_flush:
                body = json.dumps({"worker_id": wid, "boot": boot, "minute": minute, "summary": summary(buf), "samples": buf}).encode()
                upload_bytes(body, f"fast-start-test/res/{wid}/{boot}/{minute:04d}.json", content_type="application/json")
                buf, minute, t_flush = [], minute + 1, time.time() + 60
        except Exception as exc:  # noqa: BLE001
            print(f"[res_sampler] {type(exc).__name__}", file=sys.stderr)
            buf, t_flush = [], time.time() + 60
        time.sleep(every)


if __name__ == "__main__":
    run()
