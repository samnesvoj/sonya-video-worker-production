"""
res_sampler.py - resource samples of ONE worker server for load tests (off unless SONYA_RES_SAMPLER=1).

Every SONYA_RES_SAMPLER_SEC (5) seconds: GPU memory used / total, GPU + NVENC utilisation, RAM used, CPU %, load average,
number of busy worker processes. One JSON object per minute is uploaded to S3 as a NEW key
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
    try:
        import pynvml  # nvidia-ml-py (in the lock)
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:  # noqa: BLE001
        return None, None


def sample(nv, h) -> Dict[str, Any]:
    import psutil
    s: Dict[str, Any] = {"t": round(time.time(), 1), "cpu_pct": psutil.cpu_percent(interval=None),
                         "ram_used_mb": int((psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20),
                         "load1": round(os.getloadavg()[0], 2)}
    if nv and h:
        try:
            m = nv.nvmlDeviceGetMemoryInfo(h)
            s.update(vram_used_mb=int(m.used / 2**20), vram_total_mb=int(m.total / 2**20),
                     gpu_util=nv.nvmlDeviceGetUtilizationRates(h).gpu, enc_util=nv.nvmlDeviceGetEncoderUtilization(h)[0])
        except Exception:  # noqa: BLE001
            pass
    try:
        s["ffmpeg"] = sum(1 for p in psutil.process_iter(["name"]) if (p.info.get("name") or "").startswith("ffmpeg"))
    except Exception:  # noqa: BLE001
        pass
    return s


def summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(samples)}
    for k in ("vram_used_mb", "ram_used_mb", "cpu_pct", "gpu_util", "enc_util", "load1", "ffmpeg"):
        vals = [x[k] for x in samples if k in x]
        if vals:
            out[k + "_max"] = max(vals)
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
