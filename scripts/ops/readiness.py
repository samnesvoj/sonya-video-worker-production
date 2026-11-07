#!/usr/bin/env python3
"""
readiness.py - may this worker accept real jobs? Runs from the verified release before gpu_worker.
Checks: env (names only logged), mode policy, ffmpeg (+NVENC if required), GPU/CUDA/VRAM, required models
(+ sha256 vs release/models.lock.json). Any failure -> exit 1 and NO ready file.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mode_policy import enabled_modes, is_hard_disabled  # noqa: E402

REQUIRED_ENV = ("BACKEND_API_URL", "WORKER_SECRET", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_REGION")


def _p(msg: str) -> None:
    print(f"[readiness] {msg}", flush=True)


def check_env() -> List[str]:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if not (os.environ.get("S3_BUCKET_NAME") or os.environ.get("S3_BUCKET")):
        missing.append("S3_BUCKET_NAME|S3_BUCKET")
    return [f"missing env: {', '.join(missing)}"] if missing else []


def check_policy() -> List[str]:
    bad = [m for m in os.environ.get("ENABLED_MODES", "").split(",") if m.strip() and is_hard_disabled(m)]
    if bad:
        _p(f"ENABLED_MODES contains hard-disabled modes {bad}: ignored by policy")
    modes = enabled_modes()
    _p(f"enabled modes: {','.join(modes)}")
    return [] if modes else ["no runnable modes after policy"]


def check_ffmpeg() -> List[str]:
    ff = os.environ.get("FFMPEG_BIN", "ffmpeg")
    if not shutil.which(ff):
        return ["ffmpeg not found"]
    if os.environ.get("SONYA_REQUIRE_NVENC") == "1":
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if "h264_nvenc" not in enc:
            return ["h264_nvenc not available in ffmpeg"]
    return []


def check_gpu() -> List[str]:
    if os.environ.get("SONYA_SKIP_GPU_CHECK") == "1":
        _p("GPU check skipped (SONYA_SKIP_GPU_CHECK=1) - NOT allowed in production")
        return []
    if not shutil.which("nvidia-smi"):
        return ["nvidia-smi not found: no NVIDIA runtime in the container"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return [f"nvidia-smi failed: {type(exc).__name__}"]
    if not out:
        return ["no GPU reported by nvidia-smi"]
    _p(f"GPU: {out.splitlines()[0]}")
    errs: List[str] = []
    try:
        vram = int(out.splitlines()[0].split(",")[1].strip().split()[0])
        need = int(os.environ.get("SONYA_MIN_VRAM_MB", "20000"))
        if vram < need:
            errs.append(f"GPU VRAM {vram} MB < required {need} MB")
    except (IndexError, ValueError):
        errs.append("cannot parse VRAM from nvidia-smi")
    try:
        import time as _t

        from scripts.ops import timeline

        timeline.mark("cuda_init_start")
        t0 = _t.time()
        import torch  # noqa: WPS433
        if not torch.cuda.is_available():
            errs.append("torch.cuda.is_available() is False")
        else:
            x = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")   # real context + cuBLAS init
            torch.cuda.synchronize()
            assert float(x[0, 0]) == 1024.0
            ms = int((_t.time() - t0) * 1000)
            timeline.mark("cuda_init_done", ms=ms, gpu=torch.cuda.get_device_name(0))
            _p(f"torch {torch.__version__} cuda={torch.version.cuda} context+matmul ok in {ms} ms")
    except ImportError:
        errs.append("torch is not installed")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"CUDA smoke test failed ({type(exc).__name__})")
    return errs


def check_models() -> List[str]:
    """Parallel, SHA-256-verified fetch of every model the enabled modes need (scripts/ops/model_fetch.py)."""
    from scripts.ops import model_fetch, timeline

    lp = ROOT / "release" / "models.lock.json"
    lock = json.loads(lp.read_text()) if lp.is_file() else {}
    errs: List[str] = []
    if os.environ.get("SONYA_REQUIRE_MODEL_LOCK") == "1" and not lock.get("models"):
        return ["release/models.lock.json has no pinned models (SONYA_REQUIRE_MODEL_LOCK=1)"]
    models_dir = Path(os.environ.get("MODELS_LOCAL_DIR", "/opt/sonya/models"))
    items = model_fetch.plan(ROOT, enabled_modes(), models_dir, lock)
    timeline.mark("models_start", count=len(items))
    try:
        results = model_fetch.fetch_all(items)
    except Exception as exc:  # noqa: BLE001
        timeline.mark("models_failed")
        return [f"model fetch failed ({type(exc).__name__}: {str(exc)[:200]})"]
    total = sum(r.bytes for r in results)
    timeline.mark("models_done", bytes_downloaded=total, models=len(results), statuses={s: sum(1 for r in results if r.status == s) for s in {r.status for r in results}})
    for r in results:
        _p(f"model {r.key}: {r.status} {r.bytes} B {r.seconds:.1f}s")
    if os.environ.get("SONYA_REQUIRE_MODEL_LOCK") == "1":
        errs += [f"model {r.key} is not pinned/verified (status {r.status})" for r in results if r.status == "unverified"]
    return errs


def main() -> int:
    ready = Path(os.environ.get("SONYA_READY_FILE", "/run/sonya/ready"))
    if ready.exists():
        ready.unlink()
    errors: List[str] = []
    for name, fn in (("env", check_env), ("policy", check_policy), ("ffmpeg", check_ffmpeg), ("gpu", check_gpu), ("models", check_models)):
        errs = fn()
        _p(f"{name}: {'OK' if not errs else 'FAIL'}")
        errors += [f"{name}: {e}" for e in errs]
    if errors:
        for e in errors:
            _p(f"ERROR {e}")
        return 1
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ready\n")
    try:
        from scripts.ops import timeline
        timeline.mark("readiness_ok")
    except Exception:  # noqa: BLE001
        pass
    _p("worker READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
