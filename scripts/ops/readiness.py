#!/usr/bin/env python3
"""
readiness.py - may this worker accept real jobs? Runs from the verified release before gpu_worker.
Checks: env (names only logged), mode policy, ffmpeg (+NVENC if required), GPU/CUDA/VRAM, required models
(+ sha256 vs release/models.lock.json). Any failure -> exit 1 and NO ready file.
"""
from __future__ import annotations

import hashlib
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
        import torch  # noqa: WPS433
        if not torch.cuda.is_available():
            errs.append("torch.cuda.is_available() is False")
        else:
            _p(f"torch {torch.__version__} cuda={torch.version.cuda}")
    except ImportError:
        errs.append("torch is not installed")
    return errs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_models() -> List[str]:
    import yaml  # noqa: WPS433
    from scripts import model_downloader as md  # noqa: WPS433

    lp = ROOT / "release" / "models.lock.json"
    pins = (json.loads(lp.read_text()) if lp.is_file() else {}).get("models", {})
    errs: List[str] = []
    if os.environ.get("SONYA_REQUIRE_MODEL_LOCK") == "1" and not pins:
        errs.append("release/models.lock.json has no pinned models (SONYA_REQUIRE_MODEL_LOCK=1)")
    for mode in enabled_modes():
        cfg = ROOT / "modes" / mode / "mode.yaml"
        if not cfg.is_file():
            continue
        try:
            if not md.download_models_for_mode(cfg):
                errs.append(f"{mode}: required models unavailable")
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{mode}: model download failed ({type(exc).__name__})")
            continue
        for key, s in ((yaml.safe_load(cfg.read_text()) or {}).get("models") or {}).items():
            local = md._resolve_local_path(s.get("local_path"), key)  # noqa: SLF001
            if not s.get("optional", True) and not local.exists():
                errs.append(f"{mode}: required model missing: {key}")
            pin = pins.get(key)
            if pin and local.exists() and _sha256(local) != pin["sha256"]:
                errs.append(f"{mode}: model {key} sha256 mismatch")
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
    _p("worker READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
