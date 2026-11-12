#!/usr/bin/env python3
"""
readiness.py - may this worker accept real jobs? Runs from the verified release before gpu_worker.
Checks: env (names only logged), mode policy, ffmpeg, EVERY GPU (VRAM, CUDA context, real NVENC encode; a broken card is
left out, the others work), required models
(+ sha256 vs release/models.lock.json). Any failure -> exit 1 and NO ready file.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mode_policy import enabled_modes, is_hard_disabled  # noqa: E402
from scripts.ops import gpu_plan  # noqa: E402

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
    enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if os.environ.get("SONYA_REQUIRE_NVENC") == "1" and "h264_nvenc" not in enc:
        return ["h264_nvenc not available in ffmpeg"]
    _STATE["nvenc_compiled"] = "h264_nvenc" in enc     # the REAL encode is done per card in check_gpu
    return []


_STATE: Dict[str, Any] = {}


def probe_cuda(gpu: gpu_plan.Gpu, pin: bool) -> Dict[str, Any]:
    """CUDA context + matmul in a CHILD process pinned to this card (a broken card can hang or kill CUDA init; the child
    has a timeout and cannot take readiness down). -> {"ok", "ms", "uuid", "error"}."""
    env = dict(os.environ, **_pin_env(gpu, pin))
    try:
        r = subprocess.run([sys.executable, "-m", "scripts.ops.gpu_plan", "probe"], capture_output=True, text=True,
                           timeout=int(os.environ.get("SONYA_GPU_PROBE_TIMEOUT", "180")), env=env, cwd=str(ROOT))
        out = json.loads((r.stdout or "").strip().splitlines()[-1]) if (r.stdout or "").strip() else {}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "CUDA probe timed out"}
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"CUDA probe failed ({type(exc).__name__})"}
    if not out:
        return {"ok": False, "error": f"CUDA probe crashed (rc {r.returncode})"}
    if out.get("ok") and pin and out.get("count") not in (None, 1):
        return {"ok": False, "error": f"pinning did not isolate the card (torch sees {out.get('count')} devices)"}
    want = gpu.uuid.lower().removeprefix("gpu-")
    if out.get("ok") and pin and out.get("uuid") and want and str(out["uuid"]).lower().removeprefix("gpu-") != want:
        return {"ok": False, "error": "pinned process landed on another card"}
    return out


def probe_nvenc(gpu: gpu_plan.Gpu, pin: bool) -> Tuple[bool, str]:
    """A REAL 0.5 s h264_nvenc encode on this card, not just the encoder list: run 4 had h264_nvenc compiled in but no
    libnvidia-encode in the container (NVIDIA_DRIVER_CAPABILITIES without "video") -> every clip silently went to libx264."""
    ff = os.environ.get("FFMPEG_BIN", "ffmpeg")
    try:
        r = subprocess.run([ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=320x240:rate=30:duration=0.5", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=60, env=dict(os.environ, **_pin_env(gpu, pin)))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, type(exc).__name__
    return r.returncode == 0, " ".join((r.stderr or "").strip().splitlines()[-1:])


def _pin_env(gpu: gpu_plan.Gpu, pin: bool) -> Dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": gpu.uuid or str(gpu.index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"} if pin else {}


def check_gpu() -> List[str]:
    """EVERY card: VRAM, CUDA context (own pinned process), real NVENC encode. A broken card is left out (logged, timeline
    'gpu_degraded'), the server works on the others; only no usable card at all fails readiness. Writes the worker plan
    (scripts/ops/gpu_plan.py: how many workers, which card each) to $SONYA_STATE_DIR/gpu_plan.json for run_worker.sh."""
    from scripts.ops import timeline

    plan_file = gpu_plan.plan_path()
    plan_file.unlink(missing_ok=True)
    if os.environ.get("SONYA_SKIP_GPU_CHECK") == "1":
        _p("GPU check skipped (SONYA_SKIP_GPU_CHECK=1) - NOT allowed in production")
        return []
    if not shutil.which("nvidia-smi"):
        return ["nvidia-smi not found: no NVIDIA runtime in the container"]
    gpus = gpu_plan.detect()
    if not gpus:
        return ["no GPU reported by nvidia-smi"]
    caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "(unset)")
    need = int(os.environ.get("SONYA_MIN_VRAM_MB", "20000"))
    pin = len(gpus) > 1
    _p(f"{len(gpus)} GPU(s), driver {gpus[0].driver}, driver capabilities {caps}")
    for g in gpus:
        if g.vram_total_mb < need:
            g.error = f"VRAM {g.vram_total_mb} MB < required {need} MB"
    timeline.mark("cuda_init_start", gpus=len(gpus))
    from concurrent.futures import ThreadPoolExecutor

    def one(g: gpu_plan.Gpu) -> None:
        if g.error:
            g.cuda_ok = False
            return
        c = probe_cuda(g, pin)
        g.cuda_ok, g.cuda_ms = bool(c.get("ok")), c.get("ms")
        if not g.cuda_ok:
            g.error = str(c.get("error") or "CUDA probe failed")[:160]
            return
        if _STATE.get("nvenc_compiled"):
            g.nvenc_ok, why = probe_nvenc(g, pin)
            if not g.nvenc_ok:
                g.nvenc_error = why[:160]
                if os.environ.get("SONYA_REQUIRE_NVENC") == "1":
                    g.error = "h264_nvenc does not encode on this card (SONYA_REQUIRE_NVENC=1)"

    with ThreadPoolExecutor(max_workers=min(8, len(gpus))) as ex:     # all cards at once: start-up time = the slowest card
        list(ex.map(one, gpus))
    for g in gpus:
        nv = {True: "OK", False: f"NOT usable ({g.nvenc_error}) - clips of this card render on the CPU",
              None: "not in ffmpeg"}[g.nvenc_ok]
        state = f"cuda OK {g.cuda_ms} ms, nvenc: {nv}" if g.cuda_ok else f"NOT USED: {g.error}"
        _p(f"{g.short()}: {state}")
    usable = [g for g in gpus if g.usable]
    if usable:
        timeline.mark("cuda_init_done", ms=max(g.cuda_ms or 0 for g in usable), gpu=usable[0].name, gpus=len(usable))
    if len(usable) < len(gpus):
        timeline.mark("gpu_degraded", usable=len(usable), total=len(gpus),
                      failed=[{"index": g.index, "name": g.name, "reason": g.error} for g in gpus if not g.usable])
    if not usable:
        if os.environ.get("SONYA_REQUIRE_NVENC") == "1" and all(g.cuda_ok for g in gpus):
            return ["h264_nvenc does not encode"]
        return [f"no usable GPU ({'; '.join(f'GPU {g.index}: {g.error}' for g in gpus)})"]
    try:
        plan = gpu_plan.build_plan(gpus)
    except ValueError as exc:
        return [str(exc)]
    for w in plan["warnings"]:
        _p(f"WARNING {w}")
    _p(f"plan: {gpu_plan.describe(plan)}")
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(plan, indent=1))
    timeline.mark("gpu_plan", workers=plan["concurrency"], gpus=len(usable), per_gpu={str(k): v for k, v in plan["per_gpu_workers"].items()},
                  limit=plan["capacity"]["limit"], nvenc_max_sessions=plan["nvenc_max_sessions"],
                  nvenc_ok=sum(1 for g in usable if g.nvenc_ok))
    return []


def check_asr() -> List[str]:
    """Primary ASR must be usable OFFLINE before the worker accepts jobs (owner decision: GigaAM v3, faster-whisper only as fallback).
    Enforced when SONYA_REQUIRE_ASR=1 (run_worker.sh sets it by default; local smoke tools opt out)."""
    if os.environ.get("SONYA_REQUIRE_ASR") != "1":
        return []
    from scripts.shared import asr_contract as ac
    errs: List[str] = []
    try:
        backend = ac.backend_name()
    except ValueError as exc:
        return [str(exc)]
    _p(f"asr backend: {backend}")
    if os.environ.get("SONYA_ASR_OFFLINE") != "1":
        errs.append("SONYA_ASR_OFFLINE must be 1: no model download at start-up or run time")
    from scripts.shared.model_paths import models_root
    errs += [f"ASR file missing: {f.relative_to(models_root())}" for f in ac.expected_model_files(models_root(), backend) if not f.is_file()]
    try:
        import importlib
        mod = importlib.import_module("scripts.shared.asr")
        if not callable(getattr(mod, "transcribe", None)):
            errs.append("scripts.shared.asr has no transcribe() (docs/ASR_INTERFACE.md)")
    except ImportError as exc:
        errs.append(f"ASR backend module scripts.shared.asr is not importable ({type(exc).__name__})")
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
    if os.environ.get("SONYA_REQUIRE_ALL_MODELS") == "1":      # GPU test / reference runs: an optional model that is absent would silently change output
        errs += [f"model {r.key} is missing (SONYA_REQUIRE_ALL_MODELS=1)" for r in results if r.status == "skipped_optional"]
    return errs


def main() -> int:
    ready = Path(os.environ.get("SONYA_READY_FILE", "/run/sonya/ready"))
    if ready.exists():
        ready.unlink()
    errors: List[str] = []
    for name, fn in (("env", check_env), ("policy", check_policy), ("ffmpeg", check_ffmpeg), ("gpu", check_gpu), ("models", check_models), ("asr", check_asr)):
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
