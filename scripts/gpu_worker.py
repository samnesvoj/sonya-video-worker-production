"""
gpu_worker.py
=============
Full-featured GPU worker for SONYA generation pipeline.

Usage:
    python scripts/gpu_worker.py --once --job-id JOB_ID
    python scripts/gpu_worker.py --poll
    python scripts/gpu_worker.py --poll --worker-id my-gpu-node-1

Backend modes (WORKER_BACKEND_MODE env var):
    db   — default; direct PostgreSQL via prod_job_store (requires DATABASE_URL)
    api  — HTTP-only; all job operations go through BACKEND_API_URL worker endpoints
            (requires BACKEND_API_URL + WORKER_SECRET; DATABASE_URL NOT required)
            Used when the GPU instance cannot reach the private PostgreSQL server
            (e.g. vast.ai external GPU provider).

Flow (both modes):
    1. claim_job (specific or next pending)
    2. update status → claimed → downloading
    3. download input from S3
    4. ensure_models_for_mode (model_downloader)
    5. run mode via mode_registry
    6. upload all output files to S3
    7. register each file via backend
    8. complete_job (or fail_job on error)
    9. cleanup temp dir
"""
from __future__ import annotations

import os as _os

if _os.environ.get("SONYA_CPU_AFFINITY") and hasattr(_os, "sched_setaffinity"):
    # big multi-worker servers (scripts/ops/gpu_plan.cpu_windows): this worker's CPU window, set BEFORE any import starts
    # a thread pool; ffmpeg / OpenCV / torch inherit it and size their pools by it. A bad value leaves the default.
    try:
        _cpus = set()
        for _part in _os.environ["SONYA_CPU_AFFINITY"].split(","):
            _a, _, _b = _part.partition("-")
            _cpus.update(range(int(_a), int(_b or _a) + 1))
        _os.sched_setaffinity(0, _cpus & _os.sched_getaffinity(0) or _os.sched_getaffinity(0))
    except (OSError, ValueError):
        pass

import argparse
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Backend mode ───────────────────────────────────────────────────────────────

_WORKER_MODE = os.environ.get("WORKER_BACKEND_MODE", "db").lower()

# Job status string constants — identical to prod_job_store values.
# Defined here so API mode works without importing prod_job_store.
JOB_STATUS_QUEUED            = "queued"
JOB_STATUS_CLAIMED           = "claimed"
JOB_STATUS_COMPLETED         = "completed"
JOB_STATUS_FAILED            = "failed"
JOB_STATUS_DOWNLOADING       = "downloading"
JOB_STATUS_MODEL_DOWNLOADING = "model_downloading"
JOB_STATUS_MODE_RUNNING      = "mode_running"
JOB_STATUS_UPLOADING_RESULT  = "uploading_result"

if _WORKER_MODE == "db":
    from scripts.prod_job_store import (
        add_job_file                    as _db_add_file,
        claim_next_pending_job          as _db_claim_next,
        claim_specific_job              as _db_claim_specific,
        complete_job                    as _db_complete,
        fail_job                        as _db_fail,
        get_job                         as _db_get_job,
        requeue_stale_jobs              as _db_requeue_stale,
        submit_streamer_analysis_result as _db_submit_streamer_segments,
        touch_job_heartbeat             as _db_heartbeat,
        update_job_status               as _db_update_status,
    )

from scripts.mode_policy import DISABLED_CODE, ModeDisabledError, assert_mode_allowed, claimable_modes
from scripts.mode_registry import get_runner
from scripts.model_downloader import download_models_for_mode
from scripts.prod_s3_storage import (
    build_output_key,
    download_file,
    upload_file,
)
from scripts.security import new_trace_id

POLL_INTERVAL  = int(os.environ.get("WORKER_POLL_INTERVAL", "10"))
STALE_INTERVAL = int(os.environ.get("WORKER_STALE_REQUEUE_INTERVAL", "300"))

# Heartbeat cadence during a long mode run (see _do_heartbeat / _HeartbeatPulse
# below) — comfortably under WORKER_STALE_REQUEUE_INTERVAL's default
# stale_minutes=30 check, so a heartbeat is always fresh when checked.
HEARTBEAT_INTERVAL_SEC = int(os.environ.get("WORKER_HEARTBEAT_INTERVAL_SEC", "120"))

_PREVIEW_MODES = ("virality", "hooks", "educational", "streamer", "beep")


def _make_preview_assets(clip: Path) -> None:
    """Best effort: <stem>_preview.mp4 (720p, <=2.5 Mb/s, +faststart) and <stem>_poster.jpg.
    The full-quality clip stays the download file. Never fails the job."""
    import subprocess
    try:
        prev = clip.with_name(clip.stem + "_preview.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", "scale=-2:720",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-maxrate", "2500k",
                        "-bufsize", "5000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                        "-movflags", "+faststart", str(prev)], check=True, timeout=300)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(clip), "-frames:v", "1",
                        "-vf", "scale=-2:720", "-q:v", "4", str(clip.with_name(clip.stem + "_poster.jpg"))],
                       check=True, timeout=60)
    except Exception as exc:
        logger.warning("[worker] preview_assets_failed %s: %s", clip.name, type(exc).__name__)


_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".srt": "text/plain",
    ".vtt": "text/vtt",
    ".ass": "text/x-ssa",
}
_FILE_TYPES = {
    ".mp4": "output",
    ".mov": "output",
    ".png": "output",       # sonya_gen (images)
    ".jpg": "output",
    ".json": "enrichment_json",
    ".srt": "subtitle",
    ".vtt": "subtitle",
    ".ass": "subtitle",     # burned-in subtitle source of a clip: never counted as a clip
}


# -- Graceful shutdown (SIGTERM from docker stop / Vast destroy) ---------------
_STOP = threading.Event()
_ACTIVE_JOB: Dict[str, Optional[str]] = {"id": None}


def _on_sigterm(signum, _frame) -> None:
    """Stop claiming; requeue the in-flight job (retry=True); unwind so finally-blocks wipe temp data."""
    _STOP.set()
    job_id = _ACTIVE_JOB.get("id")
    logger.warning("[worker] signal=%s received active_job=%s", signum, job_id)
    if job_id:
        _ACTIVE_JOB["id"] = None
        os.environ["WORKER_API_DURABLE_SEC"] = "5"   # docker stop grants ~10 s: never block the shutdown on retries
        _fail(job_id, "WORKER_SHUTDOWN", "worker terminated while job was running", retry=True)
    raise SystemExit(143)


_FIRST_JOB = {"claimed": False, "processing": False}


def _timeline(name: str, **kw) -> None:
    try:
        from scripts.ops import timeline
        timeline.mark(name, **kw)
        if name == "first_job_processing":
            timeline.upload_async()
    except Exception:  # noqa: BLE001 - metrics never affect jobs
        pass


def _prewarm(modes) -> None:
    """Pay the one-off import / CUDA-context cost BEFORE the first user job (production path only). Never fatal."""
    t0 = time.monotonic()
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        for m in modes:
            try:
                get_runner(m)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[worker] prewarm: mode %s not importable (%s)", m, type(exc).__name__)
        try:
            import torch
            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[worker] prewarm skipped: %s", type(exc).__name__)
    _timeline("prewarm_done", seconds=round(time.monotonic() - t0, 2))


def _worker_id() -> str:
    return os.environ.get("WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"


# ── API-mode backend client ────────────────────────────────────────────────────

class _BackendAPIClient:
    """
    HTTP client for SONYA worker endpoints.
    Used when WORKER_BACKEND_MODE=api (no direct DB access).
    All requests are authenticated with Bearer WORKER_SECRET.
    """

    def __init__(self) -> None:
        self._base = os.environ.get("BACKEND_API_URL", "").rstrip("/")
        self._secret = os.environ.get("WORKER_SECRET", "")
        if not self._base:
            raise RuntimeError(
                "WORKER_BACKEND_MODE=api requires BACKEND_API_URL to be set"
            )
        if not self._secret:
            raise RuntimeError(
                "WORKER_BACKEND_MODE=api requires WORKER_SECRET to be set"
            )

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._secret}",
        }

    def _post(self, path: str, body: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
        import requests  # type: ignore
        url = f"{self._base}{path}"
        resp = requests.post(url, json=body, headers=self._headers, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(
                f"API {path} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()

    def _post_durable(self, path: str, body: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
        """_post for the calls whose loss loses the user's result (files / complete / fail): a backend or
        network blip (connection error, timeout, HTTP 5xx/408/429) is retried with backoff for up to
        WORKER_API_DURABLE_SEC (default 900 s) instead of dropping the finished job. The backend handlers
        are idempotent (files: ON CONFLICT (s3_key) DO NOTHING; complete: plain UPDATE). Other 4xx raise
        at once: retrying a rejected request cannot help."""
        budget = float(os.environ.get("WORKER_API_DURABLE_SEC", "900"))
        deadline = time.monotonic() + budget
        delay = 2.0
        while True:
            try:
                return self._post(path, body, timeout=timeout)
            except RuntimeError as exc:          # HTTP status from the backend
                m = re.search(r"HTTP (\d{3})", str(exc))
                code = int(m.group(1)) if m else 0
                if not (code >= 500 or code in (408, 429)):
                    raise
                err: Exception = exc
            except Exception as exc:             # connection reset / TLS EOF / timeout / DNS
                err = exc
            if time.monotonic() + delay > deadline:
                raise err
            logger.warning("[api] %s unreachable (%s) - retry in %.0f s", path.rsplit("/", 1)[-1],
                           str(err)[:160], delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    def claim_job(
        self,
        job_id: Optional[str],
        worker_id: str,
        modes: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        body: Dict[str, Any] = {"worker_id": worker_id}
        if job_id:
            body["job_id"] = job_id
        if modes:
            body["modes"] = modes
        data = self._post("/api/worker/claim", body)
        return data.get("job")

    def update_status(self, job_id: str, status: str) -> None:
        try:
            self._post(f"/api/worker/jobs/{job_id}/status", {"status": status})
        except Exception as exc:
            logger.warning("[api] update_status failed job_id=%s status=%s: %s", job_id, status, exc)

    def heartbeat(self, job_id: str) -> None:
        try:
            self._post(f"/api/worker/jobs/{job_id}/heartbeat", {})
        except Exception as exc:
            # Never fatal — a missed heartbeat just means this job relies
            # on the pre-heartbeat claimed_at fallback for one more cycle.
            logger.warning("[api] heartbeat failed job_id=%s: %s", job_id, exc)

    def complete_job(
        self,
        job_id: str,
        s3_output_key: str,
        clip_count: int,
        processing_ms: int,
        enrichment_keys: Optional[List[str]] = None,
    ) -> None:
        self._post_durable(
            f"/api/worker/jobs/{job_id}/complete",
            {
                "s3_output_key": s3_output_key,
                "clip_count": clip_count,
                "processing_ms": processing_ms,
                "enrichment_keys": enrichment_keys or [],
            },
        )

    def fail_job(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
        retry: bool,
    ) -> None:
        try:
            self._post_durable(
                f"/api/worker/jobs/{job_id}/fail",
                {
                    "error_code": error_code,
                    "error_message": error_message,
                    "retry": retry,
                },
            )
        except Exception as exc:
            logger.error("[api] fail_job endpoint failed job_id=%s: %s", job_id, exc)

    def add_file(
        self,
        job_id: str,
        user_id: str,
        file_type: str,
        s3_key: str,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> None:
        try:
            self._post_durable(
                f"/api/worker/jobs/{job_id}/files",
                {
                    "file_type": file_type,
                    "s3_key": s3_key,
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                },
            )
        except Exception as exc:
            logger.warning("[api] add_file failed job_id=%s file=%s: %s", job_id, filename, exc)

    def submit_streamer_segments(self, batch_id: str, analysis_result: Dict[str, Any]) -> None:
        """
        Worker-side counterpart of prod_job_store.submit_streamer_analysis_
        result() for API-mode workers (no direct DB access) — see
        scripts/streamer_routes.py's POST /api/worker/streamer/batches/
        {batch_id}/analysis-result, which calls that same store function
        server-side on receipt of this request. Not wrapped in try/except
        here (unlike heartbeat/add_file/fail_job above) — a failure here
        must propagate so the caller (process_job's analyze-phase branch)
        fails and retries the job, rather than silently completing an
        analysis job whose segments were never persisted.
        """
        self._post(
            f"/api/worker/streamer/batches/{batch_id}/analysis-result",
            {
                "segments": analysis_result.get("segments", []),
                "crop_hints": analysis_result.get("crop_hints") or {},
                "warnings": analysis_result.get("warnings") or [],
                "webcam_boxes_found": analysis_result.get("webcam_boxes_found"),
                "active_speaker_segs": analysis_result.get("active_speaker_segs"),
            },
            timeout=30,
        )


# Instantiate the API client lazily (only when mode=api)
_api_client: Optional[_BackendAPIClient] = None


def _get_api_client() -> _BackendAPIClient:
    global _api_client
    if _api_client is None:
        _api_client = _BackendAPIClient()
    return _api_client


# ── Backend-dispatching helpers ────────────────────────────────────────────────

def _update_status(job_id: str, status: str) -> None:
    if _WORKER_MODE == "api":
        _get_api_client().update_status(job_id, status)
    else:
        _db_update_status(job_id, status)


def _do_heartbeat(job_id: str) -> None:
    try:
        if _WORKER_MODE == "api":
            _get_api_client().heartbeat(job_id)
        else:
            _db_heartbeat(job_id)
    except Exception as exc:
        # Never fatal — see _HeartbeatPulse, which swallows this the same
        # way so a transient DB hiccup can't ever take down the actual
        # mode run it's just trying to keep alive.
        logger.warning("[worker] heartbeat_failed job_id=%s: %s", job_id, exc)


def _do_complete_job(
    job_id: str,
    s3_output_key: str,
    clip_count: int,
    processing_ms: int,
    enrichment_keys: Optional[List[str]],
) -> None:
    if _WORKER_MODE == "api":
        _get_api_client().complete_job(
            job_id, s3_output_key, clip_count, processing_ms, enrichment_keys
        )
    else:
        _db_complete(
            job_id=job_id,
            s3_output_key=s3_output_key,
            clip_count=clip_count,
            processing_ms=processing_ms,
            enrichment_keys=enrichment_keys,
        )


def _do_fail_job(job_id: str, error_code: str, error_message: str, retry: bool) -> None:
    if _WORKER_MODE == "api":
        _get_api_client().fail_job(job_id, error_code, error_message, retry)
    else:
        _db_fail(job_id=job_id, error_code=error_code, error_message=error_message, retry=retry)


def _do_add_file(
    job_id: str,
    user_id: str,
    file_type: str,
    s3_key: str,
    filename: str,
    content_type: str,
    size_bytes: int,
) -> None:
    if _WORKER_MODE == "api":
        _get_api_client().add_file(
            job_id, user_id, file_type, s3_key, filename, content_type, size_bytes
        )
    else:
        _db_add_file(
            job_id=job_id, user_id=user_id,
            file_type=file_type, s3_key=s3_key,
            filename=filename, content_type=content_type,
            size_bytes=size_bytes,
        )


def _do_submit_streamer_segments(batch_id: str, analysis_result: Dict[str, Any]) -> None:
    """
    Deliberately NOT swallowed in try/except (unlike heartbeat/add_file/
    fail_job) — see _BackendAPIClient.submit_streamer_segments's own
    docstring for why a failure here must propagate to the caller.
    """
    if _WORKER_MODE == "api":
        _get_api_client().submit_streamer_segments(batch_id, analysis_result)
    else:
        _db_submit_streamer_segments(batch_id, analysis_result)


# ── Model download ─────────────────────────────────────────────────────────────

def ensure_models_for_mode(mode: str) -> bool:
    """Download required models from S3 before running mode."""
    mode_yaml = _ROOT / "modes" / mode / "mode.yaml"
    if not mode_yaml.exists():
        logger.warning("[worker] no mode.yaml for mode=%s — skipping model download", mode)
        return True
    return download_models_for_mode(mode_yaml)


# ── Heartbeat ──────────────────────────────────────────────────────────────────

def _progress_args(step: Any, pct: Any) -> "tuple[str, float]":
    """Runners call progress_callback(stage, fraction) (virality, enhancer) or (fraction, text) (BEEP):
    accept both, like the Desktop mode child (sonya_desktop/modes/child.py). Never raises."""
    if isinstance(step, (int, float)) and not isinstance(step, bool) and not isinstance(pct, (int, float)):
        step, pct = pct, step
    try:
        frac = float(pct) if pct is not None else 0.0
    except (TypeError, ValueError):
        frac = 0.0
    return (str(step) if step is not None else "running"), frac


class _HeartbeatPulse:
    """
    Background thread that calls _do_heartbeat(job_id) every
    HEARTBEAT_INTERVAL_SEC seconds until stop()ped. No separate daemon/
    process — this is one thread inside the same worker process already
    running the job, started right before a long mode run and always
    stopped in a finally so it can never outlive process_job().

    Uses threading.Event.wait() as the sleep, not a recursive
    threading.Timer chain — a single wait() call is interruptible by
    stop() immediately (no waiting out the last partial interval), and
    there's no risk of overlapping timers if start()/stop() were ever
    called twice.
    """

    def __init__(self, job_id: str, interval_sec: int = HEARTBEAT_INTERVAL_SEC) -> None:
        self._job_id = job_id
        self._interval = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        # First beat only after one interval — process_job() has already
        # just set a fresh status (which bumps updated_at) right before
        # starting this, so there's no liveness gap to cover immediately.
        while not self._stop_event.wait(self._interval):
            try:
                _do_heartbeat(self._job_id)
            except Exception:
                # _do_heartbeat() already swallows its own exceptions —
                # this is defense in depth so nothing unexpected can ever
                # kill the pulse thread and silently stop all future
                # beats for the rest of a multi-hour job.
                logger.exception("[worker] heartbeat_pulse_iteration_failed job_id=%s", self._job_id)

    def start(self) -> None:
        if self._thread is not None:
            return  # already running — start() is idempotent, not additive
        self._thread = threading.Thread(
            target=self._run, name=f"heartbeat-{self._job_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None


# ── Core job processing ────────────────────────────────────────────────────────

def _finish_streamer_analyze_job(
    job_id: str,
    user_id: str,
    params: dict,
    analyze_result: Dict[str, Any],
    output_dir: Path,
    t_start: float,
) -> None:
    """
    Completion path for a streamer_phase="analyze" job — replaces steps
    4-6 of process_job() (generic output-file collection) because a JSON
    segments manifest classifies as file_type="enrichment_json"
    (_FILE_TYPES above), never "output", so the generic loop's own
    s3_output_key selection would never find one and always fail this job
    with ALL_UPLOADS_FAILED. The real result of this job — the segments
    themselves — is persisted to streamer_segments via
    _do_submit_streamer_segments(), not via any uploaded file; the JSON
    manifest uploaded below exists only so this generation_jobs row still
    has a normal, inspectable completed artifact like every other job.
    """
    batch_id = params.get("streamer_batch_id")
    if not batch_id:
        _fail(job_id, "STREAMER_BATCH_ID_MISSING",
              "analyze-phase job has no streamer_batch_id in params", retry=False)
        return

    try:
        _do_submit_streamer_segments(batch_id, analyze_result)
    except Exception as exc:
        logger.exception(
            "[worker] streamer_segments_submit_failed job_id=%s batch_id=%s: %s",
            job_id, batch_id, exc,
        )
        _fail(job_id, "STREAMER_SEGMENTS_SUBMIT_FAILED", str(exc)[:500], retry=True)
        return

    _update_status(job_id, JOB_STATUS_UPLOADING_RESULT)
    manifest_path = output_dir / "analysis.json"
    manifest_path.write_text(json.dumps(analyze_result, ensure_ascii=False), encoding="utf-8")
    s3_key = build_output_key(user_id=user_id, job_id=job_id, mode="streamer", filename="analysis.json")
    try:
        upload_file(manifest_path, s3_key, content_type="application/json")
    except Exception as exc:
        logger.error("[worker] analysis_manifest_upload_failed job_id=%s: %s", job_id, exc)
        _fail(job_id, "ANALYSIS_MANIFEST_UPLOAD_FAILED", str(exc)[:500], retry=True)
        return

    processing_ms = int((time.monotonic() - t_start) * 1000)
    segments = analyze_result.get("segments") or []
    _do_complete_job(
        job_id=job_id, s3_output_key=s3_key, clip_count=len(segments),
        processing_ms=processing_ms, enrichment_keys=[],
    )
    logger.info(
        "[worker] streamer_analyze_done job_id=%s batch_id=%s segments=%d ms=%d",
        job_id, batch_id, len(segments), processing_ms,
    )


def _fail(job_id: str, error_code: str, error_message: str, retry: bool) -> None:
    logger.error(
        "[worker] job_failed job_id=%s code=%s retry=%s msg=%s",
        job_id, error_code, retry, error_message[:200],
    )
    try:
        _do_fail_job(job_id, error_code, error_message, retry)
    except Exception as exc:
        logger.error("[worker] fail_job_backend_error job_id=%s: %s", job_id, exc)


def process_job(job: dict, worker_id: str) -> None:
    job_id  = job["id"]
    mode    = job["mode"]
    user_id = job["user_id"]
    params  = job.get("params") or {}
    s3_input = job["s3_input_key"]
    trace_id = new_trace_id()

    logger.info(
        "[worker] starting job_id=%s mode=%s worker=%s backend=%s trace=%s",
        job_id, mode, worker_id, _WORKER_MODE, trace_id,
    )
    # Mode policy first: a disabled mode (Trailer) never downloads input or models.
    try:
        assert_mode_allowed(mode)
    except ModeDisabledError as exc:
        logger.warning("[worker] job_id=%s refused: %s", job_id, exc)
        _fail(job_id, DISABLED_CODE, str(exc), retry=False)
        return

    t_start = time.monotonic()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"sonya_{job_id}_"))
    _ACTIVE_JOB["id"] = job_id
    if not _FIRST_JOB["claimed"]:
        _FIRST_JOB["claimed"] = True
        _timeline("first_job_claimed", mode=mode)

    try:
        # ── 1. Download input ─────────────────────────────────────────────────
        _update_status(job_id, JOB_STATUS_DOWNLOADING)
        input_ext  = Path(s3_input).suffix or ".mp4"
        input_path = tmp_dir / f"input{input_ext}"
        sf = params.get("source_fetch")
        if mode == "sonya_gen":
            input_path = tmp_dir / "no_input"      # text -> image: there is no input video
            input_path.touch()
        elif isinstance(sf, dict) and sf.get("url"):
            # The VPS could not fetch this link (YouTube etc.): download here, upload to s3_input_key, continue as usual.
            from scripts.source_fetch import SourceFetchError, fetch_source
            try:
                got = fetch_source(sf, mode, s3_input, tmp_dir)
                input_path = got["path"]
            except SourceFetchError as exc:
                _fail(job_id, exc.code, exc.message, retry=False)
                return
            except Exception as exc:
                logger.exception("[worker] source_fetch crashed job_id=%s", job_id)
                _fail(job_id, "SOURCE_DOWNLOAD_FAILED", "Не удалось скачать видео по ссылке. Попробуйте позже.", retry=True)
                return
        try:
            if mode != "sonya_gen" and not (isinstance(sf, dict) and sf.get("url")):
                download_file(s3_input, input_path)
            logger.info(
                "[worker] input downloaded: %s (%d bytes)",
                input_path, input_path.stat().st_size,
            )
        except Exception as exc:
            _fail(job_id, "INPUT_DOWNLOAD_FAILED", str(exc), retry=True)
            return

        # ── 2. Ensure models ──────────────────────────────────────────────────
        _update_status(job_id, JOB_STATUS_MODEL_DOWNLOADING)
        try:
            ensure_models_for_mode(mode)
        except RuntimeError as exc:
            _fail(job_id, "MODEL_DOWNLOAD_FAILED", str(exc), retry=False)
            return
        except Exception as exc:
            logger.warning(
                "[worker] model_download warning job_id=%s: %s — continuing", job_id, exc
            )

        # ── 3. Run mode ───────────────────────────────────────────────────────
        _update_status(job_id, JOB_STATUS_MODE_RUNNING)
        if not _FIRST_JOB["processing"]:
            _FIRST_JOB["processing"] = True
            _timeline("first_job_processing")
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        _STEP_STATUS = {
            "analyzing":  "analyzing",
            "yolo":       "yolo",
            "scripting":  "scripting",
            "tts":        "tts",
            "subtitles":  "subtitles",
            "assembling": "assembling",
        }

        def _progress(step=None, pct=None, *_a, **_kw) -> None:
            step, pct = _progress_args(step, pct)
            logger.info(
                "[worker] job_id=%s step=%s progress=%.0f%%", job_id, step, pct * 100
            )
            mapped = _STEP_STATUS.get(step.lower(), JOB_STATUS_MODE_RUNNING)
            try:
                _update_status(job_id, mapped)
            except Exception:
                pass

        # streamer batch flow (see docs/SONYA_AUDIT.md batch-flow brief):
        # a "streamer" job's params carry an explicit streamer_phase
        # discriminator when it's part of a two-phase batch (analyze once,
        # compose per selected segment) instead of the old single-shot
        # run() path. Absent (None) for every other job — including a
        # plain streamer job with no batch behind it, e.g. run via the
        # legacy /api/generation/jobs path — which keeps the exact
        # pre-existing behavior below.
        streamer_phase = params.get("streamer_phase") if mode == "streamer" else None

        # Heartbeat only around the actual mode run — the one phase that
        # can legitimately run long (e.g. streamer's multi-hour enrichment)
        # without a status transition in between. Started right before,
        # always stopped in finally so it can never survive this call,
        # success or failure.
        heartbeat = _HeartbeatPulse(job_id)
        heartbeat.start()
        try:
            if streamer_phase == "analyze":
                from modes.streamer.runner import analyze as _streamer_analyze
                analyze_result = _streamer_analyze(
                    input_video_path=str(input_path),
                    output_dir=str(output_dir),
                    params=params,
                    progress_callback=_progress,
                )
            elif streamer_phase == "compose":
                # compose_one() writes exactly one clip; falls through to
                # the exact same generic output-collection/upload/complete
                # path below as every other mode's result — no special
                # handling needed past this point for a compose job.
                from modes.streamer.runner import compose_one as _streamer_compose_one
                out_path = str(output_dir / "clip.mp4")
                _streamer_compose_one(
                    input_video_path=str(input_path),
                    output_path=out_path,
                    segment=params.get("streamer_segment") or {},
                    crop_hints=params.get("streamer_crop_hints"),
                )
                result = {"clips": [out_path], "mode": "streamer", "warnings": []}
            elif mode == "sonya_gen":
                runner = get_runner(mode)
                result = runner(input_video_path="", output_dir=str(output_dir), params=params, progress_callback=_progress)
            else:
                # Desktop ingest: VFR / rotated / non-H.264 sources are normalized before the mode
                # (scripts/shared/source_normalize.py; virality + beep only)
                from scripts.shared.source_normalize import normalize_source
                norm = normalize_source(str(input_path), str(output_dir.parent / "ingest"), mode)
                if norm["normalized"]:
                    logger.info("[worker] job_id=%s source normalized: %s", job_id, ",".join(norm["reasons"]))
                runner = get_runner(mode)
                result = runner(
                    input_video_path=norm["path"],
                    output_dir=str(output_dir),
                    params=params,
                    progress_callback=_progress,
                )
        except Exception as exc:
            logger.exception(
                "[worker] runner_failed job_id=%s mode=%s: %s", job_id, mode, exc
            )
            _fail(job_id, "RUNNER_FAILED", str(exc)[:500], retry=False)
            return
        finally:
            heartbeat.stop()

        if streamer_phase == "analyze":
            _finish_streamer_analyze_job(job_id, user_id, params, analyze_result, output_dir, t_start)
            return

        # ── 4. Collect output files ───────────────────────────────────────────
        output_files = [f for f in output_dir.rglob("*") if f.is_file()]
        if not output_files:
            _fail(job_id, "NO_OUTPUT_FILES", "Runner produced no output files", retry=False)
            return

        # ── 4b. Light preview copy (720p ~2.5 Mb/s, faststart) + poster per clip ─
        if mode in _PREVIEW_MODES:
            for _clip in sorted(output_dir.rglob("*.mp4")):
                if not _clip.stem.endswith("_preview"):
                    _make_preview_assets(_clip)
            output_files = [f for f in output_dir.rglob("*") if f.is_file()]

        # ── 5. Upload outputs to S3 ───────────────────────────────────────────
        _update_status(job_id, JOB_STATUS_UPLOADING_RESULT)
        s3_output_key = None
        enrichment_keys = result.get("enrichment_keys", []) if isinstance(result, dict) else []
        clip_count = 0

        for f in output_files:
            ext = f.suffix.lower()
            ct  = _CONTENT_TYPES.get(ext, "application/octet-stream")
            ft  = _FILE_TYPES.get(ext, "output")
            if f.stem.endswith("_preview") and ext == ".mp4":
                ft = "preview"          # never counted as a clip
            elif f.stem.endswith("_poster") and ext == ".jpg":
                ft = "thumbnail"
            s3k = build_output_key(
                user_id=user_id, job_id=job_id, mode=mode, filename=f.name
            )
            try:
                upload_file(f, s3k, content_type=ct)
                logger.info("[worker] uploaded %s → %s", f.name, s3k)
            except Exception as exc:
                logger.error("[worker] upload_failed file=%s: %s", f.name, exc)
                continue

            _do_add_file(
                job_id=job_id, user_id=user_id,
                file_type=ft, s3_key=s3k,
                filename=f.name, content_type=ct,
                size_bytes=f.stat().st_size,
            )

            if ft == "output":
                clip_count += 1
                if s3_output_key is None:
                    s3_output_key = s3k

        if s3_output_key is None:
            _fail(job_id, "ALL_UPLOADS_FAILED", "No output files could be uploaded", retry=True)
            return

        # ── 6. Complete job ───────────────────────────────────────────────────
        processing_ms = int((time.monotonic() - t_start) * 1000)
        _do_complete_job(
            job_id=job_id,
            s3_output_key=s3_output_key,
            clip_count=clip_count,
            processing_ms=processing_ms,
            enrichment_keys=enrichment_keys,
        )
        logger.info(
            "[worker] job_done job_id=%s clips=%d ms=%d backend=%s trace=%s",
            job_id, clip_count, processing_ms, _WORKER_MODE, trace_id,
        )

    finally:
        _ACTIVE_JOB["id"] = None
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("[worker] cleaned up tmp_dir=%s", tmp_dir)


# ── Run modes ──────────────────────────────────────────────────────────────────

def run_once(job_id: str, worker_id: str) -> None:
    """Claim a specific job by ID and process it, then exit."""
    if _WORKER_MODE == "api":
        job = _get_api_client().claim_job(job_id=job_id, worker_id=worker_id)
        if not job:
            raise SystemExit(
                f"[worker] API claim returned no job for job_id={job_id} — "
                "job may already be claimed, completed, or not exist"
            )
    else:
        job = _db_claim_specific(job_id, worker_id=worker_id)
        if not job:
            job = _db_get_job(job_id)
            if not job:
                raise SystemExit(f"Job {job_id} not found")
            if job.get("status") == JOB_STATUS_CLAIMED and job.get("worker_id") == worker_id:
                logger.info("[worker] re-processing already claimed job_id=%s", job_id)
            else:
                raise SystemExit(
                    f"Job {job_id} is in status={job.get('status')} — cannot claim"
                )

    process_job(job, worker_id)


def run_poll(worker_id: str, modes: Optional[List[str]] = None) -> None:
    """Poll for jobs continuously until SIGTERM / KeyboardInterrupt."""
    logger.info(
        "[worker] poll mode started worker_id=%s backend=%s interval=%ds",
        worker_id, _WORKER_MODE, POLL_INTERVAL,
    )
    last_stale_check = 0.0

    # Always send an explicit allow-list: an empty filter would mean "any mode".
    allowed = claimable_modes()
    modes = [m for m in (modes or allowed) if m in allowed]
    if not modes:
        raise SystemExit("[worker] no runnable modes after applying mode policy")

    if os.environ.get("SONYA_REQUIRE_READY", "") == "1" and os.environ.get("SONYA_PREWARM", "1") == "1":
        _prewarm(modes)

    # Readiness gate: never claim a real job before bootstrap+readiness wrote the ready file.
    ready_file = os.environ.get("SONYA_READY_FILE", "")
    if os.environ.get("SONYA_REQUIRE_READY", "") == "1" and not (ready_file and Path(ready_file).is_file()):
        raise SystemExit("[worker] refusing to claim jobs: readiness file missing")

    while not _STOP.is_set():
        # Stale-job requeue — only in DB mode (API server handles this itself)
        if _WORKER_MODE == "db":
            now = time.monotonic()
            if now - last_stale_check > STALE_INTERVAL:
                try:
                    requeued = _db_requeue_stale()
                    if requeued:
                        logger.info("[worker] requeued %d stale jobs", requeued)
                except Exception as exc:
                    logger.warning("[worker] requeue_stale_jobs failed: %s", exc)
                last_stale_check = now

        # Claim next job
        try:
            if _WORKER_MODE == "api":
                job = _get_api_client().claim_job(job_id=None, worker_id=worker_id, modes=modes)
            else:
                job = _db_claim_next(worker_id=worker_id, modes=modes)

            if job:
                process_job(job, worker_id)
            else:
                _STOP.wait(POLL_INTERVAL)

        except Exception as exc:
            logger.error("[worker] poll_error: %s", exc)
            _STOP.wait(POLL_INTERVAL)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SONYA GPU Worker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Process a single job and exit")
    group.add_argument("--poll", action="store_true", help="Poll for jobs continuously")
    parser.add_argument("--job-id",    help="Job ID (required with --once)")
    parser.add_argument("--worker-id", help="Worker identifier (default: hostname-pid)")
    parser.add_argument(
        "--modes", nargs="+",
        help="Whitelist of modes to process (--poll only)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    wid = args.worker_id or _worker_id()
    logger.info(
        "[worker] starting worker_id=%s backend_mode=%s python=%s host=%s",
        wid, _WORKER_MODE, sys.version.split()[0], platform.node(),
    )

    if args.once:
        if not args.job_id:
            parser.error("--once requires --job-id")
        run_once(args.job_id, worker_id=wid)
    else:
        run_poll(worker_id=wid, modes=args.modes)


if __name__ == "__main__":
    main()
