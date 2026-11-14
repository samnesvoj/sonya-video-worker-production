"""
prod_generation_api.py
======================
Full production FastAPI for SONYA generation pipeline.

Browser endpoints (identity resolved from the sonya_session HttpOnly
cookie -- see scripts/security.py::get_current_user -- the X-User-Id
header is never trusted for these):
  GET  /health
  GET  /api/health
  POST /api/generation/jobs
  POST /api/trailer/jobs
  GET  /api/generation/jobs/{job_id}
  GET  /api/generation/jobs/{job_id}/result-url
  GET  /api/generation/jobs          (list user jobs)
  POST /api/generation/jobs/from-url         (start URL ingestion)
  GET  /api/generation/jobs/from-url/{ingest_id}  (poll ingestion status)

Auth + billing endpoints (see scripts/auth_routes.py):
  GET  /api/auth/me
  POST /api/auth/request-code
  POST /api/auth/verify-code
  POST /api/auth/logout
  GET  /api/billing/subscription-status

Billing endpoints (see scripts/payment_routes.py; provider = env PAYMENT_PROVIDER):
  POST /api/billing/checkout                 (browser, plan_id -> pending payment + provider redirect_url)
  POST /api/billing/robokassa/result         (LEGACY Robokassa ResultURL -- confirms old Robokassa payments only)
  GET  /api/billing/payment-status           (browser, polled from payment/success.html + fail.html)
  POST /api/billing/selfwork/callback        (Самозанятые.рф callback -- confirmed via its GET /status)
  POST /api/billing/yookassa/webhook         (ЮKassa HTTP notification -- confirmed via GET /v3/payments/{id})

Finance Worker receipt outbox (see scripts/fiscal_routes.py; HMAC-signed, 404 unless FISCAL_OUTBOX_ENABLED):
  GET /api/internal/fiscal/health, POST /api/internal/fiscal/claim, GET /api/internal/fiscal/receipts/{id},
  POST /api/internal/fiscal/receipts/{id}/registered | /retry | /manual-required

@sonya_group_bot account-linking endpoints (see scripts/telegram_routes.py):
  POST /api/telegram/link-token              (browser, one-time deep-link token)
  POST /api/telegram/webhook                 (Telegram servers -- X-Telegram-Bot-Api-Secret-Token, not session)
  POST /api/telegram/unlink                  (browser)
  GET  /api/telegram/status                  (browser)

Streamer batch flow endpoints (see scripts/streamer_routes.py):
  POST /api/streamer/batches                       (browser, URL or file source)
  GET  /api/streamer/batches/{batch_id}             (browser, owner only)
  GET  /api/streamer/batches                        (browser, list own batches)
  POST /api/streamer/batches/{batch_id}/selection   (browser, confirm segments -> compose jobs)
  POST /api/worker/streamer/batches/{batch_id}/analysis-result  (worker, WORKER_SECRET)

Worker-internal endpoints (require Authorization: Bearer WORKER_SECRET --
unchanged, never cookie/session based):
  POST /api/worker/claim
  POST /api/worker/jobs/{job_id}/status
  POST /api/worker/jobs/{job_id}/heartbeat   (liveness only -- no status change; migration 011)
  POST /api/worker/jobs/{job_id}/complete
  POST /api/worker/jobs/{job_id}/fail
  POST /api/worker/jobs/{job_id}/files

URL-based ingestion (POST .../from-url) downloads via plain yt-dlp
(YouTube/VK/Twitch) or a direct HTTP(S) video link only -- no cookies, no
authenticated/paid-content bypass, SSRF-guarded (scripts/url_ingest.py) --
then feeds the result through the exact same validate/S3-upload/create-job
code path a browser file upload uses. See scripts/url_ingest.py for why
SONYA's recovered scripts/shared/download/downloader.py is NOT used here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import scripts.gpu_orchestrator as orchestrator
from scripts import feature_flags
from scripts.prod_job_store import (
    JOB_STATUS_QUEUED,
    JOB_STATUS_CLAIMED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELLED,
    add_job_file,
    claim_next_pending_job,
    claim_specific_job,
    complete_job,
    create_job,
    create_job_idempotent,
    create_job_with_quota,
    fail_job,
    get_job,
    get_job_by_idempotency_key,
    list_job_files,
    list_user_jobs,
    reconcile_streamer_batch,
    set_streamer_batch_status,
    touch_job_heartbeat,
    update_job_status,
    StreamerBatchTransitionError,
)
from scripts.prod_s3_storage import (
    build_input_key,
    build_output_key,
    delete_object,
    generate_presigned_get_url,
    upload_bytes,
)
from scripts.auth_routes import router as auth_router
from scripts.payment_routes import router as payment_router
from scripts.fiscal_routes import router as fiscal_router
from scripts.telegram_routes import router as telegram_router
from scripts.streamer_routes import router as streamer_router
from scripts.pipeline_routes import router as pipeline_router
from scripts.streamer_notify import notify_streamer_batch_completion
from scripts.quota_guard import check_user_quota
from scripts.rate_limiter import RateLimiter
from scripts.security import (
    assert_job_owner,
    get_allowed_origins,
    get_current_user,
    new_trace_id,
    safe_error,
    verify_browser_origin,
    verify_worker_secret,
)

# ── Priority table (server-side, from the authenticated user's plan_type in
# Postgres -- never from a client-supplied header) ──────────────────────────
_PLAN_PRIORITY: dict[str, int] = {
    "admin":   1000,
    "pro":      500,
    "paid":     300,
    "free":     100,
    "unknown":  100,
}


def _resolve_priority(user: dict) -> tuple[int, str]:
    """Return (priority, plan) from the authenticated user's DB record."""
    plan = (user.get("plan_type") or "unknown").lower().strip()
    if plan not in _PLAN_PRIORITY:
        plan = "unknown"
    return _PLAN_PRIORITY[plan], plan


def _is_pro_active(user: dict) -> bool:
    """
    Whether a LEGACY (pre-016, 500 ₽) Pro plan is currently in effect -- not
    just whether plan_type happens to say "pro". A payment that expired
    without renewal leaves plan_type="pro" on the row (nothing resets it),
    so plan_active_until must be checked too. New public plans never set
    plan_type; their rights are resolved by scripts/entitlements.py.
    """
    return entitlements.legacy_pro_active(user)


def _resolve_entitlement(user: dict, mode: str, trace_id: str) -> "entitlements.Entitlement":
    """Pre-I/O access decision for one operation (no debit). Raises the
    402 payment-required flow with a concrete `code` (see entitlements.py);
    the atomic debit is create_job_with_quota()."""
    try:
        subscriptions = entitlements.get_user_subscriptions(str(user["id"]))
    except Exception as exc:
        logger.error("[api] subscriptions_lookup_failed user_id=%s trace_id=%s error_type=%s",
                     user["id"], trace_id, type(exc).__name__)
        raise safe_error("db_error", 500, trace_id)
    try:
        return entitlements.resolve_entitlement(user, mode, subscriptions)
    except entitlements.EntitlementDenied as denied:
        raise HTTPException(status_code=denied.status_code, detail=denied.detail(trace_id))


def _priority_for(ent: "entitlements.Entitlement", priority: int, plan: str) -> tuple[int, str]:
    # A paid subscription gets the same queue priority as legacy Pro.
    if ent.kind == "subscription":
        return _PLAN_PRIORITY["pro"], "pro"
    return priority, plan


def _enforce_duration_of_bytes(ent: "entitlements.Entitlement", content: bytes, ext: str,
                               trace_id: str) -> None:
    """Plan source-length cap for an in-memory upload (writes a temp copy
    for ffprobe only when a cap applies). Raises before any S3/DB work."""
    if ent.max_source_sec is None:
        return
    with tempfile.NamedTemporaryFile(suffix=ext or ".mp4") as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            entitlements.measure_and_enforce(ent, tmp.name)
        except entitlements.EntitlementDenied as denied:
            raise HTTPException(status_code=denied.status_code, detail=denied.detail(trace_id))


def _plan_limit_reached_detail(ent: "entitlements.Entitlement", trace_id: str) -> dict:
    return entitlements.EntitlementDenied(
        entitlements.PLAN_LIMIT_REACHED,
        "Лимит тарифа исчерпан или период подписки закончился.",
        plan_mode=ent.plan_mode, plan_id=ent.plan_id,
    ).detail(trace_id)


from scripts.security_audit import (
    EVT_JOB_CREATED,
    EVT_JOB_CLAIMED,
    EVT_JOB_COMPLETED,
    EVT_JOB_FAILED,
    EVT_UPLOAD_REJECTED,
    audit,
)
from scripts.upload_security import validate_upload
from scripts import entitlements, url_ingest

logger = logging.getLogger(__name__)

# ── App ————————————————————————————————————————————————————————————————————————

app = FastAPI(
    title="SONYA Generation API",
    version="1.0.0",
    docs_url="/docs" if os.environ.get("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
)

_cors_origins = get_allowed_origins()
if "*" in _cors_origins:
    # Session auth relies on cookies -> allow_credentials=True below. Browsers
    # refuse "*" + credentials anyway, but we fail fast here rather than ship
    # a CORS config that silently doesn't work / is unsafe if it did.
    raise RuntimeError(
        "CORS_ORIGINS='*' cannot be combined with cookie-based session auth "
        "(allow_credentials=True). Set explicit origins, e.g. "
        "CORS_ORIGINS=https://sonya-e.com,https://www.sonya-e.com"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,  # required so the browser sends/receives the sonya_session cookie
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-User-Id", "X-Request-Id"],
)

# Auth + billing endpoints (see scripts/auth_routes.py)
app.include_router(auth_router)

# Billing endpoints: checkout, payment status, legacy Robokassa ResultURL (see scripts/payment_routes.py)
app.include_router(payment_router)
# Finance Worker receipt outbox -- 404 unless FISCAL_OUTBOX_ENABLED (scripts/fiscal_routes.py).
app.include_router(fiscal_router)

# @sonya_group_bot account-linking endpoints (see scripts/telegram_routes.py)
app.include_router(telegram_router)

# Streamer batch flow endpoints (see scripts/streamer_routes.py)
app.include_router(streamer_router)

# SONYA Pipeline lead form — public, rate-limited (see scripts/pipeline_routes.py)
app.include_router(pipeline_router)

# ── Mode registry ——————————————————————————————————————————————————————————————

ALLOWED_MODES = {
    "trailer_film_breaker",
    "virality",
    "stories",
    "educational",
    "streamer",
    "sonya_gen",
}

# ── Idempotency (POST /api/generation/jobs) ——————————————————————————————————————
# Optional `Idempotency-Key` header. Absent -> legacy behavior (always a new
# job, matches every client that doesn't send it yet). Present -> validated
# format/length, then used with an atomic INSERT ... ON CONFLICT DO NOTHING
# (migration 009) to guarantee at most one job per (user_id, key) even under
# concurrent requests. See create_generation_job() below.

_IDEMPOTENCY_KEY_MAX_LEN = 255
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9_\-.:]{1,255}$")

# Fields that are provably non-semantic -- client diagnostics / routing
# markers, never used to decide what the job does or how it's processed.
# Removed from the fingerprint so a version bump to `frontend` (say) never
# turns a genuine retry into a false idempotency_key_conflict. Every other
# field in `params` -- known today or added by a future frontend change --
# is included by default; see _compute_idempotency_fingerprint().
_FINGERPRINT_EXCLUDED_TOP_LEVEL_KEYS = ("frontend", "production_endpoint")


def _validate_idempotency_key(raw: Optional[str], trace_id: str) -> Optional[str]:
    """
    Returns the normalized key, or None if the header was absent entirely
    (legacy behavior — always creates a new job). Raises HTTPException(400)
    for a header that IS present but invalid: empty after trim, too long,
    or containing characters outside the allowed set.
    """
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_idempotency_key",
                    "message": "Idempotency-Key must not be empty", "trace_id": trace_id},
        )
    if len(key) > _IDEMPOTENCY_KEY_MAX_LEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_idempotency_key",
                    "message": f"Idempotency-Key must be at most {_IDEMPOTENCY_KEY_MAX_LEN} characters",
                    "trace_id": trace_id},
        )
    if not _IDEMPOTENCY_KEY_RE.match(key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_idempotency_key",
                    "message": "Idempotency-Key contains invalid characters "
                                "(allowed: letters, digits, '_', '-', '.', ':')",
                    "trace_id": trace_id},
        )
    return key


def _compute_idempotency_fingerprint(mode: str, file_content: bytes, parsed_params: Dict[str, Any]) -> str:
    """
    Canonical fingerprint of everything that defines this job.

    Deliberately NOT a hand-picked allowlist of "processing" fields --
    that drifts silently the moment a new field is added to params and
    forgotten here. Instead: take the actual normalized params dict that
    gets stored in the DB, drop only the two provably non-semantic
    top-level keys and the one non-semantic nested key (source.fileName --
    a display label; not used for s3_key or processing, and the file's
    real identity is already covered by the content hash below), and
    fingerprint everything else, whatever it is.

    No field-level normalization beyond that: a missing key and an empty
    string are NOT treated as equal (json.dumps distinguishes None/absent
    from "" -- nothing here coerces one into the other).

    canonical JSON uses sort_keys=True + compact separators so key order
    in the original request never affects the result.
    """
    normalized = copy.deepcopy(parsed_params)
    for key in _FINGERPRINT_EXCLUDED_TOP_LEVEL_KEYS:
        normalized.pop(key, None)
    source = normalized.get("source")
    if isinstance(source, dict):
        source.pop("fileName", None)

    canonical = {
        "mode": mode,
        "file_sha256": hashlib.sha256(file_content).hexdigest(),
        "params": normalized,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ── Rate limiters ——————————————————————————————————————————————————————————————

_upload_limiter = RateLimiter(key_prefix="upload", limit=20, window_seconds=3600, key_by="session")
_api_limiter    = RateLimiter(key_prefix="api",    limit=120, window_seconds=60,   key_by="session")

# ── URL ingestion status store ————————————————————————————————————————————————
# In-memory only: the API currently runs as a single uvicorn process (see
# deploy/COMMANDS_VPS.md), so this is safe and needs no migration. If the
# API is ever scaled to multiple processes/replicas, this must move to
# Postgres or Redis -- a client polling GET .../from-url/{id} could
# otherwise land on a worker process that never ran the download.
_URL_INGEST_TTL_SEC = 3600
_url_ingest_lock = threading.Lock()
_url_ingest_store: Dict[str, Dict[str, Any]] = {}


def _ingest_set(ingest_id: str, **fields: Any) -> None:
    with _url_ingest_lock:
        entry = _url_ingest_store.setdefault(ingest_id, {"created_at": time.time()})
        entry.update(fields)
        entry["updated_at"] = time.time()
        # Best-effort purge of stale entries so this dict can't grow
        # unbounded over the process lifetime.
        cutoff = time.time() - _URL_INGEST_TTL_SEC
        stale = [k for k, v in _url_ingest_store.items() if v.get("created_at", 0) < cutoff]
        for k in stale:
            _url_ingest_store.pop(k, None)


def _ingest_get(ingest_id: str) -> Optional[Dict[str, Any]]:
    with _url_ingest_lock:
        entry = _url_ingest_store.get(ingest_id)
        return dict(entry) if entry is not None else None

# ── Pydantic models ————————————————————————————————————————————————————————————

class JobResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    created_at: Optional[str] = None


class JobStatusUpdate(BaseModel):
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None


class JobCompleteRequest(BaseModel):
    s3_output_key: str
    clip_count: Optional[int] = None
    processing_ms: Optional[int] = None
    enrichment_keys: Optional[List[str]] = None


class JobFailRequest(BaseModel):
    error_code: str = "UNKNOWN_ERROR"
    error_message: str = "Job failed"
    retry: bool = True


class JobFileRequest(BaseModel):
    file_type: str
    s3_key: str
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    mode: str
    params: Optional[Dict[str, Any]] = None


class WorkerClaimRequest(BaseModel):
    worker_id: str
    job_id: Optional[str] = None
    modes: Optional[List[str]] = None


# ── Health ——————————————————————————————————————————————————————————————————————

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "sonya-generation-api", "version": "1.0.0"}


# ── Feature flags ——————————————————————————————————————————————————————————————

def _reject_if_mode_disabled(mode: str, request: Request, user_id: str, trace_id: str) -> None:  # noqa: ARG001
    """403 for a job mode that is switched off (SONYA_TRAILER_ENABLED)."""
    if not feature_flags.job_mode_disabled(mode):
        return
    logger.info("[api] feature_disabled mode=%s user_id=%s trace_id=%s", mode, user_id, trace_id)
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail=feature_flags.disabled_detail(trace_id))


@app.get("/api/features")
async def features():
    """Public, booleans only: which switchable features are on right now."""
    return feature_flags.public_flags()


# ── SONYA Gen (images): ADMIN ONLY for now (no tariff has it) ─────────────────────
# Admin = the existing mechanism: users.plan_type == 'admin' (same as auth.js / _PLAN_PRIORITY). No credits, no quota, no billing tables touched.
GEN_IMAGE_ASPECTS = ("1:1", "4:5", "3:4", "2:3", "9:16", "16:9", "3:2", "5:4")


def _is_gen_admin(user: dict) -> bool:
    return str(user.get("plan_type") or "").lower().strip() == "admin"


def _require_gen_admin(user: dict, trace_id: str) -> None:
    if not _is_gen_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail={"error": "coming_soon", "message": "скоро", "trace_id": trace_id})


class GenImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    aspect: str = "1:1"
    size: str = "standard"
    count: int = Field(1, ge=1, le=4)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)


@app.get("/api/gen/access")
async def gen_access(user: dict = Depends(get_current_user)):
    """Frontend: is the SONYA Gen (images) tab live for this user? Only {"image": bool}."""
    return {"image": _is_gen_admin(user)}


@app.post("/api/gen/image", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_gen_image_job(
    body: GenImageRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    _origin: None = Depends(verify_browser_origin),
    _rl: None = Depends(_upload_limiter),
):
    trace_id = new_trace_id()
    _require_gen_admin(user, trace_id)
    if body.aspect not in GEN_IMAGE_ASPECTS or body.size not in ("draft", "standard"):
        raise HTTPException(status_code=400, detail={"error": "invalid_params", "trace_id": trace_id})
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "invalid_params", "trace_id": trace_id})
    user_id = str(user["id"])
    job_id = str(uuid.uuid4())
    params = {"prompt": prompt, "aspect": body.aspect, "size": body.size, "count": body.count}
    if body.seed is not None:
        params["seed"] = body.seed
    try:
        result = create_job_with_quota(
            job_id=job_id, user_id=user_id, mode="sonya_gen", params=params,
            s3_input_key=f"gen/{user_id}/{job_id}/none",     # text -> image: no input object exists
            idempotency_key=None, idempotency_fingerprint=None,
            queue_priority=_PLAN_PRIORITY["admin"],
            bypass_quota=True,                                # admin: no credits, nothing debited
            subscription_id=None,
        )
    except Exception as exc:
        logger.error("[api] gen_create_failed trace_id=%s: %s", trace_id, exc)
        raise safe_error("db_error", 500, trace_id)
    logger.info("[api] gen_job_created job_id=%s user_id=%s count=%d", job_id, user_id, body.count)
    job = get_job(job_id)
    return JobResponse(job_id=job_id, status=JOB_STATUS_QUEUED, mode="sonya_gen",
                       created_at=str(job.get("created_at")) if job else None)


# ── Public: Job creation ————————————————————————————————————————————————————————

@app.post("/api/generation/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_generation_job(
    request: Request,
    response: Response,
    mode: str = Form(...),
    file: UploadFile = File(...),
    params: Optional[str] = Form(None),
    idempotency_key_header: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: dict = Depends(get_current_user),
    _origin: None = Depends(verify_browser_origin),
    _rl: None = Depends(_upload_limiter),
):
    user_id  = str(user["id"])
    trace_id = new_trace_id()

    # Priority assigned server-side, from the authenticated user's plan_type
    # in Postgres -- clients cannot override this.
    priority, plan = _resolve_priority(user)

    # Mode validation
    if mode not in ALLOWED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_mode", "allowed": sorted(ALLOWED_MODES), "trace_id": trace_id},
        )

    # SONYA Gen (images) has no upload and is admin-only for now: use POST /api/gen/image.
    if mode == "sonya_gen":
        _require_gen_admin(user, trace_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"error": "use_gen_endpoint", "message": "POST /api/gen/image", "trace_id": trace_id})

    # Feature flag: Trailer is temporarily offline (scripts/feature_flags.py).
    # Refused before quota, entitlement, upload or S3 -- nothing is created
    # or debited.
    _reject_if_mode_disabled(mode, request, user_id, trace_id)

    # Idempotency-Key format validation -- cheap, local, done before any
    # quota/DB/S3 work. None (header absent) = legacy behavior.
    idempotency_key = _validate_idempotency_key(idempotency_key_header, trace_id)

    # Params parsing
    parsed_params: Dict[str, Any] = {}
    if params:
        try:
            parsed_params = json.loads(params)
            if not isinstance(parsed_params, dict):
                raise ValueError("params must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "invalid_params", "detail": str(exc), "trace_id": trace_id},
            )

    # Quota check
    check_user_quota(user_id)

    # Entitlement (plan / mode / ops left / free quota) -- fast-fail only,
    # from the user + subscription snapshot loaded for this request. Not the
    # enforcement point: create_job_with_quota() below re-checks atomically
    # under the user's row lock and is what actually prevents overspending.
    # This early check exists to skip a wasted upload for a request that is
    # already known to be refused. Nothing is debited here.
    ent = _resolve_entitlement(user, mode, trace_id)
    priority, plan = _priority_for(ent, priority, plan)

    # Upload validation (magic bytes, size, extension) — mode-conditional
    # size cap: streamer allows long-form VODs, every other mode keeps the
    # original MAX_UPLOAD_SIZE_MB default (url_ingest._max_bytes is the one
    # canonical place this is computed; see PHASE A audit).
    try:
        content, safe_name = await validate_upload(file, max_size_bytes=url_ingest._max_bytes(mode))
    except HTTPException as exc:
        audit(EVT_UPLOAD_REJECTED, user_id=user_id, trace_id=trace_id,
              details={"mode": mode, "reason": str(exc.detail)},
              ip_address=request.client.host if request.client else None)
        raise

    # Paid-plan source length, measured on the bytes we actually received
    # (never a client-sent duration) -- before S3 and before any debit.
    _enforce_duration_of_bytes(ent, content, Path(safe_name).suffix, trace_id)

    idempotency_fingerprint = (
        _compute_idempotency_fingerprint(mode, content, parsed_params)
        if idempotency_key is not None else None
    )

    # Upload to S3. This happens unconditionally, before we know whether
    # this request will win an Idempotency-Key race -- if it loses, this
    # upload is orphaned and cleaned up below (never the winning job's
    # s3_input_key, only ever this request's own fresh key).
    ext     = Path(safe_name).suffix or ".mp4"
    job_id  = str(uuid.uuid4())
    s3_key  = build_input_key(user_id=user_id, job_id=job_id, mode=mode, ext=ext)

    try:
        upload_bytes(content, s3_key, content_type=file.content_type or "video/mp4")
    except Exception as exc:
        logger.error("[api] s3_upload_failed job_id=%s trace_id=%s: %s", job_id, trace_id, exc)
        raise safe_error("storage_error", 500, trace_id)

    # Atomic create-or-detect-conflict-or-quota-exceeded, all in one DB
    # transaction (see prod_job_store.create_job_with_quota): idempotency-key
    # lookup, free-plan quota reservation, and the job insert either all
    # commit together or all roll back together. This is the actual
    # enforcement point for free_video_limit -- a replay of an
    # already-accepted Idempotency-Key never touches quota, and concurrent
    # requests can never jointly consume more than free_video_limit units
    # (see docs/SONYA_AUDIT.md review notes for why the previous
    # check-then-increment approach raced). GPU is never triggered here; the
    # dispatcher service picks up queued jobs and calls the GPU orchestrator
    # asynchronously.
    try:
        result = create_job_with_quota(
            job_id=job_id,
            user_id=user_id,
            mode=mode,
            params=parsed_params,
            s3_input_key=s3_key,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
            queue_priority=priority,  # existing column (migration 003)
            bypass_quota=ent.kind == "legacy_pro",
            subscription_id=ent.subscription_id,
        )
    except Exception as exc:
        logger.error("[api] create_job_failed trace_id=%s: %s", trace_id, exc)
        delete_object(s3_key)  # best-effort -- this request's own upload, never an existing job's
        raise safe_error("db_error", 500, trace_id)

    if result["outcome"] == "quota_exceeded":
        # Lost the race to the atomic quota reservation -- the early
        # fast-fail check above didn't catch this (another concurrent
        # request consumed the last slot in between). This request's own
        # upload is now orphaned.
        delete_object(s3_key)
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={"error": "payment_required", "code": "FREE_PLAN_USED",
                    "message": "Бесплатная генерация уже использована. Выберите тариф SONYA.",
                    "trace_id": trace_id},
        )

    if result["outcome"] == "plan_limit_reached":
        # Same race, subscription flavor: the period's last operation (or
        # the period itself) ran out between the fast-fail check and the
        # atomic debit. Nothing was debited for this request.
        delete_object(s3_key)
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            detail=_plan_limit_reached_detail(ent, trace_id))

    if result["outcome"] == "existing":
        # Another request already holds this (user_id, idempotency_key).
        # This request's own upload is now orphaned -- clean it up
        # (best-effort; never the existing job's s3_input_key). Quota was
        # never touched for this request (checked before reservation).
        delete_object(s3_key)
        existing = result["job"]

        if existing.get("idempotency_fingerprint") == idempotency_fingerprint:
            logger.info(
                "[api] idempotency_replay job_id=%s user_id=%s trace_id=%s",
                existing.get("id"), user_id, trace_id,
            )
            response.headers["Idempotency-Replayed"] = "true"
            return JobResponse(
                job_id=str(existing.get("id")),
                status=str(existing.get("status")),
                mode=str(existing.get("mode")),
                created_at=str(existing.get("created_at")) if existing.get("created_at") else None,
            )

        logger.warning(
            "[api] idempotency_key_conflict user_id=%s key=%s trace_id=%s",
            user_id, idempotency_key, trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "idempotency_key_conflict",
                    "message": "This Idempotency-Key was already used with a different request",
                    "trace_id": trace_id},
        )

    # ── New job created — same post-processing as before ────────────────────
    created_row = result["job"]
    job_id = str(created_row["id"])

    # Persist migration-006 columns (priority + plan) — best-effort
    try:
        from scripts.prod_job_store import _get_conn  # type: ignore
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE generation_jobs SET priority=%s, plan=%s WHERE id=%s",
                        (priority, plan, job_id),
                    )
        finally:
            conn.close()
    except Exception as _exc:
        logger.warning("[api] priority_col_update_skipped job_id=%s exc=%s", job_id, _exc)

    # Register input file
    try:
        add_job_file(
            job_id=job_id, user_id=user_id,
            file_type="input", s3_key=s3_key,
            filename=safe_name, content_type=file.content_type or "video/mp4",
            size_bytes=len(content),
        )
    except Exception as exc:
        logger.warning("[api] add_input_file_failed job_id=%s: %s", job_id, exc)

    audit(EVT_JOB_CREATED, user_id=user_id, job_id=job_id, trace_id=trace_id,
          details={"mode": mode, "size_bytes": len(content), "plan": plan, "priority": priority,
                   "entitlement": ent.kind, "plan_id": ent.plan_id},
          ip_address=request.client.host if request.client else None)

    logger.info(
        "[api] job_created job_id=%s user_id=%s mode=%s plan=%s priority=%d size=%d",
        job_id, user_id, mode, plan, priority, len(content),
    )

    job = get_job(job_id)
    return JobResponse(
        job_id=job_id,
        status=JOB_STATUS_QUEUED,
        mode=mode,
        created_at=str(job.get("created_at")) if job else None,
    )


@app.post("/api/trailer/jobs", response_model=JobResponse)
async def create_trailer_job(
    request: Request,
    file: UploadFile = File(...),
    params: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
    _origin: None = Depends(verify_browser_origin),
    _rl: None = Depends(_upload_limiter),
):
    """Alias: mode=trailer_film_breaker."""
    _reject_if_mode_disabled("trailer_film_breaker", request, str(user["id"]), new_trace_id())
    return await create_generation_job(
        request=request, mode="trailer_film_breaker",
        file=file, params=params, user=user, _origin=None, _rl=None,
    )


# ── Public: Job status / result ————————————————————————————————————————————————

@app.get("/api/generation/jobs/{job_id}")
async def get_generation_job(
    request: Request,
    job_id: str,
    user: dict = Depends(get_current_user),
    _rl: None = Depends(_api_limiter),
):
    user_id = str(user["id"])
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "not_found", "trace_id": new_trace_id()})
    assert_job_owner(job, user_id)

    files = list_job_files(job_id)
    return {**job, "files": files}


@app.get("/api/generation/jobs/{job_id}/result-url")
async def get_result_url(
    request: Request,
    job_id: str,
    expires: int = Query(default=3600, ge=60, le=86400),
    user: dict = Depends(get_current_user),
):
    user_id = str(user["id"])
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "not_found", "trace_id": new_trace_id()})
    assert_job_owner(job, user_id)

    if job.get("status") != JOB_STATUS_COMPLETED:
        raise HTTPException(
            status_code=400,
            detail={"error": "job_not_done", "status": job.get("status"), "trace_id": new_trace_id()},
        )

    s3_key = job.get("s3_output_key")
    if not s3_key:
        raise HTTPException(status_code=404, detail={"error": "no_output", "trace_id": new_trace_id()})

    try:
        url = generate_presigned_get_url(s3_key, expires_in=expires)
    except Exception as exc:
        logger.error("[api] presign_failed job_id=%s: %s", job_id, exc)
        raise safe_error("storage_error", 500)

    # Every rendered clip of the job (a Vast job uploads N clips; s3_output_key is
    # only the first). The site's result grid already renders result.clips.
    clips = []
    try:
        outs = [f for f in list_job_files(job_id)
                if f.get("file_type") == "output"
                and str(f.get("s3_key") or "").lower().endswith((".mp4", ".mov", ".webm", ".png", ".jpg"))]
        _all = list_job_files(job_id)
        _aux = {str(a["s3_key"]): a for a in _all if a.get("file_type") in ("preview", "thumbnail")}
        for n, f in enumerate(sorted(outs, key=lambda f: str(f.get("s3_key"))), 1):
            _ext = str(f["s3_key"]).lower().rsplit(".", 1)[-1]
            _nm = ("sonya_gen_%d.%s" % (n, _ext)) if _ext in ("png", "jpg") else ("sonya_clip_%d.mp4" % n)
            _c = {"url": generate_presigned_get_url(f["s3_key"], expires_in=expires),
                  "download_url": generate_presigned_get_url(f["s3_key"], expires_in=expires,
                                                             download_name=_nm)}
            _stem = str(f["s3_key"]).rsplit(".", 1)[0]
            if _stem + "_preview.mp4" in _aux:      # light 720p copy for watching; download stays full quality
                _c["preview_url"] = generate_presigned_get_url(_stem + "_preview.mp4", expires_in=expires)
            if _stem + "_poster.jpg" in _aux:
                _c["poster_url"] = generate_presigned_get_url(_stem + "_poster.jpg", expires_in=expires)
            clips.append(_c)
    except Exception as exc:
        logger.warning("[api] clip_list_failed job_id=%s: %s", job_id, exc)
        clips = []
    if not clips:
        clips = [{"url": url, "download_url": generate_presigned_get_url(
            s3_key, expires_in=expires, download_name="sonya_clip_1.mp4")}]

    return {"url": url, "expires_in": expires, "clips": clips}


@app.get("/api/generation/jobs")
async def list_jobs(
    request: Request,
    limit: int  = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    user: dict = Depends(get_current_user),
    _rl: None = Depends(_api_limiter),
):
    user_id = str(user["id"])
    jobs = list_user_jobs(user_id, limit=limit, offset=offset, status=status_filter)
    return {"jobs": jobs, "count": len(jobs)}


# ── Public: Job creation from a URL ——————————————————————————————————————————————
# Downloads the video server-side (scripts/url_ingest.py: yt-dlp for
# YouTube/VK/Twitch, a size-capped streamed GET for a direct video URL —
# both SSRF-guarded), then hands the result to the exact same
# validate-bytes -> S3 upload -> create_job_idempotent -> add_job_file
# sequence create_generation_job() uses for a browser file upload. No
# separate processing pipeline: once a job row exists here, it is
# indistinguishable from one created by a file upload, and the dispatcher /
# GPU worker never know the input came from a URL.

@app.post("/api/generation/jobs/from-url", status_code=status.HTTP_202_ACCEPTED)
async def create_generation_job_from_url(
    request: Request,
    background_tasks: BackgroundTasks,
    body: UrlIngestRequest,
    idempotency_key_header: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: dict = Depends(get_current_user),
    _origin: None = Depends(verify_browser_origin),
    _rl: None = Depends(_upload_limiter),
):
    user_id  = str(user["id"])
    trace_id = new_trace_id()
    priority, plan = _resolve_priority(user)

    mode = body.mode
    if mode not in ALLOWED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_mode", "allowed": sorted(ALLOWED_MODES), "trace_id": trace_id},
        )

    # Feature flag: see create_generation_job(). Before the quota check and
    # before any network fetch of the user's URL.
    _reject_if_mode_disabled(mode, request, user_id, trace_id)

    idempotency_key = _validate_idempotency_key(idempotency_key_header, trace_id)
    parsed_params = body.params if isinstance(body.params, dict) else {}

    # Quota check happens before any network fetch -- same ordering as the
    # file-upload endpoint.
    check_user_quota(user_id)

    # Fast-fail only -- see the equivalent check in create_generation_job()
    # for why this isn't the enforcement point (create_job_with_quota() in
    # _run_url_ingest below is).
    ent = _resolve_entitlement(user, mode, trace_id)
    priority, plan = _priority_for(ent, priority, plan)

    url = body.url.strip()
    platform = url_ingest.detect_platform(url)
    if platform == "unsupported":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "unsupported_url",
                    "message": url_ingest.UnsupportedUrlError().user_message,
                    "trace_id": trace_id},
        )
    try:
        url_ingest.assert_safe_url(url)
    except url_ingest.UnsafeUrlError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "unsafe_url", "message": url_ingest.UnsafeUrlError().user_message,
                    "trace_id": trace_id},
        )

    ingest_id = str(uuid.uuid4())
    _ingest_set(
        ingest_id,
        status="checking", percent=0.0, message="Проверяем ссылку",
        user_id=user_id, job_id=None, error=None,
    )

    background_tasks.add_task(
        _run_url_ingest,
        ingest_id=ingest_id, url=url, platform=platform, mode=mode,
        parsed_params=parsed_params, user_id=user_id, trace_id=trace_id,
        idempotency_key=idempotency_key, priority=priority, plan=plan,
        ent=ent,
        client_ip=request.client.host if request.client else None,
    )

    logger.info("[api] url_ingest_started ingest_id=%s user_id=%s platform=%s trace_id=%s",
                ingest_id, user_id, platform, trace_id)

    return {"ingest_id": ingest_id, "status": "checking", "platform": platform}


@app.get("/api/generation/jobs/from-url/{ingest_id}")
async def get_url_ingest_status(
    ingest_id: str,
    user: dict = Depends(get_current_user),
    _rl: None = Depends(_api_limiter),
):
    entry = _ingest_get(ingest_id)
    # Same 404 whether the id is unknown or belongs to another user -- never
    # confirm/deny existence of another user's ingestion.
    if entry is None or entry.get("user_id") != str(user["id"]):
        raise HTTPException(status_code=404, detail={"error": "not_found", "trace_id": new_trace_id()})

    return {
        "ingest_id": ingest_id,
        "status": entry.get("status"),
        "percent": entry.get("percent"),
        "message": entry.get("message"),
        "job_id": entry.get("job_id"),
        "error": entry.get("error"),
    }


def _run_url_ingest(
    *, ingest_id: str, url: str, platform: str, mode: str,
    parsed_params: Dict[str, Any], user_id: str, trace_id: str,
    idempotency_key: Optional[str], priority: int, plan: str,
    ent: "entitlements.Entitlement",
    client_ip: Optional[str],
) -> None:
    """
    Runs in Starlette's threadpool (via BackgroundTasks), off the event
    loop, so a slow download never blocks other requests. Every exit path
    updates the ingest store; nothing here raises past this function.
    """
    local_path: Optional[str] = None
    try:
        _ingest_set(ingest_id, status="checking", message="Проверяем ссылку", percent=5.0)
        # YouTube is unreachable from this (Moscow) VPS: hand the link to the GPU worker (params.source_fetch), which
        # downloads it, checks the plan length and uploads the file to s3_input_key. Off: SONYA_SOURCE_FETCH=0.
        _sf_modes = {m.strip() for m in os.environ.get("SONYA_SOURCE_FETCH_MODES", "virality,beep,educational,streamer").split(",")}
        use_source_fetch = (platform == "youtube" and os.environ.get("SONYA_SOURCE_FETCH", "1") != "0" and mode in _sf_modes)
        if not use_source_fetch:
            url_ingest.probe(url, platform, mode=mode, max_duration_sec=ent.max_source_sec)

        _ingest_set(ingest_id, status="downloading", message="Получаем видео", percent=15.0)

        def _progress(pct: float) -> None:
            # Map the downloader's own 0..100 onto the 15..70 slice of the
            # overall ingest progress bar.
            _ingest_set(ingest_id, status="downloading", message="Получаем видео",
                        percent=15.0 + max(0.0, min(100.0, pct)) * 0.55)

        if use_source_fetch:
            content, safe_name = url.encode("utf-8"), f"{platform}_video.mp4"
            parsed_params = dict(parsed_params, source_fetch={
                "url": url, "platform": platform,
                "max_source_sec": int(ent.max_source_sec) if ent.max_source_sec else None})
        else:
            local_path, _ext = url_ingest.download_video(url, platform, progress_cb=_progress, mode=mode)

            # Authoritative paid-plan length check on the downloaded file
            # (platform metadata above is only an early hint). No debit yet.
            try:
                entitlements.measure_and_enforce(ent, local_path)
            except entitlements.EntitlementDenied as denied:
                _ingest_set(ingest_id, status="failed", error=denied.code, message=denied.message)
                return

            _ingest_set(ingest_id, status="uploading", message="Загружаем видео", percent=75.0)

            content, safe_name = url_ingest.validate_downloaded_file(
                local_path, hint_name=f"{platform}_video{Path(local_path).suffix or '.mp4'}", mode=mode
            )

        idempotency_fingerprint = (
            _compute_idempotency_fingerprint(mode, content, parsed_params)
            if idempotency_key is not None else None
        )

        job_id = str(uuid.uuid4())
        s3_key = build_input_key(user_id=user_id, job_id=job_id, mode=mode,
                                  ext=Path(safe_name).suffix or ".mp4")

        try:
            if not use_source_fetch:
                upload_bytes(content, s3_key, content_type="video/mp4")
        except Exception as exc:
            logger.error("[api] url_ingest_s3_upload_failed ingest_id=%s trace_id=%s: %s",
                         ingest_id, trace_id, exc)
            _ingest_set(ingest_id, status="failed", error="storage_error",
                        message="Ошибка загрузки в хранилище. Попробуйте позже.")
            return

        try:
            result = create_job_with_quota(
                job_id=job_id, user_id=user_id, mode=mode, params=parsed_params,
                s3_input_key=s3_key, idempotency_key=idempotency_key,
                idempotency_fingerprint=idempotency_fingerprint, queue_priority=priority,
                bypass_quota=ent.kind == "legacy_pro", subscription_id=ent.subscription_id,
            )
        except Exception as exc:
            logger.error("[api] url_ingest_create_job_failed ingest_id=%s trace_id=%s: %s",
                         ingest_id, trace_id, exc)
            delete_object(s3_key)
            _ingest_set(ingest_id, status="failed", error="db_error",
                        message="Ошибка создания задачи. Попробуйте позже.")
            return

        if result["outcome"] == "quota_exceeded":
            # Lost the race to the atomic quota reservation between the
            # fast-fail check in create_generation_job_from_url() and here.
            delete_object(s3_key)
            _ingest_set(ingest_id, status="failed", error="FREE_PLAN_USED",
                        message="Бесплатная генерация уже использована. Выберите тариф SONYA.")
            return

        if result["outcome"] == "plan_limit_reached":
            delete_object(s3_key)
            _ingest_set(ingest_id, status="failed", error=entitlements.PLAN_LIMIT_REACHED,
                        message="Лимит тарифа исчерпан или период подписки закончился.")
            return

        if result["outcome"] == "existing":
            # Same idempotency-conflict handling as the file-upload path.
            # Quota was never touched for this request.
            delete_object(s3_key)
            existing = result["job"]
            if existing.get("idempotency_fingerprint") == idempotency_fingerprint:
                _ingest_set(ingest_id, status="queued", message="Видео в очереди",
                            percent=100.0, job_id=str(existing.get("id")))
                return
            _ingest_set(ingest_id, status="failed", error="idempotency_key_conflict",
                        message="Эта задача уже была отправлена.")
            return

        job_id = str(result["job"]["id"])

        try:
            from scripts.prod_job_store import _get_conn  # type: ignore
            conn = _get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE generation_jobs SET priority=%s, plan=%s WHERE id=%s",
                            (priority, plan, job_id),
                        )
            finally:
                conn.close()
        except Exception as _exc:
            logger.warning("[api] priority_col_update_skipped job_id=%s exc=%s", job_id, _exc)

        try:
            if use_source_fetch:
                raise RuntimeError("source_fetch: the worker uploads the input")
            add_job_file(
                job_id=job_id, user_id=user_id,
                file_type="input", s3_key=s3_key,
                filename=safe_name, content_type="video/mp4",
                size_bytes=len(content),
            )
        except Exception as exc:
            logger.warning("[api] url_ingest_add_input_file_failed job_id=%s: %s", job_id, exc)

        audit(EVT_JOB_CREATED, user_id=user_id, job_id=job_id, trace_id=trace_id,
              details={"mode": mode, "size_bytes": len(content), "plan": plan,
                       "priority": priority, "source": "url", "platform": platform},
              ip_address=client_ip)

        logger.info(
            "[api] url_ingest_job_created ingest_id=%s job_id=%s user_id=%s mode=%s platform=%s size=%d",
            ingest_id, job_id, user_id, mode, platform, len(content),
        )

        _ingest_set(ingest_id, status="queued", message="Видео в очереди", percent=100.0, job_id=job_id)

    except url_ingest.DownloadLimitExceeded as exc:
        _ingest_set(ingest_id, status="failed", error="limit_exceeded", message=exc.user_message)
    except url_ingest.UnsafeUrlError as exc:
        audit(EVT_UPLOAD_REJECTED, user_id=user_id, trace_id=trace_id,
              details={"mode": mode, "reason": "unsafe_url", "platform": platform}, ip_address=client_ip)
        _ingest_set(ingest_id, status="failed", error="unsafe_url", message=exc.user_message)
    except url_ingest.UnsupportedUrlError as exc:
        _ingest_set(ingest_id, status="failed", error="unsupported_url", message=exc.user_message)
    except url_ingest.DownloadFailed as exc:
        _ingest_set(ingest_id, status="failed", error="download_failed", message=exc.user_message)
    except HTTPException as exc:
        # Raised by validate_downloaded_file() (upload_security checks) --
        # same rejection reasons a bad file upload would get.
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        audit(EVT_UPLOAD_REJECTED, user_id=user_id, trace_id=trace_id,
              details={"mode": mode, "reason": detail, "platform": platform, "source": "url"},
              ip_address=client_ip)
        _ingest_set(ingest_id, status="failed", error=detail.get("error", "invalid_video"),
                    message="Скачанное видео не прошло проверку.")
    except Exception as exc:
        logger.exception("[api] url_ingest_unexpected_error ingest_id=%s trace_id=%s: %s",
                          ingest_id, trace_id, exc)
        _ingest_set(ingest_id, status="failed", error="internal_error",
                    message="Внутренняя ошибка. Попробуйте позже.")
    finally:
        if local_path:
            url_ingest.cleanup(local_path)


# ── Worker endpoints (require WORKER_SECRET) ————————————————————————————————————

@app.post("/api/worker/claim")
async def worker_claim(
    body: WorkerClaimRequest,
    _auth: None = Depends(verify_worker_secret),
):
    """Worker claims the next available pending job, or a specific job_id."""
    trace_id = new_trace_id()

    if body.job_id:
        job = claim_specific_job(body.job_id, worker_id=body.worker_id)
    else:
        job = claim_next_pending_job(worker_id=body.worker_id, modes=body.modes)

    if not job:
        return {"job": None}

    audit(EVT_JOB_CLAIMED, job_id=job.get("id"), trace_id=trace_id,
          details={"worker_id": body.worker_id, "mode": job.get("mode")})

    return {"job": job}


@app.post("/api/worker/jobs/{job_id}/status")
async def worker_update_status(
    job_id: str,
    body: JobStatusUpdate,
    _auth: None = Depends(verify_worker_secret),
):
    _VALID_STATUSES = {
        "queued", "claimed", "gpu_requested", "gpu_booting", "worker_started",
        "preflight_running", "downloading", "model_downloading", "mode_running",
        "analyzing", "yolo", "scripting", "tts", "subtitles", "assembling",
        "uploading_result", "completed", "failed", "cancelled",
    }
    if body.status not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail={"error": "invalid_status"})
    update_job_status(job_id, body.status)
    return {"ok": True, "job_id": job_id, "status": body.status}


@app.post("/api/worker/jobs/{job_id}/heartbeat")
async def worker_heartbeat(
    job_id: str,
    _auth: None = Depends(verify_worker_secret),
):
    """
    Liveness only — refreshes heartbeat_at (migration 011), no status
    change. Called periodically by a worker during a long-running mode
    step so requeue_stale_jobs() never mistakes a slow-but-alive job for a
    stuck one. See scripts/prod_job_store.py::touch_job_heartbeat().
    """
    touch_job_heartbeat(job_id)
    return {"ok": True, "job_id": job_id}


def _maybe_reconcile_streamer_batch(job: Optional[Dict[str, Any]]) -> None:
    """
    Called after a job reaches a terminal state via worker/complete or
    worker/fail below -- the primary trigger for batch-status
    reconciliation (GET /api/streamer/batches/{id} in
    scripts/streamer_routes.py is the defensive backstop for the rare
    case this is ever missed). No-op for any job that isn't part of a
    streamer batch (params carries no streamer_batch_id -- true for every
    non-streamer job, and for a plain streamer job with no batch behind
    it).

    A compose-phase job's own completion/failure is reconciled against
    every OTHER compose job in the same batch (reconcile_streamer_batch()
    reads them all) -- see that function's docstring for the ready /
    partially_failed / failed decision. An analyze-phase job failing
    permanently (status actually "failed", not requeued for retry) fails
    the whole batch directly instead -- there's nothing to "reconcile"
    against other jobs at that point, since no compose jobs can exist yet
    (they're only created after awaiting_selection, which an analyze
    failure never reaches).
    """
    if not job:
        return
    params = job.get("params") or {}
    batch_id = params.get("streamer_batch_id")
    if not batch_id:
        return

    phase = params.get("streamer_phase")
    try:
        if phase == "compose":
            reconcile_streamer_batch(batch_id)
        elif phase == "analyze" and job.get("status") == JOB_STATUS_FAILED:
            set_streamer_batch_status(
                batch_id, "failed",
                error=job.get("last_error") or job.get("error") or "analyze_failed",
            )
    except StreamerBatchTransitionError:
        pass  # batch already terminal (e.g. cancelled) -- nothing to do
    except Exception as exc:
        logger.warning("[api] streamer_batch_reconcile_failed batch_id=%s job_id=%s: %s",
                        batch_id, job.get("id"), exc)


def _maybe_notify_streamer_batch(job: Optional[Dict[str, Any]]) -> None:
    """
    Runs as a FastAPI background task, scheduled AFTER _maybe_reconcile_
    streamer_batch(job) has already run synchronously above -- so by the
    time this actually executes, the batch's terminal status (if this
    job's completion/failure caused one) is already committed. This is
    exactly why it's backgrounded: Telegram network latency (or an outage)
    must never delay this endpoint's own response to the GPU worker, and
    notify_streamer_batch_completion() itself never raises (see its own
    docstring) so it can never turn this into a failed background task
    either. No-op for any job that isn't part of a streamer batch, same
    guard as _maybe_reconcile_streamer_batch above.
    """
    if not job:
        return
    batch_id = (job.get("params") or {}).get("streamer_batch_id")
    if not batch_id:
        return
    notify_streamer_batch_completion(batch_id)


def _cleanup_ephemeral_instance(job: Optional[Dict[str, Any]]) -> None:
    """
    Best-effort destroy of the vast.ai instance backing a job that just
    reached a terminal state. Runs as a FastAPI background task, strictly
    after the job's terminal status is already committed to Postgres, so a
    destroy failure here can never turn a completed job into a failed one.

    No-op for anything not explicitly stamped gpu_managed_by="vast_ephemeral"
    -- see prod_job_store.get_ephemeral_contract_id() for the ownership
    rule. That covers the current production config entirely: the live GPU
    is manually managed and never carries that stamp, so this is always a
    no-op today. Never raises.
    """
    if not job:
        return
    try:
        orchestrator.cleanup_instance_for_terminal_job(job)
    except Exception as exc:
        logger.warning(
            "[api] ephemeral_instance_cleanup_failed job_id=%s exc=%s",
            job.get("id"), exc,
        )


@app.post("/api/worker/jobs/{job_id}/complete")
async def worker_complete_job(
    job_id: str,
    body: JobCompleteRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(verify_worker_secret),
):
    trace_id = new_trace_id()
    complete_job(
        job_id=job_id,
        s3_output_key=body.s3_output_key,
        clip_count=body.clip_count,
        processing_ms=body.processing_ms,
        enrichment_keys=body.enrichment_keys,
    )
    job = get_job(job_id)
    audit(EVT_JOB_COMPLETED, user_id=job.get("user_id") if job else None,
          job_id=job_id, trace_id=trace_id,
          details={"clip_count": body.clip_count, "processing_ms": body.processing_ms})
    logger.info("[api] job_completed job_id=%s clips=%s ms=%s", job_id, body.clip_count, body.processing_ms)
    _maybe_reconcile_streamer_batch(job)
    background_tasks.add_task(_maybe_notify_streamer_batch, job)
    background_tasks.add_task(_cleanup_ephemeral_instance, job)
    return {"ok": True, "job_id": job_id}


@app.post("/api/worker/jobs/{job_id}/fail")
async def worker_fail_job(
    job_id: str,
    body: JobFailRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(verify_worker_secret),
):
    trace_id = new_trace_id()
    fail_job(
        job_id=job_id,
        error_code=body.error_code,
        error_message=body.error_message,
        retry=body.retry,
    )
    job = get_job(job_id)
    audit(EVT_JOB_FAILED, user_id=job.get("user_id") if job else None,
          job_id=job_id, trace_id=trace_id,
          details={"error_code": body.error_code, "retry": body.retry})
    logger.warning("[api] job_failed job_id=%s code=%s retry=%s", job_id, body.error_code, body.retry)
    _maybe_reconcile_streamer_batch(job)
    background_tasks.add_task(_maybe_notify_streamer_batch, job)
    background_tasks.add_task(_cleanup_ephemeral_instance, job)
    return {"ok": True, "job_id": job_id}


@app.post("/api/worker/jobs/{job_id}/files")
async def worker_add_file(
    job_id: str,
    body: JobFileRequest,
    _auth: None = Depends(verify_worker_secret),
):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "not_found"})

    add_job_file(
        job_id=job_id,
        user_id=job["user_id"],
        file_type=body.file_type,
        s3_key=body.s3_key,
        filename=body.filename,
        content_type=body.content_type,
        size_bytes=body.size_bytes,
        duration_sec=body.duration_sec,
    )
    return {"ok": True, "job_id": job_id, "file_type": body.file_type}
