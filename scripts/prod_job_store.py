"""
prod_job_store.py
=================
PostgreSQL job store for SONYA generation pipeline.

Tables (from migrations):
  generation_jobs  — job lifecycle
  generation_files — per-job S3 file registry

Status lifecycle:
  queued → claimed → downloading → model_downloading → mode_running
  → analyzing → yolo → scripting → tts → subtitles → assembling
  → uploading_result → completed
  (any step) → failed

Functions:
  create_job             create a new queued job
  create_job_idempotent  atomic create-or-detect-conflict insert
                         (POST /api/generation/jobs Idempotency-Key,
                         migration 009)
  get_job_by_idempotency_key  fetch the job owning a (user_id, key) pair
  create_job_with_quota  idempotency-key check + free-plan quota reservation
                         + job insert, all in one transaction (see its own
                         docstring -- this is what create_generation_job()
                         and _run_url_ingest() call; create_job_idempotent()
                         above is kept for other callers/tooling but is no
                         longer on the free-plan-gated path)
  get_job                fetch single job by id
  list_user_jobs         paginated list for a user
  update_job_status      granular status update (any status constant)
  complete_job           mark completed, set output key + metadata
  fail_job               mark failed, optionally requeue as queued
  claim_next_pending_job FOR UPDATE SKIP LOCKED — poll mode
  claim_specific_job     claim a known job_id — --once mode
  requeue_stale_jobs     reset stuck jobs → queued (liveness-aware — see
                         touch_job_heartbeat below)
  touch_job_heartbeat    refresh heartbeat_at only (migration 011) — proves
                         a worker is still alive on a long-running job
                         without touching status/updated_at; called
                         periodically by gpu_worker.py during a long mode
                         run (e.g. streamer's multi-hour enrichment)
  add_job_file           register an S3 file with a job
  list_job_files         list all files for a job

Streamer batches (migration 012) — batch foundation only, no HTTP
endpoints and no worker phase dispatch wired up yet:
  create_streamer_batch          new batch, analysis_job_id nullable
  get_streamer_batch             user_id required — ownership enforced in the query
  list_user_streamer_batches     paginated list for a user
  set_streamer_batch_status      terminal-status transitions are locked (see its docstring)
  attach_analysis_job_to_batch   set analysis_job_id once the generation job exists
  replace_streamer_segments      atomic delete-all + insert for one batch
  list_streamer_segments         ordered by ordinal
  set_streamer_segment_selection toggle one segment's `selected`
  attach_streamer_clip_job       link a real generation_jobs row to a segment (idempotent)
  list_streamer_clip_jobs        all clip-job links for a batch
  segments_from_analysis         maps analyze()'s return shape -> replace_streamer_segments() input

GPU dispatcher / Vast startup SLA (migration 006 + 007):
  get_stale_gpu_requested_jobs  find jobs stuck past VAST_STARTUP_TIMEOUT_SEC
  mark_gpu_startup_timeout      requeue (another offer) or fail a timed-out job

GPU instance ownership / cleanup (future automatic lifecycle — see
gpu_orchestrator.cleanup_instance_for_terminal_job and
gpu_dispatcher.reconcile_gpu_instance_cleanup; disabled today because the
current production GPU is manually managed, not dispatcher-provisioned):
  get_ephemeral_contract_id           contract_id to destroy for a terminal
                                       job, or None if not a dispatcher-owned
                                       instance
  mark_gpu_instance_cleanup_destroyed record a confirmed destroy (incl. 404)
  mark_gpu_instance_cleanup_error     record a failed destroy attempt
  get_terminal_jobs_pending_gpu_cleanup
                                       dispatcher reconciliation candidates —
                                       terminal + vast_ephemeral + not yet
                                       confirmed destroyed
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Status constants ——————————————————————————————————————————————————————————

JOB_STATUS_QUEUED            = "queued"
JOB_STATUS_CLAIMED           = "claimed"
JOB_STATUS_GPU_REQUESTED     = "gpu_requested"
JOB_STATUS_GPU_BOOTING       = "gpu_booting"
JOB_STATUS_WORKER_STARTED    = "worker_started"
JOB_STATUS_PREFLIGHT_RUNNING = "preflight_running"
JOB_STATUS_DOWNLOADING       = "downloading"
JOB_STATUS_MODEL_DOWNLOADING = "model_downloading"
JOB_STATUS_MODE_RUNNING      = "mode_running"
JOB_STATUS_ANALYZING         = "analyzing"
JOB_STATUS_YOLO              = "yolo"
JOB_STATUS_SCRIPTING         = "scripting"
JOB_STATUS_TTS               = "tts"
JOB_STATUS_SUBTITLES         = "subtitles"
JOB_STATUS_ASSEMBLING        = "assembling"
JOB_STATUS_UPLOADING_RESULT  = "uploading_result"
JOB_STATUS_COMPLETED         = "completed"
JOB_STATUS_FAILED            = "failed"
JOB_STATUS_CANCELLED         = "cancelled"

_ACTIVE_STATUSES = (
    JOB_STATUS_CLAIMED, JOB_STATUS_GPU_REQUESTED, JOB_STATUS_GPU_BOOTING,
    JOB_STATUS_WORKER_STARTED, JOB_STATUS_PREFLIGHT_RUNNING, JOB_STATUS_DOWNLOADING,
    JOB_STATUS_MODEL_DOWNLOADING, JOB_STATUS_MODE_RUNNING, JOB_STATUS_ANALYZING,
    JOB_STATUS_YOLO, JOB_STATUS_SCRIPTING, JOB_STATUS_TTS, JOB_STATUS_SUBTITLES,
    JOB_STATUS_ASSEMBLING, JOB_STATUS_UPLOADING_RESULT,
)

_TERMINAL_STATUSES = (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED)

# GPU instance ownership — stamped into orchestrator_payload.gpu_managed_by
# by gpu_orchestrator._trigger_vast() ONLY when the automatic dispatcher
# provisions a per-job ephemeral vast.ai instance. The current production
# GPU is a manually-provisioned, persistently-running worker
# (WORKER_LOOP=true) that claims jobs itself via claim_next_pending_job() —
# it never goes through mark_gpu_requested(), so its jobs' orchestrator_payload
# is always empty and this stamp never appears for them. See
# get_ephemeral_contract_id() below, which is the single source of truth for
# "is this job's GPU safe to destroy automatically".
GPU_MANAGED_BY_VAST_EPHEMERAL = "vast_ephemeral"

# Cleanup confirmation, stored in the SAME orchestrator_payload JSONB column
# (no schema migration needed — JSONB has no fixed key set). Written by
# gpu_orchestrator.cleanup_instance_for_terminal_job() after every destroy
# attempt, read by get_terminal_jobs_pending_gpu_cleanup() so the dispatcher's
# reconciliation pass stops re-attempting a job once destroy is confirmed.
# Absence of gpu_cleanup_status (or any value other than "destroyed") is
# always treated as "not yet confirmed — retry-eligible", which covers both
# "never attempted" and "attempted and failed" with the same retry behavior.
GPU_CLEANUP_STATUS_DESTROYED = "destroyed"
GPU_CLEANUP_STATUS_ERROR = "error"

# ── DB helpers —————————————————————————————————————————————————————————————————

_DB_AVAILABLE = False
try:
    import psycopg2
    import psycopg2.extras
    _DB_AVAILABLE = True
except ImportError:
    pass


def _get_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    if not _DB_AVAILABLE:
        raise RuntimeError("psycopg2 not installed — run: pip install psycopg2-binary")
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row(conn, sql: str, params: tuple) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def _rows(conn, sql: str, params: tuple) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ── Job CRUD ———————————————————————————————————————————————————————————————————

def create_job(
    job_id: str,
    user_id: str,
    mode: str,
    params: Dict[str, Any],
    s3_input_key: str,
    queue_priority: int = 0,
) -> str:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO generation_jobs
                        (id, user_id, mode, params, s3_input_key, status,
                         queue_priority, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (job_id, user_id, mode, json.dumps(params), s3_input_key,
                     JOB_STATUS_QUEUED, queue_priority, _now(), _now()),
                )
    finally:
        conn.close()
    return job_id


def create_job_idempotent(
    job_id: str,
    user_id: str,
    mode: str,
    params: Dict[str, Any],
    s3_input_key: str,
    idempotency_key: Optional[str],
    idempotency_fingerprint: Optional[str],
    queue_priority: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Atomic create-or-detect-conflict insert for POST /api/generation/jobs.

    Returns the newly created row when the INSERT wins (idempotency_key is
    None -- no header supplied, legacy behavior, always wins since NULL
    never conflicts with anything under standard SQL UNIQUE semantics --
    or idempotency_key is set and no row with this (user_id,
    idempotency_key) existed yet).

    Returns None when a row with the same (user_id, idempotency_key)
    already exists -- ON CONFLICT DO NOTHING matched zero rows. The caller
    must then fetch the existing row via get_job_by_idempotency_key() and
    compare idempotency_fingerprint to decide replay (200/202, same job)
    vs conflict (409, different payload under the same key).

    Deliberately does NOT do a SELECT before this INSERT -- conflict
    detection is entirely the database's, via the unique index
    (ux_jobs_user_idempotency_key, migration 009). A SELECT first would
    not protect against two concurrent requests racing past it.

    Deliberately does NOT use ON CONFLICT ... DO UPDATE -- generation_jobs
    has a BEFORE UPDATE trigger (trg_jobs_updated_at) that would bump
    updated_at on the existing row for every duplicate/replayed request,
    even though nothing about that row actually changed.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO generation_jobs
                        (id, user_id, mode, params, s3_input_key, status,
                         queue_priority, created_at, updated_at,
                         idempotency_key, idempotency_fingerprint)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, idempotency_key) DO NOTHING
                    RETURNING *
                    """,
                    (job_id, user_id, mode, json.dumps(params), s3_input_key,
                     JOB_STATUS_QUEUED, queue_priority, _now(), _now(),
                     idempotency_key, idempotency_fingerprint),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    finally:
        conn.close()


def create_job_with_quota(
    *,
    job_id: str,
    user_id: str,
    mode: str,
    params: Dict[str, Any],
    s3_input_key: str,
    idempotency_key: Optional[str],
    idempotency_fingerprint: Optional[str],
    queue_priority: int,
    bypass_quota: bool,
    subscription_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Atomically, in ONE transaction: check for an existing job under
    (user_id, idempotency_key), reserve one unit of quota, and insert the
    new job -- so a job insert failure can never leave a "spent" quota unit
    behind, and a replayed request never spends quota a second time.

    Which quota (see scripts/entitlements.py):
      subscription_id given -> one operation of that user_subscriptions
                               period (must belong to user_id, still be
                               running, and have ops left); the job records
                               it in charged_subscription_id
      else bypass_quota     -> nothing (legacy Pro)
      else                  -> one free-plan unit

    Returns one of:
      {"outcome": "existing",           "job": {...}}  -- idempotency_key already
                                                          had a row; quota untouched
      {"outcome": "quota_exceeded",     "job": None}   -- new request, but
                                                          free_video_used >= free_video_limit
      {"outcome": "plan_limit_reached", "job": None}   -- new request, but the
                                                          subscription period is
                                                          exhausted or already over
      {"outcome": "created",            "job": {...}}  -- new job row inserted

    Concurrency: the first statement locks the user's own row
    (SELECT ... FOR UPDATE), which serializes every job-creation attempt
    for that SAME user (not globally) for the short duration of this
    transaction -- long enough to make both the idempotency-key lookup and
    the quota reservation race-free together, without depending on
    create_job_idempotent's separate-connection unique-index-race pattern
    (which is safe for uniqueness alone, but not for "check quota, then
    decide whether to spend it" without also serializing per user). The
    actual expensive work (download/S3 upload) happens entirely before this
    function is called -- this transaction only ever does two small
    SELECTs, at most one UPDATE, and one INSERT.

    If the job insert fails for any reason (unique-constraint violation
    from a bug elsewhere, connection loss, etc.), the whole transaction
    rolls back -- including the quota UPDATE above it in the same
    transaction -- so free_video_used is never incremented for a job that
    was not actually created.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (user_id,))
                if cur.fetchone() is None:
                    raise RuntimeError(f"create_job_with_quota: user not found: {user_id}")

                if idempotency_key is not None:
                    cur.execute(
                        "SELECT * FROM generation_jobs WHERE user_id = %s AND idempotency_key = %s",
                        (user_id, idempotency_key),
                    )
                    existing = cur.fetchone()
                    if existing is not None:
                        return {"outcome": "existing", "job": dict(existing)}

                if subscription_id is not None:
                    cur.execute(
                        """
                        UPDATE user_subscriptions
                        SET ops_used = ops_used + 1
                        WHERE id = %s AND user_id = %s
                          AND ops_used < ops_limit
                          AND period_start <= NOW() AND period_end > NOW()
                        RETURNING id
                        """,
                        (subscription_id, user_id),
                    )
                    if cur.fetchone() is None:
                        return {"outcome": "plan_limit_reached", "job": None}
                elif not bypass_quota:
                    cur.execute(
                        """
                        UPDATE users
                        SET free_video_used = free_video_used + 1, updated_at = %s
                        WHERE id = %s AND free_video_used < free_video_limit
                        RETURNING id
                        """,
                        (_now(), user_id),
                    )
                    if cur.fetchone() is None:
                        return {"outcome": "quota_exceeded", "job": None}

                cur.execute(
                    """
                    INSERT INTO generation_jobs
                        (id, user_id, mode, params, s3_input_key, status,
                         queue_priority, created_at, updated_at,
                         idempotency_key, idempotency_fingerprint,
                         charged_subscription_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (job_id, user_id, mode, json.dumps(params), s3_input_key,
                     JOB_STATUS_QUEUED, queue_priority, _now(), _now(),
                     idempotency_key, idempotency_fingerprint, subscription_id),
                )
                created = cur.fetchone()
            return {"outcome": "created", "job": dict(created)}
    finally:
        conn.close()


def get_job_by_idempotency_key(user_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the job that owns a given (user_id, idempotency_key) pair --
    called only after create_job_idempotent() reports a conflict, to
    compare idempotency_fingerprint and decide replay vs 409.
    """
    conn = _get_conn()
    try:
        return _row(
            conn,
            "SELECT * FROM generation_jobs WHERE user_id = %s AND idempotency_key = %s",
            (user_id, idempotency_key),
        )
    finally:
        conn.close()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        return _row(conn, "SELECT * FROM generation_jobs WHERE id = %s", (job_id,))
    finally:
        conn.close()


def list_user_jobs(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        if status:
            sql = """
                SELECT * FROM generation_jobs
                WHERE user_id = %s AND status = %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
            """
            params = (user_id, status, limit, offset)
        else:
            sql = """
                SELECT * FROM generation_jobs
                WHERE user_id = %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)
        return _rows(conn, sql, params)
    finally:
        conn.close()


def update_job_status(job_id: str, status: str) -> None:
    """Update job to any valid status (including granular pipeline steps)."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE generation_jobs SET status=%s, updated_at=%s WHERE id=%s",
                    (status, _now(), job_id),
                )
    finally:
        conn.close()


def cancel_orphan_job(job_id: str) -> bool:
    """
    Cancels a just-created generation_jobs row that lost a concurrent race
    to attach itself somewhere (see streamer_routes.py's POST .../selection:
    two simultaneous requests can both pass create_job() for the same
    segment before either reaches attach_streamer_clip_job(), and the
    join-table's own PRIMARY KEY on segment_id lets only one of those
    attachments win). The losing job_id would otherwise sit at status
    'queued' forever with nothing referencing it -- NOT harmless, since
    get_next_queued_job_for_dispatch() picks up ANY queued row regardless
    of whether anything is attached to it, so a real GPU worker would
    eventually claim, download, and process a duplicate clip nobody will
    ever see.

    Atomic conditional UPDATE, only from 'queued' -- if the tiny race
    window meant a worker already claimed this job before we got here
    (status is no longer 'queued'), this is a no-op (returns False): let
    that in-flight run finish rather than cancel work already underway.
    Returns True if this call actually cancelled the job.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE generation_jobs SET status=%s, updated_at=%s "
                    "WHERE id=%s AND status=%s",
                    (JOB_STATUS_CANCELLED, _now(), job_id, JOB_STATUS_QUEUED),
                )
                return cur.rowcount > 0
    finally:
        conn.close()


def complete_job(
    job_id: str,
    s3_output_key: str,
    clip_count: Optional[int] = None,
    processing_ms: Optional[int] = None,
    enrichment_keys: Optional[List[str]] = None,
) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs SET
                        status          = %s,
                        s3_output_key   = %s,
                        clip_count      = %s,
                        processing_ms   = %s,
                        enrichment_keys = %s,
                        completed_at    = %s,
                        updated_at      = %s
                    WHERE id = %s
                    """,
                    (JOB_STATUS_COMPLETED, s3_output_key, clip_count, processing_ms,
                     enrichment_keys or [], _now(), _now(), job_id),
                )
    finally:
        conn.close()


def fail_job(
    job_id: str,
    error_code: str,
    error_message: str,
    retry: bool = True,
) -> None:
    """
    Mark job as failed.
    If retry=True and retry_count < max_retries: requeue as 'queued'.
    If retry=False or exhausted: mark 'failed' permanently.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT retry_count, max_retries FROM generation_jobs WHERE id=%s",
                    (job_id,),
                )
                row = cur.fetchone()
                if not row:
                    logger.warning("[job_store] fail_job: job not found id=%s", job_id)
                    return
                retry_count = row["retry_count"]
                max_retries = row["max_retries"]
                can_retry   = retry and (retry_count < max_retries)
                new_status  = JOB_STATUS_QUEUED if can_retry else JOB_STATUS_FAILED

                cur.execute(
                    """
                    UPDATE generation_jobs SET
                        status      = %s,
                        last_error  = %s,
                        error       = %s,
                        retry_count = retry_count + 1,
                        updated_at  = %s
                    WHERE id = %s
                    """,
                    (new_status, error_code, error_message[:2000], _now(), job_id),
                )
                if can_retry:
                    logger.info("[job_store] job requeued job_id=%s attempt=%d/%d",
                                job_id, retry_count + 1, max_retries)
                else:
                    logger.warning("[job_store] job permanently failed job_id=%s code=%s",
                                   job_id, error_code)
    finally:
        conn.close()


# ── Claim / poll ———————————————————————————————————————————————————————————————

def claim_next_pending_job(
    worker_id: str,
    modes: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Atomically claim the next queued job using FOR UPDATE SKIP LOCKED.
    Transitions status: queued → claimed.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if modes:
                    cur.execute(
                        """
                        UPDATE generation_jobs
                        SET status     = %s,
                            worker_id  = %s,
                            claimed_at = %s,
                            started_at = %s,
                            updated_at = %s
                        WHERE id = (
                            SELECT id FROM generation_jobs
                            WHERE status = %s AND mode = ANY(%s)
                            ORDER BY queue_priority DESC, created_at ASC
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                        """,
                        (JOB_STATUS_CLAIMED, worker_id, _now(), _now(), _now(),
                         JOB_STATUS_QUEUED, list(modes)),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE generation_jobs
                        SET status     = %s,
                            worker_id  = %s,
                            claimed_at = %s,
                            started_at = %s,
                            updated_at = %s
                        WHERE id = (
                            SELECT id FROM generation_jobs
                            WHERE status = %s
                            ORDER BY queue_priority DESC, created_at ASC
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                        """,
                        (JOB_STATUS_CLAIMED, worker_id, _now(), _now(), _now(),
                         JOB_STATUS_QUEUED),
                    )
                row = cur.fetchone()
                return dict(row) if row else None
    finally:
        conn.close()


# Statuses from which a job may still be claimed by the worker the GPU
# dispatcher already provisioned an instance for. The dispatcher sets
# status=gpu_requested (mark_gpu_requested) immediately after requesting an
# ephemeral instance -- well before that instance boots and its worker gets
# a chance to call /api/worker/claim. If only 'queued' were accepted here,
# the claim would always find 0 matching rows and the job would get stuck
# in gpu_requested until the startup-SLA timeout destroys the instance and
# blacklists the (innocent) host. gpu_booting is included for the same
# reason even though nothing sets it today.
_CLAIMABLE_STATUSES = (JOB_STATUS_QUEUED, JOB_STATUS_GPU_REQUESTED, JOB_STATUS_GPU_BOOTING)


def claim_specific_job(job_id: str, worker_id: str) -> Optional[Dict[str, Any]]:
    """
    Claim a specific job by ID. Returns None if not claimable.

    Accepts status in _CLAIMABLE_STATUSES (not just 'queued') -- see comment
    above. Also stamps worker_started_at atomically with the claim, so
    get_stale_gpu_requested_jobs() correctly stops treating this job as an
    unclaimed/stuck startup once a worker has actually checked in.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET status            = %s,
                        worker_id         = %s,
                        claimed_at        = %s,
                        started_at        = %s,
                        worker_started_at = %s,
                        updated_at        = %s
                    WHERE id = %s AND status = ANY(%s)
                    RETURNING *
                    """,
                    (JOB_STATUS_CLAIMED, worker_id, _now(), _now(), _now(), _now(),
                     job_id, list(_CLAIMABLE_STATUSES)),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    finally:
        conn.close()


def requeue_stale_jobs(stale_minutes: int = 30) -> int:
    """
    Reset stuck active jobs → queued when stuck longer than stale_minutes.
    Returns count of requeued jobs.

    Liveness check is COALESCE(heartbeat_at, claimed_at): a job whose
    worker is periodically calling touch_job_heartbeat() (long streamer
    enrichment, say) keeps getting a fresh timestamp here and is never
    mistaken for stuck, no matter how long a single status phase runs. A
    job that never heartbeats (every other mode today, and any job whose
    worker actually died) falls back to the original claimed_at-only
    check — unchanged behavior for the case this function already
    handled correctly.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                active_list = list(_ACTIVE_STATUSES)
                placeholders = ",".join(["%s"] * len(active_list))
                cur.execute(
                    f"""
                    UPDATE generation_jobs
                    SET status       = %s,
                        worker_id    = NULL,
                        claimed_at   = NULL,
                        heartbeat_at = NULL,
                        updated_at   = %s
                    WHERE status IN ({placeholders})
                      AND COALESCE(heartbeat_at, claimed_at) < NOW() - INTERVAL '%s minutes'
                      AND retry_count < max_retries
                    """,
                    (JOB_STATUS_QUEUED, _now(), *active_list, stale_minutes),
                )
                return cur.rowcount
    finally:
        conn.close()


def touch_job_heartbeat(job_id: str) -> None:
    """
    Refresh heartbeat_at only — proves the worker processing this job is
    still alive, without implying any status transition (no semantic
    status change, no updated_at bump). See requeue_stale_jobs() above for
    how this is consumed, and gpu_worker.py for the periodic caller that
    starts when a long mode run begins and stops (via try/finally) the
    moment it completes, fails, or the process exits.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE generation_jobs SET heartbeat_at=%s WHERE id=%s",
                    (_now(), job_id),
                )
    finally:
        conn.close()


# ── Files ——————————————————————————————————————————————————————————————————————

def add_job_file(
    job_id: str,
    user_id: str,
    file_type: str,
    s3_key: str,
    filename: str,
    content_type: str = "application/octet-stream",
    size_bytes: Optional[int] = None,
    duration_sec: Optional[float] = None,
) -> str:
    file_id = str(uuid.uuid4())
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO generation_files
                        (id, job_id, user_id, file_type, s3_key, filename,
                         content_type, size_bytes, duration_sec, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (s3_key) DO NOTHING
                    """,
                    (file_id, job_id, user_id, file_type, s3_key, filename,
                     content_type, size_bytes, duration_sec, _now()),
                )
    finally:
        conn.close()
    return file_id


def list_job_files(job_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        return _rows(conn,
                     "SELECT * FROM generation_files WHERE job_id=%s ORDER BY created_at ASC",
                     (job_id,))
    finally:
        conn.close()


# ── GPU dispatcher queue API (migration 006) ————————————————————————————————————

def get_next_queued_job_for_dispatch() -> Optional[Dict[str, Any]]:
    """
    Peek at the next dispatchable job without locking it.

    Selects status='queued', attempts < max_attempts,
    locked_until IS NULL or expired,
    ordered by priority DESC then queued_at ASC (FIFO within same priority).

    Returns the row dict or None.  Use lock_job_for_dispatch to atomically
    acquire the job before calling the orchestrator.
    """
    conn = _get_conn()
    try:
        return _row(
            conn,
            """
            SELECT *
            FROM generation_jobs
            WHERE status = 'queued'
              AND attempts < max_attempts
              AND (locked_until IS NULL OR locked_until < NOW())
            ORDER BY priority DESC, queued_at ASC
            LIMIT 1
            """,
            (),
        )
    finally:
        conn.close()


def lock_job_for_dispatch(job_id: str, lock_seconds: int = 120) -> Optional[Dict[str, Any]]:
    """
    Atomically lock a queued job for the dispatcher.

    Uses SELECT … FOR UPDATE SKIP LOCKED so two dispatcher instances never
    race on the same row.  Increments attempts and sets locked_until.

    Returns the updated row, or None if the job was already taken.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET
                        attempts     = attempts + 1,
                        locked_until = NOW() + (%s || ' seconds')::INTERVAL,
                        updated_at   = NOW()
                    WHERE id = (
                        SELECT id
                        FROM generation_jobs
                        WHERE id = %s
                          AND status = 'queued'
                          AND attempts < max_attempts
                          AND (locked_until IS NULL OR locked_until < NOW())
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING *
                    """,
                    (str(lock_seconds), job_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    finally:
        conn.close()


def mark_gpu_requested(
    job_id: str,
    orchestrator_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Set status=gpu_requested, gpu_status=requested, record payload + timestamp."""
    import json as _json
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET
                        status               = 'gpu_requested',
                        gpu_status           = 'requested',
                        gpu_requested_at     = NOW(),
                        locked_until         = NULL,
                        orchestrator_payload = %s,
                        orchestrator_error   = NULL,
                        updated_at           = NOW()
                    WHERE id = %s
                    """,
                    (_json.dumps(orchestrator_payload or {}), job_id),
                )
    finally:
        conn.close()
    logger.info("[job_store] gpu_requested job_id=%s", job_id)


def mark_gpu_request_failed(job_id: str, error: str) -> None:
    """
    Record a failed GPU orchestration attempt.

    If attempts >= max_attempts → status='failed' + failed_at.
    Otherwise → status='queued' for dispatcher retry.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET
                        gpu_status         = 'request_failed',
                        orchestrator_error = %s,
                        locked_until       = NULL,
                        status = CASE
                            WHEN attempts >= max_attempts THEN 'failed'
                            ELSE 'queued'
                        END,
                        failed_at = CASE
                            WHEN attempts >= max_attempts THEN NOW()
                            ELSE NULL
                        END,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (error[:2000], job_id),
                )
    finally:
        conn.close()
    logger.warning("[job_store] gpu_request_failed job_id=%s error=%.120s", job_id, error)


def get_stale_gpu_requested_jobs(timeout_sec: int) -> List[Dict[str, Any]]:
    """
    Find jobs stuck in 'gpu_requested' past the production Vast startup SLA.

    A job is a startup_timeout when:
      status = 'gpu_requested'
      gpu_requested_at < NOW() - timeout_sec seconds
      worker_started_at IS NULL

    Used by gpu_dispatcher.cleanup_stale_gpu_requests(), which is run before
    every dispatch pass so a bad instance never blocks the queue for longer
    than timeout_sec.
    """
    conn = _get_conn()
    try:
        return _rows(
            conn,
            """
            SELECT *
            FROM generation_jobs
            WHERE status = 'gpu_requested'
              AND worker_started_at IS NULL
              AND gpu_requested_at IS NOT NULL
              AND gpu_requested_at < NOW() - (%s || ' seconds')::INTERVAL
            """,
            (str(timeout_sec),),
        )
    finally:
        conn.close()


def mark_gpu_startup_timeout(
    job_id: str,
    error: str,
    max_startup_retries: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve a Vast startup-SLA timeout for one job.

    attempts is NOT incremented here — it was already incremented by
    lock_job_for_dispatch() at dispatch time, so it already reflects this
    attempt.

    Retry cap: LEAST(generation_jobs.max_attempts, max_startup_retries) when
    max_startup_retries is given (VAST_MAX_STARTUP_RETRIES), otherwise falls
    back to the job's own max_attempts column.

      attempts <  cap  -> status='queued'  (next dispatcher pass retries a
                          different offer/host), orchestrator_error=
                          'startup timeout; retrying another offer'
      attempts >= cap  -> status='failed', failed_at=NOW()

    Returns the updated row so the caller can log the outcome without a
    second query.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if max_startup_retries is not None:
                    cur.execute(
                        """
                        UPDATE generation_jobs
                        SET
                            error              = %(error)s,
                            orchestrator_error = CASE
                                WHEN attempts >= LEAST(max_attempts, %(cap)s) THEN %(error)s
                                ELSE 'startup timeout; retrying another offer'
                            END,
                            gpu_status = CASE
                                WHEN attempts >= LEAST(max_attempts, %(cap)s) THEN 'startup_timeout_failed'
                                ELSE 'startup_timeout_retry'
                            END,
                            status = CASE
                                WHEN attempts >= LEAST(max_attempts, %(cap)s) THEN 'failed'
                                ELSE 'queued'
                            END,
                            failed_at = CASE
                                WHEN attempts >= LEAST(max_attempts, %(cap)s) THEN NOW()
                                ELSE NULL
                            END,
                            locked_until = NULL,
                            updated_at = NOW()
                        WHERE id = %(job_id)s
                        RETURNING *
                        """,
                        {"error": error[:2000], "cap": max_startup_retries, "job_id": job_id},
                    )
                else:
                    cur.execute(
                        """
                        UPDATE generation_jobs
                        SET
                            error              = %(error)s,
                            orchestrator_error = CASE
                                WHEN attempts >= max_attempts THEN %(error)s
                                ELSE 'startup timeout; retrying another offer'
                            END,
                            gpu_status = CASE
                                WHEN attempts >= max_attempts THEN 'startup_timeout_failed'
                                ELSE 'startup_timeout_retry'
                            END,
                            status = CASE
                                WHEN attempts >= max_attempts THEN 'failed'
                                ELSE 'queued'
                            END,
                            failed_at = CASE
                                WHEN attempts >= max_attempts THEN NOW()
                                ELSE NULL
                            END,
                            locked_until = NULL,
                            updated_at = NOW()
                        WHERE id = %(job_id)s
                        RETURNING *
                        """,
                        {"error": error[:2000], "job_id": job_id},
                    )
                row = cur.fetchone()
                result = dict(row) if row else {}
    finally:
        conn.close()
    logger.warning(
        "[job_store] gpu_startup_timeout job_id=%s new_status=%s attempts=%s/%s",
        job_id, result.get("status"), result.get("attempts"), result.get("max_attempts"),
    )
    return result


# ── GPU instance ownership / cleanup ─────────────────────────────────────────

def _parse_orchestrator_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = job.get("orchestrator_payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return {}
    return payload if isinstance(payload, dict) else {}


def get_ephemeral_contract_id(job: Dict[str, Any]) -> Optional[str]:
    """
    Return the vast.ai contract_id whose ephemeral instance should be
    destroyed now that `job` has reached a terminal state, or None if it
    must NOT be destroyed.

    Fail-safe by construction — returns None (never destroy) unless ALL of:
      - job["status"] is terminal (completed/failed/cancelled). A job still
        mid-flight is never a candidate, no matter what its payload says.
      - job["orchestrator_payload"]["gpu_managed_by"] == GPU_MANAGED_BY_VAST_EPHEMERAL
        — stamped only by gpu_orchestrator._trigger_vast(). Any other value,
        or the key missing entirely (manually-managed GPU, or orchestrator_payload
        itself missing/null/unparseable), returns None.
      - a contract_id is present in that payload.

    Never raises — a malformed payload (bad JSON, wrong type) is treated the
    same as "no payload", i.e. never destroy. This is the single source of
    truth for "is this job's GPU instance safe to destroy automatically";
    callers must not destroy based on any other signal.
    """
    if job.get("status") not in _TERMINAL_STATUSES:
        return None
    payload = _parse_orchestrator_payload(job)
    if payload.get("gpu_managed_by") != GPU_MANAGED_BY_VAST_EPHEMERAL:
        return None
    contract_id = payload.get("contract_id")
    return str(contract_id) if contract_id else None


def _merge_orchestrator_payload(job_id: str, patch: Dict[str, Any]) -> None:
    """
    Shallow-merge `patch` into orchestrator_payload via Postgres JSONB `||`
    (top-level keys in `patch` overwrite existing ones; everything else in
    the column — contract_id, gpu_managed_by, offer/host identifiers — is
    left untouched). Works whether the existing value is NULL or a JSONB
    object. No schema change: orchestrator_payload already has no fixed key
    set (migration 006).
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET orchestrator_payload = COALESCE(orchestrator_payload, '{}'::jsonb) || %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(patch), job_id),
                )
    finally:
        conn.close()


def mark_gpu_instance_cleanup_destroyed(job_id: str) -> None:
    """
    Record that the job's ephemeral vast.ai instance has been confirmed
    destroyed (including "already gone" / 404 — see destroy_vast_instance).
    After this, get_terminal_jobs_pending_gpu_cleanup() no longer returns
    this job, so the dispatcher's reconciliation pass stops re-attempting it.
    """
    _merge_orchestrator_payload(job_id, {
        "gpu_cleanup_status": GPU_CLEANUP_STATUS_DESTROYED,
        "gpu_cleanup_last_attempt_at": _now().isoformat(),
        "gpu_cleanup_error": None,
    })
    logger.info("[job_store] gpu_instance_cleanup_destroyed job_id=%s", job_id)


def mark_gpu_instance_cleanup_error(job_id: str, error: str) -> None:
    """
    Record a failed destroy attempt. Deliberately does NOT touch job
    status/completed_at/failed_at — a Vast API error must never turn a
    completed job into a failed one. gpu_cleanup_status stays anything-but-
    "destroyed", so get_terminal_jobs_pending_gpu_cleanup() keeps returning
    this job for the next dispatcher tick to retry.
    """
    _merge_orchestrator_payload(job_id, {
        "gpu_cleanup_status": GPU_CLEANUP_STATUS_ERROR,
        "gpu_cleanup_last_attempt_at": _now().isoformat(),
        "gpu_cleanup_error": (error or "")[:2000],
    })
    logger.warning("[job_store] gpu_instance_cleanup_error job_id=%s error=%.120s", job_id, error)


def get_terminal_jobs_pending_gpu_cleanup() -> List[Dict[str, Any]]:
    """
    Find terminal jobs whose ephemeral vast.ai instance is not yet confirmed
    destroyed. Used by gpu_dispatcher.reconcile_gpu_instance_cleanup() — the
    durable retry path for cleanup_instance_for_terminal_job(), covering the
    window where the FastAPI BackgroundTask fast path never ran (API process
    crashed/restarted between committing the terminal status and running the
    task) as well as any prior destroy attempt that errored.

    A job is a candidate when ALL of:
      status              terminal (completed/failed/cancelled)
      gpu_managed_by       = 'vast_ephemeral'  (never true for the manually-
                            managed production GPU, or any job with a
                            missing/unrecognized ownership stamp)
      contract_id          present
      gpu_cleanup_status   NOT 'destroyed' (covers both "never attempted"
                            and "attempted and failed" — same retry path)

    Manually-managed and unknown-ownership jobs can never match — there is
    no contract_id/gpu_managed_by stamp for the query to find.
    """
    conn = _get_conn()
    try:
        return _rows(
            conn,
            """
            SELECT *
            FROM generation_jobs
            WHERE status = ANY(%s)
              AND orchestrator_payload->>'gpu_managed_by' = %s
              AND orchestrator_payload->>'contract_id' IS NOT NULL
              AND COALESCE(orchestrator_payload->>'gpu_cleanup_status', '') <> %s
            """,
            (list(_TERMINAL_STATUSES), GPU_MANAGED_BY_VAST_EPHEMERAL, GPU_CLEANUP_STATUS_DESTROYED),
        )
    finally:
        conn.close()


def mark_worker_started(job_id: str) -> None:
    """Transition to worker_started and record timestamp."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET
                        status            = 'worker_started',
                        gpu_status        = 'worker_running',
                        worker_started_at = NOW(),
                        updated_at        = NOW()
                    WHERE id = %s
                    """,
                    (job_id,),
                )
    finally:
        conn.close()
    logger.info("[job_store] worker_started job_id=%s", job_id)


def mark_job_completed(
    job_id: str,
    s3_output_key: str,
    clip_count: Optional[int] = None,
    processing_ms: Optional[int] = None,
) -> None:
    """Dispatcher-friendly complete: delegates to complete_job + sets gpu_status=done."""
    complete_job(
        job_id=job_id,
        s3_output_key=s3_output_key,
        clip_count=clip_count,
        processing_ms=processing_ms,
    )
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE generation_jobs SET gpu_status='gpu_completed', updated_at=NOW() WHERE id=%s",
                    (job_id,),
                )
    finally:
        conn.close()
    logger.info("[job_store] job_completed_dispatcher job_id=%s", job_id)


def mark_job_failed(job_id: str, error_code: str, error_message: str) -> None:
    """Dispatcher-friendly fail: delegates to fail_job + sets gpu_status=failed + failed_at."""
    fail_job(
        job_id=job_id,
        error_code=error_code,
        error_message=error_message,
        retry=False,
    )
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET gpu_status='failed', failed_at=NOW(), updated_at=NOW()
                    WHERE id=%s
                    """,
                    (job_id,),
                )
    finally:
        conn.close()
    logger.warning("[job_store] job_failed_dispatcher job_id=%s", job_id)


def count_active_gpu_jobs() -> int:
    """
    Count jobs currently in GPU-active states.
    Used by the dispatcher to enforce MAX_ACTIVE_GPU_JOBS concurrency cap.
    """
    conn = _get_conn()
    try:
        row = _row(
            conn,
            "SELECT COUNT(*) AS n FROM generation_jobs WHERE status = ANY(%s)",
            (list(_ACTIVE_STATUSES),),
        )
        return int(row["n"]) if row else 0
    finally:
        conn.close()


# ── Streamer batches (PHASE A batch foundation, migration 012) ─────────────────
#
# streamer_batches / streamer_segments / streamer_clip_jobs. See
# modes/streamer/runner.py's analyze()/compose_one() split and the
# PHASE A / batch-foundation audit for the architecture this backs.
#
# Store layer only — no HTTP endpoints, no worker phase dispatch yet. Every
# read that takes a user_id enforces ownership in the query itself (defense
# in depth, not dependent on a future API-layer check remembering to).
#
# generation_jobs itself is NOT touched by any of this — batch-awareness is
# entirely a side table (streamer_clip_jobs), never a new column or FK
# direction on generation_jobs's own load-bearing schema.

STREAMER_BATCH_STATUS_QUEUED             = "queued"
STREAMER_BATCH_STATUS_INGESTING          = "ingesting"
STREAMER_BATCH_STATUS_ANALYZING          = "analyzing"
STREAMER_BATCH_STATUS_AWAITING_SELECTION = "awaiting_selection"
STREAMER_BATCH_STATUS_GENERATING         = "generating"
STREAMER_BATCH_STATUS_READY              = "ready"
STREAMER_BATCH_STATUS_PARTIALLY_FAILED   = "partially_failed"
STREAMER_BATCH_STATUS_FAILED             = "failed"
STREAMER_BATCH_STATUS_CANCELLED          = "cancelled"

_STREAMER_BATCH_TERMINAL_STATUSES = frozenset({
    STREAMER_BATCH_STATUS_READY,
    STREAMER_BATCH_STATUS_PARTIALLY_FAILED,
    STREAMER_BATCH_STATUS_FAILED,
    STREAMER_BATCH_STATUS_CANCELLED,
})


class StreamerBatchTransitionError(ValueError):
    """Raised by set_streamer_batch_status() when asked to move a batch
    out of a terminal status — see that function's docstring."""


class StreamerClipJobConflictError(ValueError):
    """Raised by attach_streamer_clip_job() when a segment already has a
    DIFFERENT job attached — see that function's docstring."""


class StreamerSegmentsLockedError(ValueError):
    """Raised by replace_streamer_segments() when the batch already has at
    least one streamer_clip_jobs row attached — see that function's
    docstring."""


def create_streamer_batch(
    user_id: str,
    preset_snapshot: Dict[str, Any],
    analysis_job_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new batch, optionally already pointed at an analysis job.
    analysis_job_id is nullable by design (migration 012): a batch exists
    from the start of ingest, before the analysis generation_jobs row
    necessarily exists yet — see attach_analysis_job_to_batch() for
    attaching it once it does.

    idempotency_key (migration 014) makes batch creation itself idempotent:
    a double-click or network retry on POST /api/streamer/batches with the
    SAME (user_id, idempotency_key) must not create a second batch (and,
    downstream, must not charge Free-plan quota twice). Enforced via an
    atomic INSERT ... ON CONFLICT (user_id, idempotency_key) DO NOTHING —
    exact same pattern as create_job_idempotent() (migration 009) — not a
    SELECT-then-INSERT, so two truly concurrent requests with the same key
    still resolve to exactly one row. idempotency_key=None (no header —
    legacy behavior) never conflicts with anything: NULL is distinct from
    every other NULL under PostgreSQL UNIQUE semantics, so this is a no-op
    change for every existing caller that doesn't pass it.

    The returned dict carries one extra, non-persisted key —
    "_created_now": True when this call's own INSERT won (a genuinely new
    batch), False when it returned a pre-existing row instead. The caller
    (POST /api/streamer/batches) uses this to decide whether to proceed
    with ingest/quota at all, or short-circuit and return the existing
    batch's current state untouched.
    """
    batch_id = str(uuid.uuid4())
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO streamer_batches
                        (id, user_id, analysis_job_id, preset_snapshot, status,
                         idempotency_key, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, idempotency_key) DO NOTHING
                    RETURNING *
                    """,
                    (batch_id, user_id, analysis_job_id, json.dumps(preset_snapshot),
                     STREAMER_BATCH_STATUS_QUEUED, idempotency_key, _now(), _now()),
                )
                row = cur.fetchone()
                if row:
                    result = dict(row)
                    result["_created_now"] = True
                    return result

                existing = _row(
                    conn,
                    "SELECT * FROM streamer_batches WHERE user_id=%s AND idempotency_key=%s",
                    (user_id, idempotency_key),
                )
                result = dict(existing)
                result["_created_now"] = False
                return result
    finally:
        conn.close()


def get_streamer_batch(batch_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    user_id is required, not optional — ownership is enforced in the query
    itself (WHERE id=... AND user_id=...) so a caller can never accidentally
    return another user's batch by forgetting a check afterward. A batch
    that exists but belongs to someone else returns None, same as one that
    doesn't exist at all — never distinguishes the two.
    """
    conn = _get_conn()
    try:
        return _row(
            conn,
            "SELECT * FROM streamer_batches WHERE id = %s AND user_id = %s",
            (batch_id, user_id),
        )
    finally:
        conn.close()


def list_user_streamer_batches(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        if status:
            sql = """
                SELECT * FROM streamer_batches
                WHERE user_id = %s AND status = %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
            """
            params = (user_id, status, limit, offset)
        else:
            sql = """
                SELECT * FROM streamer_batches
                WHERE user_id = %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)
        return _rows(conn, sql, params)
    finally:
        conn.close()


def set_streamer_batch_status(
    batch_id: str,
    status: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update a batch's status. Minimal transition guard only, not a state-
    machine framework: once a batch is in a terminal status (ready /
    partially_failed / failed / cancelled), it can never move to a
    DIFFERENT status again — e.g. ready -> analyzing or failed ->
    generating both raise StreamerBatchTransitionError. Setting the SAME
    terminal status again is a no-op success (idempotent replay), not an
    error. No ordering/forward-flow validation beyond that — any
    non-terminal -> any status is allowed, same looseness
    update_job_status() already has for generation_jobs.

    There is no existing "transition validation" layer anywhere in this
    codebase to place this above instead — update_job_status() for
    generation_jobs is a plain unconditional UPDATE, legality enforced
    purely by which callers invoke it in what order. This is a new,
    deliberately minimal rule specific to batches, living in the store
    layer because no batch API layer exists yet to put it in instead.

    The guard is enforced by the UPDATE's own WHERE clause (status <>
    ALL(terminal_statuses) OR status = the new status), NOT by a separate
    SELECT-then-UPDATE — a check-then-write here would leave a window for
    a second, concurrent call to slip a real transition through between
    the read and the write. The single statement is atomic: Postgres
    evaluates WHERE against the row as it stands at that instant, so two
    concurrent set_streamer_batch_status() calls on the same terminal
    batch can never both "win" — at most one row-affecting UPDATE happens,
    same guarantee create_job_with_quota's ON CONFLICT already relies on
    elsewhere in this file, just via WHERE instead of a unique index. The
    follow-up SELECT below only ever runs to build a clear error message
    after the UPDATE already found 0 rows to touch — it never decides
    whether the mutation happens.

    Raises ValueError if batch_id doesn't exist, StreamerBatchTransitionError
    if it exists but is terminal and `status` differs from its current one.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # COALESCE keeps the original completed_at on an idempotent
                # same-terminal-status replay, and leaves it untouched (NULL)
                # for a non-terminal status.
                new_completed_at = _now() if status in _STREAMER_BATCH_TERMINAL_STATUSES else None
                cur.execute(
                    """
                    UPDATE streamer_batches
                    SET status = %s, error = %s, updated_at = %s,
                        completed_at = COALESCE(completed_at, %s)
                    WHERE id = %s
                      AND (status <> ALL(%s) OR status = %s)
                    RETURNING *
                    """,
                    (status, error, _now(), new_completed_at, batch_id,
                     list(_STREAMER_BATCH_TERMINAL_STATUSES), status),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)

                # 0 rows affected — either the batch doesn't exist, or it's
                # terminal and this was a real (rejected) transition. This
                # SELECT is diagnostic only, purely to phrase the right
                # error; see the atomicity note above.
                existing = _row(conn, "SELECT * FROM streamer_batches WHERE id=%s", (batch_id,))
                if not existing:
                    raise ValueError(f"streamer_batch not found: {batch_id}")
                raise StreamerBatchTransitionError(
                    f"streamer_batch {batch_id} is terminal ({existing['status']!r}) — "
                    f"cannot transition to {status!r}"
                )
    finally:
        conn.close()


def attach_analysis_job_to_batch(batch_id: str, analysis_job_id: str) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE streamer_batches SET analysis_job_id=%s, updated_at=%s WHERE id=%s",
                    (analysis_job_id, _now(), batch_id),
                )
    finally:
        conn.close()


def replace_streamer_segments(
    batch_id: str,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Atomically replace ALL segments for a batch: delete every existing row
    for this batch_id, then insert the given list, one transaction — no
    orphaned rows left behind from a previous analysis pass, and a failure
    partway through never leaves a mix of old and new segments.

    Each item in `segments` must provide start_sec/duration_sec/title;
    score/description/recommended/selected/crop_hints/metadata are
    optional (DB defaults apply: recommended=False, selected=True).
    `ordinal` is taken from list position (0-based) unless given
    explicitly. See segments_from_analysis() for building this list from
    modes.streamer.runner.analyze()'s own return shape.

    Only callable before any streamer_clip_jobs exist for this batch —
    enforced, not just documented: raises StreamerSegmentsLockedError if
    even one clip job is already attached. Without this guard, ON DELETE
    CASCADE on streamer_clip_jobs.segment_id would silently delete those
    join rows the moment their segment is deleted here (never the
    underlying generation_jobs rows themselves, but the batch would
    silently lose track of jobs already in flight). Intended caller is the
    analyze phase, once, before any selection/generation has happened; if
    segments genuinely need to change after generation started, that's a
    deliberate decision for a caller to make explicitly (e.g. cancel the
    batch first), not something this function does on its own.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM streamer_clip_jobs WHERE batch_id=%s LIMIT 1",
                    (batch_id,),
                )
                if cur.fetchone():
                    raise StreamerSegmentsLockedError(
                        f"streamer_batch {batch_id} already has clip jobs attached — "
                        f"refusing to replace its segments"
                    )
                cur.execute("DELETE FROM streamer_segments WHERE batch_id=%s", (batch_id,))
                inserted: List[Dict[str, Any]] = []
                for i, seg in enumerate(segments):
                    seg_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO streamer_segments
                            (id, batch_id, ordinal, start_sec, duration_sec, title,
                             description, score, recommended, selected,
                             crop_hints, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                        """,
                        (
                            seg_id, batch_id, seg.get("ordinal", i),
                            seg["start_sec"], seg["duration_sec"], seg["title"],
                            seg.get("description"), seg.get("score"),
                            seg.get("recommended", False), seg.get("selected", True),
                            json.dumps(seg.get("crop_hints") or {}),
                            json.dumps(seg.get("metadata") or {}),
                            _now(), _now(),
                        ),
                    )
                    inserted.append(dict(cur.fetchone()))
                return inserted
    finally:
        conn.close()


def list_streamer_segments(batch_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        return _rows(
            conn,
            "SELECT * FROM streamer_segments WHERE batch_id=%s ORDER BY ordinal ASC",
            (batch_id,),
        )
    finally:
        conn.close()


def set_streamer_segment_selection(segment_id: str, selected: bool) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE streamer_segments SET selected=%s, updated_at=%s WHERE id=%s",
                    (selected, _now(), segment_id),
                )
    finally:
        conn.close()


def attach_streamer_clip_job(batch_id: str, segment_id: str, job_id: str) -> Dict[str, Any]:
    """
    Link a real generation_jobs row to a segment. segment_id is PRIMARY KEY
    on streamer_clip_jobs (migration 012), so at most one job can ever be
    attached to a given segment.

    Idempotent for an exact repeat: calling this again with the SAME
    segment_id + job_id that's already attached returns the existing row,
    no error — matches this codebase's existing idempotency-key philosophy
    (see create_job_with_quota's own "existing" outcome). A genuine
    conflict — segment_id already attached to a DIFFERENT job_id — raises
    StreamerClipJobConflictError instead of silently overwriting the link.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO streamer_clip_jobs (batch_id, segment_id, job_id, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (segment_id) DO NOTHING
                    RETURNING *
                    """,
                    (batch_id, segment_id, job_id, _now()),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)

                existing = _row(
                    conn,
                    "SELECT * FROM streamer_clip_jobs WHERE segment_id=%s",
                    (segment_id,),
                )
                if existing and existing["job_id"] == job_id:
                    return existing  # idempotent replay
                raise StreamerClipJobConflictError(
                    f"segment {segment_id} is already attached to a different job "
                    f"({existing['job_id'] if existing else '?'} != {job_id})"
                )
    finally:
        conn.close()


def list_streamer_clip_jobs(batch_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        return _rows(
            conn,
            "SELECT * FROM streamer_clip_jobs WHERE batch_id=%s ORDER BY created_at ASC",
            (batch_id,),
        )
    finally:
        conn.close()


def segments_from_analysis(
    analysis_result: Dict[str, Any],
    titles: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Maps modes.streamer.runner.analyze()'s return shape into the dict shape
    replace_streamer_segments() expects — the segment-persistence "store
    contract" from the PHASE A brief. Called by submit_streamer_analysis_
    result() (see scripts/gpu_worker.py's analyze-phase branch and POST
    /api/worker/streamer/batches/{id}/analysis-result).

    Per-segment title/description/recommended are read straight off each
    input segment dict when present, falling back to a generic numbered
    title / no description / not-recommended otherwise. Today's real
    analyze() output (modes/streamer/runner.py, via _fallback_segments())
    never includes these keys — only start_sec/duration_sec/score/source —
    so this fallback is what actually runs in production right now; the
    pass-through exists so a caller that DOES have real per-segment
    content (a future real analyzer, or a test/dev harness submitting
    fixture segments through this exact same worker contract) doesn't have
    it silently discarded. `titles` (positional, index-matched) is a
    second, narrower override kept for backward compatibility with any
    existing caller — a segment's own "title" key wins if both are given.
    """
    segments = analysis_result.get("segments", [])
    crop_hints = analysis_result.get("crop_hints") or {}
    out = []
    for i, seg in enumerate(segments):
        if titles and i < len(titles):
            title = titles[i]
        else:
            title = seg.get("title") or f"Тема {i + 1}"
        out.append({
            "ordinal": i,
            "start_sec": seg["start_sec"],
            "duration_sec": seg["duration_sec"],
            "title": title,
            "description": seg.get("description"),
            "score": seg.get("score"),
            "recommended": bool(seg.get("recommended", False)),
            "crop_hints": crop_hints,
            "metadata": {"source": seg.get("source")},
        })
    return out


def get_streamer_batch_by_id(batch_id: str) -> Optional[Dict[str, Any]]:
    """
    Unscoped (no user_id) batch lookup — worker-internal use only (see
    POST /api/worker/streamer/batches/{batch_id}/analysis-result in
    scripts/streamer_routes.py, authenticated by WORKER_SECRET, which has
    no session/user context to scope by). Every browser-facing route must
    keep using get_streamer_batch(batch_id, user_id) instead, which
    enforces ownership in the query itself.
    """
    conn = _get_conn()
    try:
        return _row(conn, "SELECT * FROM streamer_batches WHERE id = %s", (batch_id,))
    finally:
        conn.close()


def submit_streamer_analysis_result(
    batch_id: str,
    analysis_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Single entry point for "the analyze phase worker job just finished":
    normalize analyze()'s raw segments via segments_from_analysis() and
    persist them, then move the batch to awaiting_selection.

    Shared by both gpu_worker.py backend modes: the "db" worker calls this
    directly (see _db_submit_streamer_segments), and the "api" worker
    reaches it indirectly via POST /api/worker/streamer/batches/{id}/
    analysis-result (scripts/streamer_routes.py), which just calls this
    same function server-side.

    Idempotent under retry: the analyze job's OWN generation_jobs row can
    be retried by the dispatcher after a TRANSIENT failure in a step that
    runs AFTER segments were already successfully persisted here (e.g. the
    analysis-manifest S3 upload in gpu_worker.py's
    _finish_streamer_analyze_job — see retry=True there) — re-running
    analyze() and calling this function a second time for the SAME batch
    must NOT delete and re-insert segments with fresh UUIDs: that would
    invalidate anything the frontend already rendered/cached, and could
    turn an in-flight POST .../selection (using the first call's segment
    ids) into a spurious "foreign_segment" rejection. So: if this batch
    already has segments, this call is a no-op that returns them
    unchanged -- first-call-wins, not last-call-wins. Locks the batch row
    (SELECT ... FOR UPDATE) before checking, so two genuinely concurrent
    submits for the same batch_id serialize here instead of both reading
    "no segments yet" and both inserting.

    Does NOT use replace_streamer_segments() (that function is an
    unconditional force-replace, still used/tested as its own primitive
    elsewhere) -- this is a separate, narrower "insert once" contract.

    Deliberately does NOT touch analysis_job_id (already attached at
    batch-creation time — see attach_analysis_job_to_batch()) or the
    analysis generation_jobs row's own status (the caller still reports
    that job's own completion through the normal worker/complete flow,
    separately — this function is purely about the batch and its
    segments).
    """
    segments = segments_from_analysis(analysis_result)
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM streamer_batches WHERE id=%s FOR UPDATE", (batch_id,))
                cur.execute(
                    "SELECT * FROM streamer_segments WHERE batch_id=%s ORDER BY ordinal ASC",
                    (batch_id,),
                )
                existing = [dict(r) for r in cur.fetchall()]
                if existing:
                    return existing

                inserted: List[Dict[str, Any]] = []
                for i, seg in enumerate(segments):
                    seg_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO streamer_segments
                            (id, batch_id, ordinal, start_sec, duration_sec, title,
                             description, score, recommended, selected,
                             crop_hints, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                        """,
                        (
                            seg_id, batch_id, seg.get("ordinal", i),
                            seg["start_sec"], seg["duration_sec"], seg["title"],
                            seg.get("description"), seg.get("score"),
                            seg.get("recommended", False), seg.get("selected", True),
                            json.dumps(seg.get("crop_hints") or {}),
                            json.dumps(seg.get("metadata") or {}),
                            _now(), _now(),
                        ),
                    )
                    inserted.append(dict(cur.fetchone()))
    finally:
        conn.close()

    set_streamer_batch_status(batch_id, STREAMER_BATCH_STATUS_AWAITING_SELECTION)
    return inserted


def reconcile_streamer_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    """
    Defensive batch-status reconciliation from its compose (streamer_clip_
    jobs) generation_jobs' current statuses. Called from two places: right
    after a compose job's worker/complete or worker/fail endpoint runs
    (the primary trigger), and defensively from GET /api/streamer/batches/
    {id} in case that hook was ever missed (worker crash between its own
    /complete call and this reconciliation, process restart, etc).

    No-op (returns the batch as-is) if:
      - the batch doesn't exist (returns None)
      - the batch is already terminal (ready/partially_failed/failed/
        cancelled) — reconciliation never re-opens a settled batch
      - the batch has no streamer_clip_jobs yet (still awaiting_selection
        or earlier — nothing to reconcile)
      - at least one attached clip job's generation_jobs row is still
        non-terminal (queued/claimed/...) — batch stays "generating",
        no status write happens at all (idempotent no-op, not even a
        same-status UPDATE)

    Once every attached clip job has reached a terminal generation_jobs
    status (completed/failed/cancelled):
      - all completed                       -> ready
      - none completed (all failed/cancelled) -> failed
      - a mix                               -> partially_failed

    Uses set_streamer_batch_status()'s own atomic terminal-transition guard
    — this function never needs its own locking on top of that.
    """
    conn = _get_conn()
    try:
        batch = _row(conn, "SELECT * FROM streamer_batches WHERE id=%s", (batch_id,))
    finally:
        conn.close()
    if not batch:
        return None
    if batch["status"] in _STREAMER_BATCH_TERMINAL_STATUSES:
        return batch

    clip_jobs = list_streamer_clip_jobs(batch_id)
    if not clip_jobs:
        return batch

    _TERMINAL_JOB_STATUSES = {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}
    statuses = []
    for cj in clip_jobs:
        job = get_job(cj["job_id"])
        statuses.append(job["status"] if job else JOB_STATUS_FAILED)

    if not all(s in _TERMINAL_JOB_STATUSES for s in statuses):
        return batch

    completed = sum(1 for s in statuses if s == JOB_STATUS_COMPLETED)
    if completed == len(statuses):
        return set_streamer_batch_status(batch_id, STREAMER_BATCH_STATUS_READY)
    if completed == 0:
        return set_streamer_batch_status(batch_id, STREAMER_BATCH_STATUS_FAILED)
    return set_streamer_batch_status(batch_id, STREAMER_BATCH_STATUS_PARTIALLY_FAILED)


# ── @sonya_group_bot completion notification delivery (migration 015) ──────────
#
# telegram_notified_at (migration 012) is the once-only "already sent"
# guard; these three functions add a lease-based atomic CLAIM on top of it
# so at most one process is ever actively attempting a send for a given
# batch at a time -- see scripts/streamer_notify.py's notify_streamer_
# batch_completion() for the orchestration that calls these.
#
# Exactly-once is NOT achievable here: Telegram's sendMessage and this
# module's Postgres commit are two separate systems with no shared
# transaction. If the process dies in the narrow window after Telegram
# has accepted the message but before mark_streamer_notification_sent()
# commits, a retry will send a duplicate. SONYA's policy is deliberately
# at-least-once (never silently losing a notification) over a complex
# pseudo-transactional scheme that can't actually make this exactly-once
# anyway -- see the module docstring in scripts/streamer_notify.py.

_NOTIFY_CLAIM_LEASE_SECONDS = 300

_NOTIFY_ELIGIBLE_STATUSES = (STREAMER_BATCH_STATUS_READY, STREAMER_BATCH_STATUS_PARTIALLY_FAILED)


def claim_streamer_notification(batch_id: str) -> bool:
    """
    Atomic conditional UPDATE, not a SELECT-then-UPDATE: succeeds only if
    the batch is currently eligible (status ready/partially_failed),
    hasn't already been sent (telegram_notified_at IS NULL), and has no
    ACTIVE claim -- telegram_notify_claimed_at is NULL, or older than
    _NOTIFY_CLAIM_LEASE_SECONDS (a stale claim from a crashed process, or
    the backoff window after a previous failed attempt -- see this
    column's own doc note in migration 015 for why a failure doesn't
    clear it). Increments telegram_notify_attempts on every successful
    claim, including retries.

    Returns True if THIS call won the claim (the caller must now actually
    attempt the send and call mark_streamer_notification_sent() or
    mark_streamer_notification_failed()); False if ineligible, already
    sent, or another process currently holds the claim.
    """
    now = _now()
    cutoff = now - timedelta(seconds=_NOTIFY_CLAIM_LEASE_SECONDS)
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE streamer_batches
                    SET telegram_notify_claimed_at = %s,
                        telegram_notify_attempts = telegram_notify_attempts + 1,
                        updated_at = %s
                    WHERE id = %s
                      AND status = ANY(%s)
                      AND telegram_notified_at IS NULL
                      AND (telegram_notify_claimed_at IS NULL OR telegram_notify_claimed_at < %s)
                    """,
                    (now, now, batch_id, list(_NOTIFY_ELIGIBLE_STATUSES), cutoff),
                )
                return cur.rowcount > 0
    finally:
        conn.close()


def mark_streamer_notification_sent(batch_id: str) -> None:
    """
    Called ONLY after a confirmed 2xx from Telegram (see telegram_bot.
    send_message's own True/False contract) -- never before the send is
    attempted. telegram_notified_at being non-NULL is the permanent,
    once-only guard: claim_streamer_notification() can never succeed for
    this batch again after this commits.
    """
    now = _now()
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE streamer_batches
                    SET telegram_notified_at = %s,
                        telegram_notify_last_error = NULL,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (now, now, batch_id),
                )
    finally:
        conn.close()


def mark_streamer_notification_failed(batch_id: str, error: str) -> None:
    """
    Called when send_message() returned False (or the send was never
    attempted, e.g. the user unlinked Telegram between claim and send).
    Deliberately does NOT clear telegram_notify_claimed_at -- the claim's
    own lease (_NOTIFY_CLAIM_LEASE_SECONDS) is what naturally rate-limits
    the next retry attempt, so a burst of GET/reconcile calls right after
    a failure can't hammer the Telegram API. `error` must already be a
    short, safe, code-controlled string (e.g. "telegram_send_failed") --
    never str(exception) or anything that could embed a bot token or
    webhook URL; see scripts/telegram_bot.py's own docstring for why.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE streamer_batches
                    SET telegram_notify_last_error = %s,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (error[:500], _now(), batch_id),
                )
    finally:
        conn.close()
