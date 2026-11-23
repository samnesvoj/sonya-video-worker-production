"""
quota_routes.py
===============
Limits per subscription (owner 2026-11-22: «купил подписку — и пусть SONYA считает, сколько осталось видео»).

  GET  /api/me/quota?mode=virality           -> scripts/quota_view.build(): {plan, plan_label, kind, used, limit,
                                                left, resets_at, unlimited, cost{standard,high}, action, ...}
  POST /api/generation/jobs/{job_id}/cancel  -> cancel the user's OWN job while it has not reached a GPU yet
                                                (status queued / source_wait); the quota comes back at once
                                                (migration 025 trigger, same as any failed job).

Nothing here changes prices, plans or the debit itself (prod_job_store.create_job_with_quota).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from scripts.rate_limiter import RateLimiter
from scripts.security import assert_job_owner, get_current_user, new_trace_id, safe_error, verify_browser_origin

logger = logging.getLogger(__name__)
router = APIRouter()

_read_limiter = RateLimiter(key_prefix="quota", limit=120, window_seconds=60, key_by="session")
_cancel_limiter = RateLimiter(key_prefix="job_cancel", limit=30, window_seconds=600, key_by="session")

CANCELLABLE = ("queued", "source_wait")      # before any GPU work: dispatcher has not claimed it yet


def quota_for_user(user: dict, mode: str = "virality", clip_type: str = "") -> dict:
    from scripts import prod_job_store, quota_view
    job_mode = quota_view.job_mode_of(mode)
    pack = prod_job_store.pack_status(str(user["id"]), "beep") if job_mode == "beep" else None
    is_admin = str(user.get("plan_type") or "").lower().strip() == "admin"   # = prod_generation_api._is_admin_user
    return quota_view.build(user, job_mode, is_admin=is_admin, pack=pack,
                            clip_type=clip_type or quota_view.clip_of(mode))


@router.get("/api/me/quota")
async def get_my_quota(mode: str = "virality", clip: str = "", user: dict = Depends(get_current_user),
                       _rl: None = Depends(_read_limiter)):
    trace_id = new_trace_id()
    try:
        return quota_for_user(user, mode, clip[:32])
    except Exception as exc:
        logger.warning("[quota] view_failed user_id=%s error_type=%s", user.get("id"), type(exc).__name__)
        raise safe_error("db_error", 500, trace_id)


def cancel_before_gpu(job_id: str, user_id: str) -> bool:
    """Atomic: only the owner's job, only from a pre-GPU status. True = cancelled now."""
    from scripts.prod_job_store import _get_conn
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE generation_jobs SET status='cancelled', last_error='USER_CANCELLED', "
                    "error='Отменено до начала обработки', updated_at=NOW() "
                    "WHERE id=%s AND user_id=%s AND status = ANY(%s) RETURNING id",
                    (job_id, str(user_id), list(CANCELLABLE)),
                )
                return cur.fetchone() is not None
    finally:
        conn.close()


@router.post("/api/generation/jobs/{job_id}/cancel")
async def cancel_my_job(request: Request, job_id: str, user: dict = Depends(get_current_user),
                        _origin: None = Depends(verify_browser_origin), _rl: None = Depends(_cancel_limiter)):
    from scripts.prod_job_store import get_job
    user_id = str(user["id"])
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "not_found", "trace_id": new_trace_id()})
    assert_job_owner(job, user_id)
    if not cancel_before_gpu(job_id, user_id):
        raise HTTPException(status_code=409, detail={
            "error": "job_started", "trace_id": new_trace_id(),
            "message": "Обработка уже началась — отменить нельзя."})
    logger.info("[quota] job_cancelled_before_gpu job_id=%s user_id=%s", job_id, user_id)
    from scripts import auth_store
    fresh = auth_store.get_user_by_id(user_id) or user          # free_video_used just changed (refund)
    mode = str(job.get("mode") or "virality")
    clip = entitlements_clip(job.get("params"))
    return {"ok": True, "id": job_id, "status": "cancelled", "quota": quota_for_user(fresh, mode, clip)}


def entitlements_clip(params) -> str:
    from scripts import entitlements
    return entitlements.clip_type_of(params if isinstance(params, dict) else {})
