"""
bot_bridge_routes.py
====================
Bridge for @sonya_group_ai (the long-polling bot on 147.45.235.79, owner 2026-11-20: «человек кидает в бота
ссылку → предлагаем по подписке → сразу обрабатывается видео»).

Every endpoint needs X-Sonya-Bot-Secret (= SONYA_BOT_BRIDGE_SECRET, the same secret as POST /api/telegram/bot-link)
and acts for the SONYA account LINKED to body.telegram_user_id (users.telegram_linked). No new billing logic: the
endpoints call the very functions the site calls (from-url, file upload, checkout, payment-status), with the linked
user in place of the session user -- same entitlements, quotas, 2K units, length caps, refunds, idempotency.

  POST /api/telegram/bot/account     {telegram_user_id}                  -> linked?, access per bot mode, preset
  POST /api/telegram/bot/quota       {telegram_user_id}                  -> «Мой лимит»: left / limit / resets (quota_view)
  POST /api/telegram/bot/pricing     {}                                  -> plans from scripts/pricing.py (prices as sold)
  POST /api/telegram/bot/jobs/url    {telegram_user_id, url, clip, quality, trim_start_sec, trim_end_sec, key}
  POST /api/telegram/bot/jobs/file   multipart: telegram_user_id, clip, quality, trim_*, key, file
  POST /api/telegram/bot/ingest      {telegram_user_id, ingest_id}       -> URL download state (the site's poll)
  POST /api/telegram/bot/job         {telegram_user_id, job_id}          -> status, stage, notice, clips
  POST /api/telegram/bot/checkout    {telegram_user_id, plan_id, key}    -> ЮKassa redirect_url (site's checkout)
  POST /api/telegram/bot/payment     {telegram_user_id, invoice_id}      -> paid / pending (site's payment-status)

Off switch: SONYA_BOT_BRIDGE=off -> 404 everywhere (the bot then answers «откройте сайт»).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import (APIRouter, BackgroundTasks, File, Form, HTTPException, Request, Response,
                     UploadFile)

logger = logging.getLogger(__name__)
router = APIRouter()

SITE = os.environ.get("SONYA_PUBLIC_URL", "https://sonya.group").rstrip("/")

# bot mode key -> (API mode, params.clipType). Same mapping as the site's normalizeMode() (auth.js).
CLIPS: Dict[str, tuple] = {
    "viral": ("virality", "viral"),
    "streamer": ("streamer", "streamer"),
    "educational": ("educational", "educational"),
    "story": ("virality", "storytelling"),
    "hooks": ("virality", "hooks"),
    "beep": ("beep", None),
    "trailer": ("virality", "trailer"),          # admins only (the API refuses it for everyone else)
}
HIGH_CLIPS = ("viral", "story", "hooks")
PRESET_DEFAULTS = {"clipType": "", "subtitleStyle": "viral", "quality": "standard", "edu_template": 1,
                   "streamer_layout": 1, "moment_types": [], "max_clips": 0, "hook": False, "hookTitle": True,
                   "jumpCuts": True}
# job status -> stage shown in the bot (the site's processing page shows the same four steps)
STAGES = {
    "queued": "queue", "claimed": "queue", "gpu_requested": "queue", "gpu_booting": "queue",
    "source_wait": "download", "worker_started": "download", "preflight_running": "download",
    "downloading": "download", "model_downloading": "download",
    "mode_running": "analyze", "analyzing": "analyze", "yolo": "analyze", "scripting": "analyze",
    "tts": "edit", "subtitles": "edit", "assembling": "edit", "uploading_result": "edit",
    "completed": "done", "failed": "failed", "cancelled": "failed",
}


def _enabled() -> bool:
    return os.environ.get("SONYA_BOT_BRIDGE", "on").strip().lower() not in ("off", "0", "false", "no")


def _check_secret(request: Request) -> None:
    expected = os.environ.get("SONYA_BOT_BRIDGE_SECRET", "").strip()
    provided = request.headers.get("X-Sonya-Bot-Secret", "").strip()
    if not _enabled():
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


def _tg_id(value: Any) -> int:
    if isinstance(value, bool):
        value = None
    try:
        v = int(str(value).strip())
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail={"error": "invalid_payload"})
    if v <= 0:
        raise HTTPException(status_code=400, detail={"error": "invalid_payload"})
    return v


def find_linked_user(telegram_user_id: int) -> Optional[Dict[str, Any]]:
    """The SONYA user row (same shape as get_current_user's) linked to this Telegram account, or None."""
    from scripts import auth_store
    conn = auth_store._get_conn()
    try:
        row = auth_store._row(conn, "SELECT * FROM users WHERE telegram_user_id = %s AND telegram_linked "
                                    "ORDER BY telegram_linked_at DESC NULLS LAST LIMIT 1", (telegram_user_id,))
    finally:
        conn.close()
    return dict(row) if row else None


def _user_or_401(telegram_user_id: int) -> Dict[str, Any]:
    user = find_linked_user(telegram_user_id)
    if not user:
        raise HTTPException(status_code=401, detail={"error": "not_linked",
                                                     "message": "Telegram не привязан к аккаунту SONYA."})
    return user


async def _json(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid_payload"})
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={"error": "invalid_payload"})
    return body


def _clip(value: Any, user: Dict[str, Any]) -> str:
    from scripts import prod_generation_api as api
    clip = str(value or "").strip().lower()
    if clip not in CLIPS or (clip == "trailer" and not api._is_admin_user(user)):
        raise HTTPException(status_code=400, detail={"error": "invalid_mode", "message": "Неизвестный режим."})
    return clip


def preset_of(user_id: str) -> tuple:
    """(name, full settings) of the user's default preset from the «Пресеты» app; ("", defaults) when none."""
    try:
        from scripts import presets_routes
        doc = presets_routes.load(user_id)
    except Exception as exc:  # noqa: BLE001 - no preset = the standard look
        logger.warning("[bot] preset_load_failed user_id=%s: %s", user_id, type(exc).__name__)
        doc = {}
    p = next((x for x in doc.get("presets") or [] if x.get("id") == doc.get("default_id")), None)
    s = dict(PRESET_DEFAULTS)
    if p:
        s.update({k: v for k, v in (p.get("settings") or {}).items() if k in PRESET_DEFAULTS})
    return (str(p.get("name") or "") if p else ""), s


def build_params(clip: str, url: Optional[str], platform: str, preset: tuple, quality: str,
                 trim_start: Optional[float], trim_end: Optional[float], file_name: Optional[str] = None) -> Dict[str, Any]:
    """Job params exactly as the site's collectFormData() + presetJob*() (app.js) / the BEEP app build them."""
    name, s = preset
    source = ({"mode": "url", "url": url, "platform": platform} if url else
              {"mode": "upload", "url": None, "fileName": file_name or "telegram_video.mp4", "platform": "upload"})
    trim: Dict[str, Any] = {}
    if trim_end:
        st = max(0.0, float(trim_start or 0))
        trim = {"trim_start_sec": st, "trim_end_sec": float(trim_end)}
        if not st:
            trim["trim_first_sec"] = float(trim_end)
    if clip == "beep":
        return {"beep": {"effect": "beep", "categories": ["mat"]}, "source": source, "brand": "sonya",
                "frontend": "telegram_bot", **trim}
    _mode, clip_type = CLIPS[clip]
    sub = s.get("subtitleStyle") or "viral"
    if sub == "gradient" and clip not in HIGH_CLIPS + ("streamer",):
        sub = "viral"                      # the colour choice exists only for Вирал/Сторителлинг/Хуки (+ Стример)
    settings: Dict[str, Any] = {"subtitleStyle": sub, "voiceoverEnabled": False, "voiceType": "male-1",
                                "translation": "none",
                                "quality": "high" if quality == "high" and clip in HIGH_CLIPS else "standard"}
    if clip == "educational":
        settings["edu_template"] = int(s.get("edu_template") or 1)
    if clip == "streamer":
        settings["streamer_layout"] = int(s.get("streamer_layout") or 1)
        settings["moment_types"] = list(s.get("moment_types") or [])
        settings["hookTitle"] = bool(s.get("hookTitle"))
        settings["jumpCuts"] = bool(s.get("jumpCuts"))
    if name:
        settings["preset"] = name
    out: Dict[str, Any] = {"source": source, "brand": "sonya", "clipType": clip_type, "settings": settings,
                           "frontend": "telegram_bot", **trim}
    if s.get("max_clips"):
        out["max_clips"] = int(s["max_clips"])
    if s.get("hook") and clip in ("viral", "story"):
        out["hook"] = 1
    return out


def _access(user: Dict[str, Any], clip: str) -> Dict[str, Any]:
    """May this user start this bot mode now, what is left, the length cap. Nothing is debited."""
    from scripts import entitlements, prod_generation_api as api
    mode = CLIPS[clip][0]
    out: Dict[str, Any] = {"allowed": True, "left": None, "kind": None, "code": None, "message": None}
    try:
        ent = api._resolve_entitlement(user, mode, "bot", {"clipType": CLIPS[clip][1] or ""})
    except HTTPException as exc:
        d = exc.detail if isinstance(exc.detail, dict) else {}
        out.update(allowed=False, code=d.get("code") or d.get("error"), message=d.get("message"),
                   plan_mode=d.get("plan_mode"))
        ent = entitlements.Entitlement(kind="free", plan_mode=None)
    else:
        out["kind"] = ent.kind
        if api._is_admin_user(user) or ent.kind == "legacy_pro":
            out["left"] = None
        elif ent.kind == "subscription":
            for sub in entitlements.get_user_subscriptions(str(user["id"])):
                if str(sub["id"]) == str(ent.subscription_id):
                    out["left"] = max(0, int(sub["ops_limit"]) - int(sub["ops_used"]))
                    out["plan_id"] = sub["plan_id"]
        elif ent.kind == "pack":
            out["left"] = api.pack_status(str(user["id"]), mode)["left"]
        else:
            out["left"] = max(0, int(user.get("free_video_limit") or 0) - int(user.get("free_video_used") or 0))
    cap = api._source_cap_sec(user, ent, mode)
    out["max_source_minutes"] = int(cap // 60)
    return out


def _quality_on() -> bool:
    return os.environ.get("SONYA_QUALITY_HIGH", "on").strip().lower() not in ("off", "0", "false", "no")


# ── account ───────────────────────────────────────────────────────────────────

@router.post("/api/telegram/bot/account")
async def bot_account(request: Request):
    _check_secret(request)
    body = await _json(request)
    tg = _tg_id(body.get("telegram_user_id"))
    user = find_linked_user(tg)
    if not user:
        return {"linked": False, "connect_url": SITE + "/?tg=connect"}
    from scripts import prod_generation_api as api, quality
    admin = api._is_admin_user(user)
    clips = [c for c in CLIPS if c != "trailer" or admin]
    name, s = preset_of(str(user["id"]))
    return {
        "linked": True,
        "is_admin": admin,
        "email_hint": _email_hint(user.get("email")),
        "access": {c: _access(user, c) for c in clips},
        "preset": {"name": name or "Стандарт SONYA", "quality": s.get("quality"), "clipType": s.get("clipType")},
        "quality": {"available": _quality_on(), "clips": list(HIGH_CLIPS),
                    "units": 1 if admin else quality.HIGH_UNITS},
        "urls": {"presets": SITE + "/#/presets", "works": SITE + "/#/works", "tariffs": SITE + "/?paywall=1",
                 "site": SITE + "/"},
    }



@router.post("/api/telegram/bot/quota")
async def bot_quota(request: Request):
    """«Мой лимит» (2026-11-22): the same numbers and words as the site's glass card (scripts/quota_view.py)."""
    _check_secret(request)
    body = await _json(request)
    tg = _tg_id(body.get("telegram_user_id"))
    user = find_linked_user(tg)
    if not user:
        return {"linked": False, "connect_url": SITE + "/?tg=connect"}
    from scripts import quota_routes, quota_view
    q = quota_routes.quota_for_user(user, "virality")
    others = [o for o in q.get("others") or []]
    return {"linked": True, "quota": q, "lines": quota_view.text_lines(q),
            "others": [{"plan_label": o["plan_label"], "left": o["left"], "limit": o["limit"],
                        "resets_on": quota_view.short_date(o["resets_at"])} for o in others],
            "urls": {"tariffs": SITE + "/?paywall=1", "site": SITE + "/"}}

def _email_hint(email: Any) -> str:
    e = str(email or "")
    if "@" not in e:
        return ""
    local, dom = e.split("@", 1)
    return (local[:2] + "•••" if len(local) > 2 else local[:1] + "•••") + "@" + dom


# ── pricing ───────────────────────────────────────────────────────────────────

@router.post("/api/telegram/bot/pricing")
async def bot_pricing(request: Request):
    _check_secret(request)
    from scripts import feature_flags, pricing
    plans = []
    for p in pricing.PLAN_CATALOG.values():
        if feature_flags.plan_disabled(p):
            continue
        plans.append({"plan_id": p.plan_id, "mode": p.plan_mode, "kind": p.plan_type,
                      "price": int(p.amount) if p.amount == int(p.amount) else float(p.amount),
                      "ops": p.ops_limit, "max_source_minutes": p.max_source_sec // 60,
                      "days": p.duration_days, "title": p.description})
    return {"plans": plans}


# ── jobs ──────────────────────────────────────────────────────────────────────

def _trim(body: Dict[str, Any]) -> tuple:
    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f >= 0 else None
    st, en = num(body.get("trim_start_sec")), num(body.get("trim_end_sec"))
    if en is not None and st is not None and en <= st:
        raise HTTPException(status_code=422, detail={"error": "trim_range", "code": "SOURCE_TRIM_RANGE",
                                                     "message": "Начало отрезка позже конца. Выберите отрезок заново."})
    return st, en


def _key(value: Any) -> Optional[str]:
    k = str(value or "").strip()
    return ("tg-" + k)[:120] if k else None


@router.post("/api/telegram/bot/jobs/url", status_code=202)
async def bot_job_from_url(request: Request, background_tasks: BackgroundTasks):
    _check_secret(request)
    body = await _json(request)
    user = _user_or_401(_tg_id(body.get("telegram_user_id")))
    clip = _clip(body.get("clip"), user)
    url = str(body.get("url") or "").strip()
    from scripts import prod_generation_api as api, url_ingest
    st, en = _trim(body)
    params = build_params(clip, url, url_ingest.detect_platform(url), preset_of(str(user["id"])),
                          str(body.get("quality") or "standard"), st, en)
    req = api.UrlIngestRequest(url=url, mode=CLIPS[clip][0], params=params)
    res = await api.create_generation_job_from_url(
        request=request, background_tasks=background_tasks, body=req,
        idempotency_key_header=_key(body.get("key")), user=user, _origin=None, _rl=None)
    logger.info("[bot] job_from_url user_id=%s clip=%s ingest_id=%s", user["id"], clip, res.get("ingest_id"))
    return res


@router.post("/api/telegram/bot/jobs/file", status_code=202)
async def bot_job_from_file(
    request: Request,
    response: Response,
    telegram_user_id: str = Form(...),
    clip: str = Form(...),
    quality: str = Form("standard"),
    trim_start_sec: Optional[str] = Form(None),
    trim_end_sec: Optional[str] = Form(None),
    key: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    _check_secret(request)
    user = _user_or_401(_tg_id(telegram_user_id))
    c = _clip(clip, user)
    from scripts import prod_generation_api as api
    st, en = _trim({"trim_start_sec": trim_start_sec, "trim_end_sec": trim_end_sec})
    params = build_params(c, None, "upload", preset_of(str(user["id"])), quality, st, en,
                          file_name=file.filename)
    res = await api.create_generation_job(
        request=request, response=response, mode=CLIPS[c][0], file=file, params=json.dumps(params),
        idempotency_key_header=_key(key), user=user, _origin=None, _rl=None)
    logger.info("[bot] job_from_file user_id=%s clip=%s job_id=%s", user["id"], c, res.job_id)
    return {"job_id": res.job_id, "status": res.status, "mode": res.mode}


@router.post("/api/telegram/bot/ingest")
async def bot_ingest(request: Request):
    _check_secret(request)
    body = await _json(request)
    user = _user_or_401(_tg_id(body.get("telegram_user_id")))
    from scripts import prod_generation_api as api
    return await api.get_url_ingest_status(ingest_id=str(body.get("ingest_id") or ""), user=user, _rl=None)


@router.post("/api/telegram/bot/job")
async def bot_job(request: Request):
    _check_secret(request)
    body = await _json(request)
    user = _user_or_401(_tg_id(body.get("telegram_user_id")))
    from scripts import prod_generation_api as api
    job_id = str(body.get("job_id") or "")
    job = api.get_job(job_id) if job_id else None
    if not job or str(job.get("user_id")) != str(user["id"]):
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    job = api._with_notice(job)
    st = str(job.get("status") or "")
    clips = 0
    if st == "completed":
        try:
            clips = sum(1 for f in api.list_job_files(job_id) if f.get("file_type") == "output"
                        and str(f.get("s3_key") or "").lower().endswith(".mp4"))
        except Exception:  # noqa: BLE001
            clips = int(job.get("clip_count") or 0)
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    return {"job_id": job_id, "status": st, "stage": STAGES.get(st, "analyze"), "notice": job.get("notice"),
            "mode": job.get("mode"), "clip_type": params.get("clipType"), "clips": clips or int(job.get("clip_count") or 0),
            "error": job.get("error") if st in ("failed", "cancelled") else None,
            "error_code": job.get("last_error") if st in ("failed", "cancelled") else None,
            "refunded": bool(params.get("quota_refunded")),
            **(_delivery(job_id) if st == "completed" else {}),
            "works_url": SITE + "/#/works"}


def _delivery(job_id: str) -> Dict[str, Any]:
    """Clips sonya-tg-deliver (same host) already sent to Telegram: its sqlite state, read-only. None = unknown."""
    import sqlite3
    path = os.environ.get("SONYA_TG_STATE", "/var/lib/sonya-tg/state.db")
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        try:
            ok = c.execute("SELECT count(*) FROM sent WHERE job_id=? AND status='ok'", (job_id,)).fetchone()[0]
            bad = c.execute("SELECT count(*) FROM sent WHERE job_id=? AND status!='ok' AND attempts>=5",
                            (job_id,)).fetchone()[0]
        finally:
            c.close()
        return {"delivered": int(ok), "delivery_failed": int(bad)}
    except Exception:  # noqa: BLE001 - no state file / locked: the bot waits with a timeout
        return {"delivered": None, "delivery_failed": 0}


# ── payments (the site's ЮKassa checkout, unchanged) ─────────────────────────

@router.post("/api/telegram/bot/checkout")
async def bot_checkout(request: Request):
    _check_secret(request)
    body = await _json(request)
    user = _user_or_401(_tg_id(body.get("telegram_user_id")))
    from scripts import payment_routes
    res = await payment_routes.checkout(
        body=payment_routes.CheckoutRequest(plan_id=str(body.get("plan_id") or "")), request=request,
        user=user, _origin=None, idempotency_key_header=_key(body.get("key")))
    logger.info("[bot] checkout user_id=%s plan_id=%s invoice=%s", user["id"], body.get("plan_id"),
                res.get("invoice_id") if isinstance(res, dict) else None)
    return res


@router.post("/api/telegram/bot/payment")
async def bot_payment(request: Request):
    _check_secret(request)
    body = await _json(request)
    user = _user_or_401(_tg_id(body.get("telegram_user_id")))
    from scripts import payment_routes
    try:
        invoice_id = int(body.get("invoice_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail={"error": "invalid_payload"})
    return await payment_routes.payment_status(invoice_id=invoice_id, provider_return=None, user=user)
