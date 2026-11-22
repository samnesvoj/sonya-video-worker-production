"""
midroll_routes.py
=================
«Мидролл» (2026-11-22, clients: «не только баннер, а ещё и мидролл»): the user's own ad video, inserted by the worker
(scripts/shared/midroll.py) inside every clip. The ad is uploaded once here and saved in the user's preset (Пресеты →
Эффекты → Реклама), so it is not uploaded again for every cut.

  POST /api/midroll/upload   multipart file (any video, 1-30 s, <= 100 MB) -> {key, name, dur, w, h, audio, video_url, poster_url}
  GET  /api/midroll/url?key= -> {video_url, poster_url} (fresh presigned links for the preview; own keys only)

Storage: S3 users/<user_id>/midroll/<sha256[:16]><ext> (+ _poster.jpg). Same bytes = same key (no duplicates).
Job params: params.midroll (the site sends it from the default preset; jobs created elsewhere - bot, Mini App
links - get the default preset's mid-roll at claim time, prod_generation_api._with_midroll). Only keys of the job's
own user reach a worker (with_midroll()).
Nothing here touches quotas, payments or jobs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from scripts.rate_limiter import RateLimiter
from scripts.security import get_current_user, new_trace_id, safe_error, verify_browser_origin

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_MB = int(os.environ.get("SONYA_MIDROLL_MAX_MB", "100"))
MIN_S, MAX_S = 1.0, 30.0
EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
PLACES = ("mid", "pre", "post")
ATS = ("auto", "after")
FITS = ("auto", "blur", "fit", "fill")
_KEY = re.compile(r"^users/([0-9A-Za-z_-]{1,64})/midroll/([0-9a-f]{16})\.(mp4|mov|m4v|webm|mkv)$")

_limiter = RateLimiter(key_prefix="midroll_upload", limit=30, window_seconds=3600, key_by="session")


def enabled() -> bool:
    return str(os.environ.get("SONYA_MIDROLL", "")).strip().lower() not in ("0", "off", "false", "no")


def own_key(key: Any, user_id: str) -> bool:
    m = _KEY.match(str(key or ""))
    return bool(m) and m.group(1) == str(user_id)


def _num(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else max(lo, min(hi, f))


def clean(raw: Any, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """A preset's / job's midroll settings -> only known fields; None when there is no valid ad of this user.
    user_id None = shape check only (a guest preset in the browser is never sent here)."""
    if not isinstance(raw, dict):
        return None
    key = str(raw.get("key") or "")
    if not _KEY.match(key) or (user_id is not None and not own_key(key, user_id)):
        return None
    place = raw.get("place") if raw.get("place") in PLACES else "mid"
    at = raw.get("at") if raw.get("at") in ATS else "auto"
    fit = raw.get("fit") if raw.get("fit") in FITS else "auto"
    every = 2 if str(raw.get("every")) == "2" else 1
    mt = int(_num(raw.get("max_total"), 0, 600, 0))
    return {
        "on": raw.get("on") is not False,
        "key": key,
        "name": re.sub(r"[\x00-\x1f<>]", "", str(raw.get("name") or ""))[:80],
        "dur": round(_num(raw.get("dur"), 0, MAX_S + 1, 0), 2),
        "w": int(_num(raw.get("w"), 0, 10000, 0)),
        "h": int(_num(raw.get("h"), 0, 10000, 0)),
        "audio": bool(raw.get("audio", True)),
        "place": place, "at": at, "after_s": int(_num(raw.get("after_s"), 3, 120, 20)),
        "every": every, "fit": fit, "max_total": mt if mt in (0, 60, 90) else 0,
    }


def job_value(raw: Any, user_id: str) -> Optional[Dict[str, Any]]:
    """What the worker gets: the cleaned settings of this user's ad when switched on, else None."""
    c = clean(raw, user_id)
    if not c or not c["on"]:
        return None
    return {k: c[k] for k in ("key", "place", "at", "after_s", "every", "fit", "max_total", "dur")}


def with_midroll(job: dict, default_preset_loader=None) -> dict:
    """Claim time (prod_generation_api.worker_claim): params.midroll of the job checked against its user (another
    user's key is dropped); no params.midroll at all (bot / Mini App links) -> the user's default preset's mid-roll.
    Never raises, never writes the DB."""
    try:
        params = job.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        if not isinstance(params, dict):
            return job
        if not enabled():                                  # ops switch: no ad reaches any worker
            return dict(job, params=dict(params, midroll=None)) if params.get("midroll") else job
        uid = str(job.get("user_id") or "")
        if "midroll" in params:
            val = job_value(params.get("midroll"), uid) if params.get("midroll") else None
            if params.get("midroll") and val is None:
                logger.warning("[midroll] job_id=%s: params.midroll dropped (not this user's ad / bad value)", job.get("id"))
        else:
            if default_preset_loader is None:
                from scripts.presets_routes import load as default_preset_loader
            doc = default_preset_loader(uid) or {}
            pre = next((p for p in doc.get("presets") or [] if p.get("id") == doc.get("default_id")), None)
            val = job_value(((pre or {}).get("settings") or {}).get("midroll"), uid)
            if val is None:
                return job
        job = dict(job, params=dict(params, midroll=val))
        if val:
            logger.info("[midroll] job_id=%s place=%s every=%s", job.get("id"), val["place"], val["every"])
    except Exception as exc:   # never block a claim over the ad
        logger.warning("[midroll] skipped job_id=%s: %s", job.get("id"), type(exc).__name__)
    return job


# ── upload ────────────────────────────────────────────────────────────────────

def probe(path: str) -> Optional[Dict[str, Any]]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "format=duration:stream=codec_type,width,height:stream_side_data=rotation", "-of", "json", path],
                             capture_output=True, text=True, timeout=30)
        j = json.loads(out.stdout or "{}")
    except Exception:
        return None
    v = next((s for s in j.get("streams") or [] if s.get("codec_type") == "video"), None)
    if not v:
        return None
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd and abs(int(float(sd["rotation"]))) % 180 == 90:
            w, h = h, w
    try:
        dur = float((j.get("format") or {}).get("duration") or 0)
    except ValueError:
        dur = 0.0
    return {"dur": round(dur, 2), "w": w, "h": h,
            "audio": any(s.get("codec_type") == "audio" for s in j.get("streams") or [])}


def _poster(src: str, dur: float, dst: str) -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(1.0, dur / 2):.2f}", "-i", src,
                            "-frames:v", "1", "-vf", "scale='min(540,iw)':-2", "-q:v", "4", dst],
                           capture_output=True, timeout=30)
        return r.returncode == 0 and Path(dst).is_file()
    except Exception:
        return False


def _urls(key: str) -> Dict[str, Optional[str]]:
    from scripts.prod_s3_storage import generate_presigned_get_url, object_exists
    poster = key.rsplit(".", 1)[0] + "_poster.jpg"
    return {"video_url": generate_presigned_get_url(key, expires_in=6 * 3600),
            "poster_url": generate_presigned_get_url(poster, expires_in=6 * 3600) if object_exists(poster) else None}


def _err(code: str, msg: str, trace_id: str, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": msg, "trace_id": trace_id})


@router.post("/api/midroll/upload")
async def upload_midroll(request: Request, file: UploadFile = File(...), user: dict = Depends(get_current_user),
                         _o: None = Depends(verify_browser_origin), _rl: None = Depends(_limiter)):
    trace_id = new_trace_id()
    if not enabled():
        raise _err("midroll_off", "Мидролл временно выключен.", trace_id, 503)
    from scripts.upload_security import validate_upload
    content, safe_name = await validate_upload(file, max_size_bytes=MAX_MB * 1024 * 1024)
    ext = Path(safe_name).suffix.lower()
    if ext not in EXTS:
        ext = ".mp4" if ext in (".mpeg", ".mpg", ".3gp", ".avi") else ext
    if ext not in EXTS:
        raise _err("unsupported_format", "Нужен видеофайл: MP4, MOV, WEBM или MKV.", trace_id)
    uid = str(user["id"])
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "ad" + ext)
        with open(src, "wb") as fh:
            fh.write(content)
        info = probe(src)
        if not info or info["w"] < 64 or info["h"] < 64:
            raise _err("invalid_video", "Не удалось прочитать видео. Попробуйте другой файл.", trace_id)
        if info["dur"] < MIN_S:
            raise _err("too_short", "Ролик слишком короткий: нужно от 1 секунды.", trace_id)
        if info["dur"] > MAX_S + 0.5:
            raise _err("too_long", f"Ролик длится {info['dur']:.0f} с, а максимум — 30 с. Обычно хватает 3–15 с.", trace_id)
        key = f"users/{uid}/midroll/{hashlib.sha256(content).hexdigest()[:16]}{ext}"
        poster = os.path.join(td, "poster.jpg")
        try:
            from scripts.prod_s3_storage import object_exists, upload_bytes, upload_file
            if not object_exists(key):
                upload_bytes(content, key, content_type="video/quicktime" if ext == ".mov" else f"video/{ext[1:]}")
            if _poster(src, info["dur"], poster):
                upload_file(Path(poster), key.rsplit(".", 1)[0] + "_poster.jpg", content_type="image/jpeg")
            urls = _urls(key)
        except Exception as exc:
            logger.error("[midroll] upload_failed trace_id=%s: %s", trace_id, type(exc).__name__)
            raise safe_error("storage_error", 500, trace_id)
    name = re.sub(r"[\x00-\x1f<>]", "", Path(file.filename or safe_name).name)[:80]
    logger.info("[midroll] uploaded user=%s key=%s dur=%.1f %dx%d audio=%s", uid, key, info["dur"], info["w"], info["h"], info["audio"])
    return dict(key=key, name=name, **info, **urls)


@router.get("/api/midroll/url")
async def midroll_url(key: str, user: dict = Depends(get_current_user)):
    trace_id = new_trace_id()
    if not own_key(key, str(user["id"])):
        raise _err("not_found", "Ролик не найден.", trace_id, 404)
    try:
        return _urls(key)
    except Exception as exc:
        logger.error("[midroll] url_failed trace_id=%s: %s", trace_id, type(exc).__name__)
        raise safe_error("storage_error", 500, trace_id)
