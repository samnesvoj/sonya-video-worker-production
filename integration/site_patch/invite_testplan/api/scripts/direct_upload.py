"""
direct_upload.py -- browser -> S3 direct upload (presigned multipart), 2026-11-16.

Why: a file upload went browser -> nginx (500 MB cap) -> API, which read the whole
file into memory (VPS has 2 GB RAM) and the browser gave up after ~20-40 s on a
phone (nginx 499, «Сервер недоступен»). Now the bytes never touch the VPS:

  POST /api/generation/uploads/init      quota/entitlement/size/extension checks -> S3 multipart upload + token
  POST /api/generation/uploads/parts     presigned PUT URLs for the given part numbers
  POST /api/generation/uploads/status    parts already in S3 (resume after a lost network / app in background)
  POST /api/generation/uploads/complete  S3 complete -> magic bytes (first 64 bytes) -> ffprobe over a presigned
                                         GET URL (length cap / segment cut, on disk, never in RAM) -> job
  POST /api/generation/uploads/abort     drop the multipart upload

The token is an HMAC-signed, self-contained session (user, job id, S3 key, upload id, size, part size) so a
restart of the API never loses an upload. Job creation after complete is the same create_job_with_quota()
transaction as POST /api/generation/jobs (idempotency, quota, plan debit).

Off switch: SONYA_DIRECT_UPLOAD=off (init answers 503 DIRECT_UPLOAD_OFF; the browser uses the old path).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from scripts.rate_limiter import RateLimiter
from scripts.security import get_current_user, new_trace_id, safe_error, verify_browser_origin
from scripts import upload_security as _us

logger = logging.getLogger(__name__)

router = APIRouter()

MB = 1024 * 1024
GB = 1024 * MB
_PART_SMALL = 8 * MB           # files < 1 GB: 8 MB parts (a lost part on a phone costs little)
_PART_LARGE = 16 * MB          # bigger files: 16 MB parts (20 GB = 1280 parts, S3 limit is 10000)
_URL_TTL_SEC = 3 * 3600        # presigned PUT lifetime; the browser asks again on 403
_TOKEN_TTL_SEC = 3 * 24 * 3600  # resume window (an unfinished multipart is dropped by the bucket lifecycle after 3 days)
_MAX_PARTS_PER_CALL = 64

_init_limiter = RateLimiter(key_prefix="dupload", limit=30, window_seconds=3600, key_by="session")
_parts_limiter = RateLimiter(key_prefix="duparts", limit=600, window_seconds=60, key_by="session")


def _enabled() -> bool:
    return os.environ.get("SONYA_DIRECT_UPLOAD", "on").strip().lower() not in ("0", "off", "false", "no")


def _hard_max_bytes() -> int:
    return int(os.environ.get("DIRECT_UPLOAD_MAX_MB", "20480")) * MB


def _api():
    from scripts import prod_generation_api as api   # lazy: that module includes this router
    return api


# ── S3 ────────────────────────────────────────────────────────────────────────
_s3 = None


def _client():
    global _s3
    if _s3 is None:
        from scripts.prod_s3_storage import _client as mk
        _s3 = mk()
    return _s3


def _bucket() -> str:
    from scripts.prod_s3_storage import _bucket as b
    return b()


# ── Token ─────────────────────────────────────────────────────────────────────

def _secret() -> bytes:
    base = os.environ.get("AUTH_SECRET") or os.environ.get("S3_SECRET_ACCESS_KEY") or ""
    if not base:
        raise RuntimeError("AUTH_SECRET not set")
    return hmac.new(base.encode(), b"sonya-direct-upload-v1", hashlib.sha256).digest()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(payload: Dict[str, Any]) -> str:
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _err(code: int, error: str, message: str, trace_id: str, **extra) -> HTTPException:
    return HTTPException(status_code=code, detail={"error": error, "code": error.upper(), "message": message,
                                                   "trace_id": trace_id, **extra})


def read_token(token: str, user_id: str, trace_id: str) -> Dict[str, Any]:
    try:
        body, sig = str(token or "").split(".", 1)
        good = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise ValueError("bad signature")
        data = json.loads(_unb64(body))
    except Exception:
        raise _err(400, "upload_token_invalid", "Сессия загрузки не найдена. Загрузите файл заново.", trace_id)
    if str(data.get("uid")) != str(user_id):
        raise _err(403, "upload_token_invalid", "Сессия загрузки принадлежит другому аккаунту.", trace_id)
    if float(data.get("exp") or 0) < time.time():
        raise _err(410, "upload_expired", "Сессия загрузки устарела. Загрузите файл заново.", trace_id)
    return data


# ── Limits ────────────────────────────────────────────────────────────────────

def part_size_for(size: int) -> int:
    ps = _PART_SMALL if size < GB else _PART_LARGE
    while math.ceil(size / ps) > 10000:
        ps *= 2
    return ps


def max_bytes_for(cap_sec: int, mode: str, trimming: bool) -> int:
    """Streamer and a chosen segment: up to the hard ceiling (20 GB). Other modes scale with the tariff
    length cap at ~3 MB/s of source (a 60-min phone video in 4K is ~10 GB), never below 2 GB."""
    hard = _hard_max_bytes()
    if mode == "streamer" or trimming:
        return hard
    return min(hard, max(2 * GB, int(cap_sec) * 3_000_000))


def _gb(n: int) -> str:
    v = n / GB
    return (f"{v:.1f}".rstrip("0").rstrip(".") if v < 10 else f"{v:.0f}").replace(".", ",")


# ── Models ────────────────────────────────────────────────────────────────────

class InitRequest(BaseModel):
    mode: str
    filename: str = Field(..., min_length=1, max_length=400)
    size: int = Field(..., gt=0)
    content_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class PartsRequest(BaseModel):
    token: str
    parts: List[int]


class TokenRequest(BaseModel):
    token: str


class CompleteRequest(BaseModel):
    token: str
    params: Optional[Dict[str, Any]] = None


# ── Shared checks ─────────────────────────────────────────────────────────────

def _check_mode(api, mode: str, request: Request, user_id: str, trace_id: str) -> None:
    if mode not in api.ALLOWED_MODES:
        raise HTTPException(status_code=400, detail={"error": "invalid_mode", "allowed": sorted(api.ALLOWED_MODES),
                                                     "trace_id": trace_id})
    if mode == "sonya_gen":
        raise HTTPException(status_code=400, detail={"error": "use_gen_endpoint", "message": "POST /api/gen/image",
                                                     "trace_id": trace_id})
    api._reject_if_mode_disabled(mode, request, user_id, trace_id)


def _check_name(filename: str, content_type: Optional[str], trace_id: str) -> str:
    if _us._DOUBLE_EXT.search(filename):
        raise _err(400, "invalid_filename", "Недопустимое имя файла. Переименуйте файл и попробуйте снова.", trace_id)
    ext = Path(filename).suffix.lower()
    if ext not in _us.ALLOWED_EXTENSIONS:
        raise _err(400, "unsupported_format",
                   "Этот формат не поддерживается. Подойдут MP4, MOV, MKV, WEBM, AVI, MPEG, 3GP.", trace_id,
                   allowed=sorted(_us.ALLOWED_EXTENSIONS))
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct and ct not in _us.ALLOWED_MIME_TYPES and ct != "application/octet-stream":
        raise _err(400, "unsupported_content_type", "Это не видеофайл. Выберите видео.", trace_id)
    return _us._safe_filename(filename)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/generation/uploads/init")
async def init_upload(body: InitRequest, request: Request, user: dict = Depends(get_current_user),
                      _o: None = Depends(verify_browser_origin), _rl: None = Depends(_init_limiter)):
    trace_id = new_trace_id()
    if not _enabled():
        raise _err(503, "direct_upload_off", "Прямая загрузка выключена.", trace_id)
    api = _api()
    user_id = str(user["id"])
    mode = body.mode
    _check_mode(api, mode, request, user_id, trace_id)
    api._reject_trailer_clip(user, body.params or {}, trace_id)
    safe_name = _check_name(body.filename, body.content_type, trace_id)
    api.check_user_quota(user_id)
    ent = api._resolve_entitlement(user, mode, trace_id, body.params or {})   # refusals (paywall codes) before any upload
    api._apply_quality(user, ent, mode, dict(body.params or {}), trace_id)   # 2K: enough units left? (before upload)
    cap = api._source_cap_sec(user, ent, mode)
    trimming = api._trim_request(body.params or {}, cap)[1] > 0
    limit = max_bytes_for(cap, mode, trimming)
    if body.size > limit:
        api.audit(api.EVT_UPLOAD_REJECTED, user_id=user_id, trace_id=trace_id,
                  details={"mode": mode, "reason": "file_too_large", "size": body.size, "path": "direct"},
                  ip_address=request.client.host if request.client else None)
        raise _err(413, "file_too_large",
                   f"Файл слишком большой ({_gb(body.size)} ГБ). Максимум — {_gb(limit)} ГБ. "
                   "Сожмите видео или вставьте ссылку на него.", trace_id, max_mb=limit // MB)

    ext = Path(safe_name).suffix.lower() or ".mp4"
    job_id = str(uuid.uuid4())
    key = api.build_input_key(user_id=user_id, job_id=job_id, mode=mode, ext=ext)
    ct = (body.content_type or "").split(";")[0].strip().lower()
    if ct not in _us.ALLOWED_MIME_TYPES:
        ct = "video/mp4"
    try:
        up = await run_in_threadpool(lambda: _client().create_multipart_upload(Bucket=_bucket(), Key=key, ContentType=ct))
    except Exception as exc:
        logger.error("[direct_upload] create_multipart_failed trace_id=%s: %s", trace_id, exc)
        raise safe_error("storage_error", 503, trace_id)
    ps = part_size_for(body.size)
    n = math.ceil(body.size / ps)
    tok = make_token({"v": 1, "uid": user_id, "jid": job_id, "key": key, "up": up["UploadId"], "mode": mode,
                      "size": body.size, "ps": ps, "n": n, "name": safe_name, "ct": ct,
                      "exp": int(time.time()) + _TOKEN_TTL_SEC})
    logger.info("[direct_upload] init job_id=%s user_id=%s mode=%s size=%d parts=%dx%d trace_id=%s",
                job_id, user_id, mode, body.size, n, ps, trace_id)
    return {"token": tok, "part_size": ps, "part_count": n, "max_bytes": limit}


@router.post("/api/generation/uploads/parts")
async def part_urls(body: PartsRequest, user: dict = Depends(get_current_user),
                    _o: None = Depends(verify_browser_origin), _rl: None = Depends(_parts_limiter)):
    trace_id = new_trace_id()
    t = read_token(body.token, str(user["id"]), trace_id)
    nums = sorted({int(p) for p in body.parts})[:_MAX_PARTS_PER_CALL]
    if not nums or nums[0] < 1 or nums[-1] > int(t["n"]):
        raise _err(400, "invalid_parts", "Неверные номера частей.", trace_id)
    c, b = _client(), _bucket()

    def sign():
        return {str(p): c.generate_presigned_url("upload_part", Params={
            "Bucket": b, "Key": t["key"], "UploadId": t["up"], "PartNumber": p}, ExpiresIn=_URL_TTL_SEC) for p in nums}
    return {"urls": await run_in_threadpool(sign), "expires_in": _URL_TTL_SEC}


def _list_parts(t: Dict[str, Any]) -> List[Dict[str, Any]]:
    c, b, out, marker = _client(), _bucket(), [], 0
    while True:
        r = c.list_parts(Bucket=b, Key=t["key"], UploadId=t["up"], MaxParts=1000, PartNumberMarker=marker)
        out += [{"n": int(p["PartNumber"]), "size": int(p["Size"]), "etag": p["ETag"]} for p in r.get("Parts", [])]
        if not r.get("IsTruncated"):
            return out
        marker = int(r.get("NextPartNumberMarker") or out[-1]["n"])


def _expected_size(t: Dict[str, Any], n: int) -> int:
    return int(t["ps"]) if n < int(t["n"]) else int(t["size"]) - int(t["ps"]) * (int(t["n"]) - 1)


def _is_no_such_upload(exc: Exception) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
    return code in ("NoSuchUpload", "404")


def _object_size(key: str) -> Optional[int]:
    try:
        return int(_client().head_object(Bucket=_bucket(), Key=key)["ContentLength"])
    except Exception:
        return None


@router.post("/api/generation/uploads/status")
async def upload_status(body: TokenRequest, user: dict = Depends(get_current_user),
                        _o: None = Depends(verify_browser_origin), _rl: None = Depends(_parts_limiter)):
    trace_id = new_trace_id()
    t = read_token(body.token, str(user["id"]), trace_id)
    try:
        parts = await run_in_threadpool(_list_parts, t)
    except Exception as exc:
        if _is_no_such_upload(exc):
            if await run_in_threadpool(_object_size, t["key"]) == int(t["size"]):
                return {"completed": True, "done": list(range(1, int(t["n"]) + 1)), "part_count": t["n"]}
            raise _err(410, "upload_expired", "Сессия загрузки устарела. Загрузите файл заново.", trace_id)
        logger.error("[direct_upload] list_parts_failed trace_id=%s: %s", trace_id, exc)
        raise safe_error("storage_error", 503, trace_id)
    done = [p["n"] for p in parts if p["size"] == _expected_size(t, p["n"])]
    return {"completed": False, "done": done, "part_count": t["n"], "part_size": t["ps"]}


@router.post("/api/generation/uploads/abort")
async def upload_abort(body: TokenRequest, user: dict = Depends(get_current_user),
                       _o: None = Depends(verify_browser_origin)):
    trace_id = new_trace_id()
    t = read_token(body.token, str(user["id"]), trace_id)
    try:
        await run_in_threadpool(lambda: _client().abort_multipart_upload(Bucket=_bucket(), Key=t["key"], UploadId=t["up"]))
    except Exception as exc:
        logger.info("[direct_upload] abort_skipped trace_id=%s: %s", trace_id, exc)
    # Already completed (e.g. refused as too long) and no job was created from it -> delete the object.
    api = _api()
    try:
        has_job = api.get_job(t["jid"]) is not None
    except Exception:
        has_job = True
    if not has_job:
        api.delete_object(t["key"])
    return {"ok": True}


# ── Complete ──────────────────────────────────────────────────────────────────

def _finish_s3(t: Dict[str, Any], trace_id: str) -> str:
    """Complete the multipart upload (idempotent: an already completed upload of the right size is fine).
    Returns the object's ETag."""
    c, b = _client(), _bucket()
    try:
        parts = _list_parts(t)
    except Exception as exc:
        if _is_no_such_upload(exc) and _object_size(t["key"]) == int(t["size"]):
            return c.head_object(Bucket=b, Key=t["key"])["ETag"]
        if _is_no_such_upload(exc):
            raise _err(410, "upload_expired", "Сессия загрузки устарела. Загрузите файл заново.", trace_id)
        raise
    have = {p["n"]: p for p in parts if p["size"] == _expected_size(t, p["n"])}
    missing = [n for n in range(1, int(t["n"]) + 1) if n not in have]
    if missing:
        raise _err(409, "upload_incomplete", "Файл загружен не полностью. Продолжаем загрузку…", trace_id,
                   missing=missing[:200])
    c.complete_multipart_upload(Bucket=b, Key=t["key"], UploadId=t["up"], MultipartUpload={
        "Parts": [{"PartNumber": n, "ETag": have[n]["etag"]} for n in range(1, int(t["n"]) + 1)]})
    size = _object_size(t["key"])
    if size != int(t["size"]):
        raise _err(409, "upload_size_mismatch", "Размер файла не совпал. Загрузите файл заново.", trace_id)
    return c.head_object(Bucket=b, Key=t["key"])["ETag"]


def _check_magic(key: str) -> bool:
    head = _client().get_object(Bucket=_bucket(), Key=key, Range="bytes=0-63")["Body"].read()
    return len(head) >= 16 and _us._detect_magic(head[:32]) is not None


def _presigned_get(key: str) -> str:
    return _client().generate_presigned_url("get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=3600)


def _cut_segment(t: Dict[str, Any], start: int, length: int, dur: float, trace_id: str) -> str:
    """Stream-copy the chosen segment from S3 (HTTP range reads by ffmpeg) to a temp file ON DISK, upload it
    next to the original, drop the original. Returns the new key."""
    est = int(int(t["size"]) * min(1.0, (length + 30) / max(dur, 1.0)) * 1.1) + 64 * MB
    tmp_root = os.environ.get("DIRECT_UPLOAD_TMP", "/var/tmp")
    if shutil.disk_usage(tmp_root).free < est + 2 * GB:
        raise _err(507, "trim_no_space", "Сервер сейчас не может вырезать отрезок из такого большого файла. "
                   "Обрежьте видео на телефоне или вставьте ссылку.", trace_id)
    ext = Path(t["key"]).suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="sonya_du_", dir=tmp_root) as td:
        out = os.path.join(td, "seg" + ext)
        seek = ["-ss", str(int(start))] if start > 0 else []
        cmd = ["ffmpeg", "-y", "-v", "error", "-rw_timeout", "60000000", *seek, "-i", _presigned_get(t["key"]),
               "-t", str(int(length)), "-c", "copy", "-map", "0:v:0", "-map", "0:a:0?"]
        if ext in (".mp4", ".mov", ".3gp"):
            cmd += ["-movflags", "+faststart"]
        try:
            subprocess.run(cmd + [out], check=True, capture_output=True, timeout=1800)
        except Exception as exc:
            logger.warning("[direct_upload] trim_failed trace_id=%s: %s", trace_id, str(exc)[:300])
            raise _err(500, "trim_failed", "Не удалось вырезать отрезок. Попробуйте другой файл.", trace_id)
        new_key = t["key"].rsplit("/", 1)[0] + "/original_segment" + ext
        _client().upload_file(out, _bucket(), new_key, ExtraArgs={"ContentType": t.get("ct") or "video/mp4"})
    from scripts.prod_s3_storage import delete_object
    delete_object(t["key"])
    return new_key


def _check_length(api, user, ent, mode, t, params, trace_id) -> tuple:
    """(s3_key, size) after the length rule: probe over a presigned URL; too long -> 413 can_trim, or the
    chosen segment is cut (on disk). Same rules as _enforce_upload_length (in-memory path)."""
    from scripts import entitlements
    cap = api._source_cap_sec(user, ent, mode)
    start, trim = api._trim_request(params, cap)
    try:
        dur = entitlements.probe_duration_sec(_presigned_get(t["key"]))
    except entitlements.DurationCheckUnavailable:
        if ent.max_source_sec is not None:
            raise HTTPException(status_code=503, detail={
                "error": "duration_check_unavailable", "code": entitlements.DURATION_CHECK_UNAVAILABLE,
                "message": "Проверка видео временно недоступна. Попробуйте позже.", "trace_id": trace_id})
        return t["key"], int(t["size"]), None
    if dur is None:
        if ent.max_source_sec is not None:
            raise HTTPException(status_code=422, detail={
                "error": "source_duration_unknown", "code": entitlements.SOURCE_DURATION_UNKNOWN,
                "message": "Не удалось определить длительность видео. Проверьте файл и попробуйте снова.",
                "trace_id": trace_id})
        return t["key"], int(t["size"]), None
    if dur <= cap:
        return t["key"], int(t["size"]), dur
    if not trim:
        raise HTTPException(status_code=413, detail=api._too_long_detail(trace_id, dur, cap))
    if start >= dur:
        raise HTTPException(status_code=422, detail={"error": "trim_range", "code": "SOURCE_TRIM_RANGE", "trace_id": trace_id,
                            "message": "Начало отрезка позже конца видео. Выберите отрезок заново."})
    key = _cut_segment(t, start, trim, dur, trace_id)
    return key, _object_size(key) or 0, min(trim, dur - start)


def _fingerprint(api, mode: str, etag: str, size: int, params: Dict[str, Any]) -> str:
    normalized = copy.deepcopy(params)
    for k in api._FINGERPRINT_EXCLUDED_TOP_LEVEL_KEYS:
        normalized.pop(k, None)
    if isinstance(normalized.get("source"), dict):
        normalized["source"].pop("fileName", None)
    canon = {"mode": mode, "s3_etag": str(etag).strip('"'), "size": int(size), "params": normalized}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@router.post("/api/generation/uploads/complete", status_code=status.HTTP_202_ACCEPTED)
async def complete_upload(body: CompleteRequest, request: Request, response: Response,
                          idempotency_key_header: Optional[str] = Header(None, alias="Idempotency-Key"),
                          user: dict = Depends(get_current_user), _o: None = Depends(verify_browser_origin),
                          _rl: None = Depends(_parts_limiter)):
    api = _api()
    trace_id = new_trace_id()
    user_id = str(user["id"])
    t = read_token(body.token, user_id, trace_id)
    mode = t["mode"]
    params: Dict[str, Any] = dict(body.params or {})
    api._reject_trailer_clip(user, params, trace_id)
    idempotency_key = api._validate_idempotency_key(idempotency_key_header, trace_id)

    # Replay of an already accepted complete (network dropped on the answer): the job exists -> same answer.
    if idempotency_key:
        try:
            prev = api.get_job_by_idempotency_key(user_id, idempotency_key)
        except Exception:
            prev = None
        if prev and str(prev.get("id")) == t["jid"]:
            response.headers["Idempotency-Replayed"] = "true"
            return api.JobResponse(job_id=str(prev["id"]), status=str(prev.get("status")), mode=str(prev.get("mode")),
                                   created_at=str(prev.get("created_at")) if prev.get("created_at") else None)

    _check_mode(api, mode, request, user_id, trace_id)
    priority, plan = api._resolve_priority(user)
    ent = api._resolve_entitlement(user, mode, trace_id, params)
    priority, plan = api._priority_for(ent, priority, plan)
    units = api._apply_quality(user, ent, mode, params, trace_id)

    try:
        etag = await run_in_threadpool(_finish_s3, t, trace_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[direct_upload] complete_failed job_id=%s trace_id=%s: %s", t["jid"], trace_id, exc)
        raise safe_error("storage_error", 503, trace_id)

    def reject(exc: HTTPException, key: str):
        from scripts.prod_s3_storage import delete_object
        delete_object(key)
        api.audit(api.EVT_UPLOAD_REJECTED, user_id=user_id, trace_id=trace_id,
                  details={"mode": mode, "reason": str(exc.detail)[:300], "path": "direct"},
                  ip_address=request.client.host if request.client else None)
        return exc

    if not await run_in_threadpool(_check_magic, t["key"]):
        raise reject(_err(400, "invalid_video_content", "Файл не похож на видео. Выберите другой файл.", trace_id), t["key"])
    try:
        s3_key, size, _dur = await run_in_threadpool(_check_length, api, user, ent, mode, t, params, trace_id)
    except HTTPException as exc:
        if exc.status_code == 413:
            # Too long: the file stays so the browser can ask for a segment and call complete again
            # (no re-upload). «Cancel» -> /abort deletes it; nothing else points at it.
            api.audit(api.EVT_UPLOAD_REJECTED, user_id=user_id, trace_id=trace_id,
                      details={"mode": mode, "reason": "source_too_long", "path": "direct"},
                      ip_address=request.client.host if request.client else None)
        elif exc.status_code == 422:
            raise reject(exc, t["key"])
        raise

    fp = _fingerprint(api, mode, etag, size, params) if idempotency_key else None
    job_id = t["jid"]
    try:
        result = await run_in_threadpool(lambda: api.create_job_with_quota(
            job_id=job_id, user_id=user_id, mode=mode, params=params, s3_input_key=s3_key,
            idempotency_key=idempotency_key, idempotency_fingerprint=fp, queue_priority=priority,
            bypass_quota=ent.kind == "legacy_pro", subscription_id=ent.subscription_id,
            pack_mode=(ent.plan_mode if ent.kind == "pack" else None), units=units))
    except Exception as exc:
        logger.error("[direct_upload] create_job_failed trace_id=%s: %s", trace_id, exc)
        raise safe_error("db_error", 500, trace_id)   # object kept: a retry of complete re-uses it

    outcome = result["outcome"]
    if outcome == "quota_exceeded":
        api.delete_object(s3_key)
        raise HTTPException(status_code=402, detail={"error": "payment_required", "code": "FREE_PLAN_USED",
                            "message": "Бесплатная генерация уже использована. Выберите тариф SONYA.", "trace_id": trace_id})
    if outcome == "plan_limit_reached":
        api.delete_object(s3_key)
        raise HTTPException(status_code=402, detail=api._plan_limit_reached_detail(ent, trace_id))
    if outcome == "existing":
        existing = result["job"]
        if str(existing.get("s3_input_key") or "") != s3_key:
            api.delete_object(s3_key)
        if existing.get("idempotency_fingerprint") == fp or str(existing.get("id")) == job_id:
            response.headers["Idempotency-Replayed"] = "true"
            return api.JobResponse(job_id=str(existing.get("id")), status=str(existing.get("status")),
                                   mode=str(existing.get("mode")),
                                   created_at=str(existing.get("created_at")) if existing.get("created_at") else None)
        raise HTTPException(status_code=409, detail={"error": "idempotency_key_conflict",
                            "message": "This Idempotency-Key was already used with a different request", "trace_id": trace_id})

    job_id = str(result["job"]["id"])
    try:
        from scripts.prod_job_store import _get_conn  # type: ignore
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE generation_jobs SET priority=%s, plan=%s WHERE id=%s", (priority, plan, job_id))
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[direct_upload] priority_col_update_skipped job_id=%s exc=%s", job_id, exc)
    try:
        api.add_job_file(job_id=job_id, user_id=user_id, file_type="input", s3_key=s3_key, filename=t["name"],
                         content_type=t.get("ct") or "video/mp4", size_bytes=size)
    except Exception as exc:
        logger.warning("[direct_upload] add_input_file_failed job_id=%s: %s", job_id, exc)
    api.audit(api.EVT_JOB_CREATED, user_id=user_id, job_id=job_id, trace_id=trace_id,
              details={"mode": mode, "size_bytes": size, "plan": plan, "priority": priority, "entitlement": ent.kind,
                       "plan_id": ent.plan_id, "path": "direct_s3"},
              ip_address=request.client.host if request.client else None)
    logger.info("[direct_upload] job_created job_id=%s user_id=%s mode=%s plan=%s size=%d", job_id, user_id, mode, plan, size)
    job = api.get_job(job_id)
    return api.JobResponse(job_id=job_id, status=api.JOB_STATUS_QUEUED, mode=mode,
                           created_at=str(job.get("created_at")) if job else None)
