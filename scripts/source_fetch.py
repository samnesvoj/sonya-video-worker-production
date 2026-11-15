"""Worker-side "download the source by link" (params.source_fetch).

The Moscow VPS cannot reach youtube.com, so the site hands such a job to the GPU worker with
params.source_fetch = {"url", "platform", "max_source_sec"}. The worker downloads the best <=1080p (H.264 preferred) with
yt-dlp (+deno), checks the length against the plan, uploads the file to the job's s3_input_key and carries on as usual."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from scripts import url_ingest

logger = logging.getLogger(__name__)


class SourceFetchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


# (substring of the yt-dlp error, lower-cased) -> (code, message for the user)
_REASONS = (
    # YouTube flags some data-centre IPs: the same link works from another server. The worker requeues the job and
    # stops claiming; the dispatcher bans this machine and starts another one (dispatcher/queue.py bot_check_workers).
    (("confirm you’re not a bot", "confirm you're not a bot", "not a bot"),
     ("SOURCE_BOT_CHECK", "YouTube не отдал видео серверу обработки (проверка «не бот»). Попробуйте ещё раз через несколько минут или загрузите файл.")),
    (("private video", "this video is private", "sign in to view", "login required"),
     ("SOURCE_PRIVATE", "Видео приватное или закрыто по ссылке. Откройте доступ (публичное видео) и попробуйте снова.")),
    (("confirm your age", "age-restricted", "age restricted"),
     ("SOURCE_AGE_RESTRICTED", "Видео с ограничением по возрасту — скачать его по ссылке нельзя. Загрузите файл.")),
    (("video unavailable", "has been removed", "no longer available", "does not exist", "been terminated", "deleted", "not available"),
     ("SOURCE_UNAVAILABLE", "Видео недоступно или удалено. Проверьте ссылку.")),
    (("members-only", "members only", "join this channel", "premium"),
     ("SOURCE_PRIVATE", "Видео только для подписчиков/участников канала — скачать его нельзя. Загрузите файл.")),
    (("not made this video available in your country", "geo", "blocked it in your country"),
     ("SOURCE_GEO", "Видео недоступно в стране сервера обработки. Загрузите файл.")),
    (("live event", "is live", "premiere"),
     ("SOURCE_LIVE", "Это идущая трансляция или премьера — дождитесь окончания и повторите.")),
    # still 403 after the in-place retries (url_ingest): requeued once (retry=True in gpu_worker), may land on another host
    (("http error 403", "403: forbidden"),
     ("SOURCE_FORBIDDEN", "YouTube временно не отдал видео серверу обработки. Попробуйте ещё раз через несколько минут или загрузите файл.")),
    (("unsupported url",), ("SOURCE_UNSUPPORTED", "Ссылка не поддерживается. Поддерживаются YouTube, VK, Twitch.")),
)
_GENERIC = ("SOURCE_DOWNLOAD_FAILED", "Не удалось скачать видео по ссылке. Проверьте, что она рабочая и видео публичное.")


def classify_error(exc: BaseException) -> tuple[str, str]:
    text = f"{exc} {exc.__cause__ or ''}".lower()
    for needles, res in _REASONS:
        if any(n in text for n in needles):
            return res
    return _GENERIC


def _duration(path: str) -> Optional[float]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return None


def _check_free_space(work_dir: Path, seconds: Optional[float]) -> None:
    """Before the download (speedfix 2026-11-15): ~0.6 MB/s of 1080p H.264, x2 for the yt-dlp merge + upload copy,
    + 1 GB slack. A full disk otherwise breaks normalize/render half an hour later."""
    if not seconds:
        return
    need = int(seconds * 1.2e6) + (1 << 30)
    try:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(work_dir).free
    except OSError:
        return
    if free < need:
        logger.error("[source_fetch] low disk: free %d MB < need %d MB for %.0f s", free >> 20, need >> 20, seconds)
        raise SourceFetchError("SOURCE_NO_SPACE", "На сервере сейчас не хватает места для такого длинного видео. "
                                                  "Попробуйте ещё раз через несколько минут.")


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    return one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many


def upload_done(got: Optional[Dict[str, Any]], timeout: float = 900.0) -> Optional[str]:
    """Wait for the background S3 copy of fetch_source(background_upload=True). -> None, or the error text."""
    box = (got or {}).get("upload")
    if not box:
        return None
    box["thread"].join(timeout)
    if box["thread"].is_alive():
        return "timeout"
    return str(box["error"])[:200] if box["error"] else None


def fetch_source(sf: Dict[str, Any], mode: str, s3_key: str, work_dir: Path,
                 progress: Optional[Callable[[float], None]] = None, background_upload: bool = False) -> Dict[str, Any]:
    """Download sf['url'], enforce limits, upload to s3_key. Returns {"path": local file, "duration": sec, "bytes": n}.
    Raises SourceFetchError with a message meant for the user."""
    from scripts.prod_s3_storage import upload_file
    t0 = time.monotonic()

    url = str(sf.get("url") or "").strip()
    platform = str(sf.get("platform") or "") or url_ingest.detect_platform(url)
    if not url or platform == "unsupported":
        raise SourceFetchError(*_REASONS[-1][1])
    os.environ.setdefault("SONYA_YTDLP_MAX_HEIGHT", "1440")   # GPU worker: better source for the 9:16 crop
    plan_cap = sf.get("max_source_sec")
    cap = int(plan_cap) if plan_cap else url_ingest._max_duration_sec(mode)
    # The user chose a segment (start..end, or the first N minutes) instead of a refusal: its length <= cap.
    start, trim = url_ingest.trim_request(sf, cap)
    try:
        url_ingest.assert_safe_url(url)
        # FIRST step, before any download: a too-long video fails here within seconds. A segment lifts the
        # length/size checks of the whole video (only the segment is downloaded).
        info = url_ingest.probe(url, platform, mode=mode, max_duration_sec=(10 ** 6 if trim else cap),
                                max_bytes=(1 << 50 if trim else url_ingest.long_source_overrides(cap, mode).get("max_bytes")))
        total = (info or {}).get("duration")
        if trim and total and start >= total:
            raise SourceFetchError("SOURCE_TRIM_RANGE", "Начало отрезка позже конца видео. Выберите отрезок заново.")
        lim = url_ingest.long_source_overrides(trim or total or cap, mode)
        _check_free_space(work_dir, trim or total)
        path, _ext = url_ingest.download_video(url, platform, progress_cb=progress, mode=mode, trim_sec=trim or None,
                                               trim_start=start, **lim)
    except url_ingest.DownloadLimitExceeded as exc:
        raise SourceFetchError("SOURCE_TOO_LONG", exc.user_message) from exc
    except url_ingest.UnsafeUrlError as exc:
        raise SourceFetchError("SOURCE_UNSAFE_URL", exc.user_message) from exc
    except url_ingest.UnsupportedUrlError as exc:
        raise SourceFetchError(*_REASONS[-1][1]) from exc
    except url_ingest.DownloadFailed as exc:
        logger.warning("[source_fetch] failed platform=%s: %s", platform, str(exc)[:300])
        raise SourceFetchError(*classify_error(exc)) from exc
    try:
        dur = _duration(path)
        if trim and dur is not None and dur > trim + 3:    # keyframe-rounded section came out longer: cut exactly
            path = url_ingest.trim_file(path, trim)
            dur = _duration(path)
        if dur is None:
            raise SourceFetchError("SOURCE_DURATION_UNKNOWN", "Не удалось определить длительность видео. Попробуйте другую ссылку или загрузите файл.")
        if dur > cap:
            mx = cap // 60
            raise SourceFetchError("SOURCE_TOO_LONG",
                                   f"Видео длиннее, чем позволяет тариф: {int(dur // 60)} мин, максимум — {mx} {_plural(mx, 'минута', 'минуты', 'минут')}.")
        size = os.path.getsize(path)
        dest = work_dir / f"input{Path(path).suffix or '.mp4'}"
        shutil.move(path, dest)
        t_dl = time.monotonic() - t0
        if background_upload:
            # Load test 2026-11-16: the S3 copy of a 3 h source (2 GB) blocked the job before ASR. The copy is a record
            # (a requeued job fetches the link again), so it runs next to the processing; the worker joins it before
            # the job completes (upload_done()).
            box: Dict[str, Any] = {"error": None}

            def _up():
                try:
                    upload_file(dest, s3_key, content_type="video/mp4")
                except Exception as exc:   # noqa: BLE001 - reported by upload_done()
                    box["error"] = exc
            th = threading.Thread(target=_up, name="source-s3-copy", daemon=True)
            th.start()
            box["thread"] = th
            logger.info("[source_fetch] ok platform=%s duration=%.0fs bytes=%d download=%.0fs (S3 copy in background)",
                        platform, dur, size, t_dl)
            return {"path": dest, "duration": dur, "bytes": size, "upload": box, "download_s": round(t_dl, 1)}
        try:
            upload_file(dest, s3_key, content_type="video/mp4")
        except Exception as exc:
            raise SourceFetchError("SOURCE_STORAGE", "Не удалось сохранить скачанное видео. Попробуйте позже.") from exc
        logger.info("[source_fetch] ok platform=%s duration=%.0fs bytes=%d download=%.0fs upload=%.0fs", platform, dur, size,
                    t_dl, time.monotonic() - t0 - t_dl)
        return {"path": dest, "duration": dur, "bytes": size, "download_s": round(t_dl, 1)}
    finally:
        url_ingest.cleanup(path)
