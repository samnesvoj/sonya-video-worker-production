"""Worker-side "download the source by link" (params.source_fetch).

The Moscow VPS cannot reach youtube.com, so the site hands such a job to the GPU worker with
params.source_fetch = {"url", "platform", "max_source_sec"}. The worker downloads the best <=1080p (H.264 preferred) with
yt-dlp (+deno), checks the length against the plan, uploads the file to the job's s3_input_key and carries on as usual."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
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


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    return one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many


def fetch_source(sf: Dict[str, Any], mode: str, s3_key: str, work_dir: Path,
                 progress: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    """Download sf['url'], enforce limits, upload to s3_key. Returns {"path": local file, "duration": sec, "bytes": n}.
    Raises SourceFetchError with a message meant for the user."""
    from scripts.prod_s3_storage import upload_file

    url = str(sf.get("url") or "").strip()
    platform = str(sf.get("platform") or "") or url_ingest.detect_platform(url)
    if not url or platform == "unsupported":
        raise SourceFetchError(*_REASONS[-1][1])
    os.environ.setdefault("SONYA_YTDLP_MAX_HEIGHT", "1440")   # GPU worker: better source for the 9:16 crop
    plan_cap = sf.get("max_source_sec")
    cap = int(plan_cap) if plan_cap else url_ingest._max_duration_sec(mode)
    try:
        url_ingest.assert_safe_url(url)
        url_ingest.probe(url, platform, mode=mode, max_duration_sec=cap)
        path, _ext = url_ingest.download_video(url, platform, progress_cb=progress, mode=mode)
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
        if dur is None:
            raise SourceFetchError("SOURCE_DURATION_UNKNOWN", "Не удалось определить длительность видео. Попробуйте другую ссылку или загрузите файл.")
        if dur > cap:
            mx = cap // 60
            raise SourceFetchError("SOURCE_TOO_LONG",
                                   f"Видео длиннее, чем позволяет тариф: {int(dur // 60)} мин, максимум — {mx} {_plural(mx, 'минута', 'минуты', 'минут')}.")
        size = os.path.getsize(path)
        dest = work_dir / f"input{Path(path).suffix or '.mp4'}"
        shutil.move(path, dest)
        try:
            upload_file(dest, s3_key, content_type="video/mp4")
        except Exception as exc:
            raise SourceFetchError("SOURCE_STORAGE", "Не удалось сохранить скачанное видео. Попробуйте позже.") from exc
        logger.info("[source_fetch] ok platform=%s duration=%.0fs bytes=%d", platform, dur, size)
        return {"path": dest, "duration": dur, "bytes": size}
    finally:
        url_ingest.cleanup(path)
