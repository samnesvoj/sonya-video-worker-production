"""
url_ingest.py
=============
Server-side "download video from URL" for SONYA's generation pipeline.

Supported sources:
  - YouTube  (youtube.com, youtu.be)
  - VK Video (vk.com, vkvideo.ru)
  - Twitch   (twitch.tv) — same public-VOD path as YouTube/VK, no extra code
  - Direct HTTP(S) links to a video file (checked by Content-Type / extension)

All three platform sources go through plain `yt-dlp` (a standard, widely
used open-source extractor) — nothing here impersonates another app, spoofs
a device/bundle ID, or calls an undocumented third-party proxy API. No
cookies file, no authenticated/paid-content bypass: only what is publicly
reachable without logging in is supported. (SONYA's own recovered
`scripts/shared/download/downloader.py` — ported from a decompiled
third-party binary — intentionally is NOT used here: its fallback paths
impersonate the Any4k Android app and call an unrelated product's
(boosta.pro) private API. This module replaces it for production use.)

Security:
  - Only http/https URLs are accepted.
  - SSRF guard: the resolved IP(s) of the URL's host are checked against
    private / loopback / link-local / cloud-metadata ranges before any
    request is made (both for the yt-dlp path and the direct-download
    path), and redirects are re-checked the same way.
  - Downloads are size- and duration-bounded; a direct download aborts the
    moment it exceeds the byte cap, it never trusts Content-Length alone.
  - Every temp file this module creates is the caller's responsibility to
    remove (download_video() returns the path; callers use try/finally).
"""
from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep missing
    yt_dlp = None  # type: ignore
    YT_DLP_AVAILABLE = False


# ── Errors ────────────────────────────────────────────────────────────────

class UrlIngestError(Exception):
    """Base class for all url_ingest failures. `.user_message` is safe to
    show to the end user (Russian, no internals leaked)."""
    user_message = "Не удалось обработать ссылку."


class UnsupportedUrlError(UrlIngestError):
    user_message = "Ссылка не поддерживается. Используйте YouTube, VK или прямую ссылку на видеофайл."


class UnsafeUrlError(UrlIngestError):
    user_message = "Эта ссылка недоступна для загрузки."


class DownloadLimitExceeded(UrlIngestError):
    def __init__(self, message: str, duration_sec: Optional[float] = None, cap_sec: Optional[int] = None):
        super().__init__(message)
        self.user_message = message
        self.duration_sec, self.cap_sec = duration_sec, cap_sec   # set when the LENGTH is the problem


class DownloadFailed(UrlIngestError):
    user_message = "Не удалось скачать видео по ссылке. Проверьте, что она рабочая и видео публичное."


# ── Config ────────────────────────────────────────────────────────────────
#
# Mode-conditional limits: streamer mode is the one long-form use case
# (multi-hour VODs) — every other mode keeps today's short-clip limits
# completely unchanged. `mode` defaults to None everywhere below, which
# always resolves to the original (pre-streamer) env vars/defaults, so any
# caller that doesn't pass `mode` yet behaves exactly as before. These two
# functions are the ONLY place either limit is computed — every call site
# in this module threads `mode` through rather than re-deriving a value.

_STREAMER_DEFAULT_MAX_DURATION_SEC = 7 * 60 * 60  # 7h — see PHASE A audit: real streams run 4-7h
_STREAMER_DEFAULT_MAX_UPLOAD_MB = 20480            # 20 GB — generous starting point for a multi-hour
                                                     # VOD at typical stream bitrates; an ops-tunable
                                                     # default, not a rigorously derived ceiling.


def _max_bytes(mode: Optional[str] = None) -> int:
    if mode == "streamer":
        return int(os.environ.get("STREAMER_MAX_UPLOAD_SIZE_MB",
                                    str(_STREAMER_DEFAULT_MAX_UPLOAD_MB))) * 1024 * 1024
    return int(os.environ.get("MAX_UPLOAD_SIZE_MB", "2048")) * 1024 * 1024


def _max_duration_sec(mode: Optional[str] = None) -> int:
    if mode == "streamer":
        return int(os.environ.get("STREAMER_URL_MAX_DURATION_SEC",
                                    str(_STREAMER_DEFAULT_MAX_DURATION_SEC)))
    return int(os.environ.get("URL_DOWNLOAD_MAX_DURATION_SEC", "3600"))  # 60 min


def _network_timeout_sec() -> int:
    return int(os.environ.get("URL_DOWNLOAD_TIMEOUT_SEC", "600"))  # 10 min


_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

_DIRECT_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".3gp", ".m3u8"}
_DIRECT_VIDEO_CONTENT_TYPES = (
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
    "video/webm", "video/mpeg", "video/3gpp", "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
)

_FORBIDDEN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ── Platform detection ───────────────────────────────────────────────────

def detect_platform(url: str) -> str:
    """Returns 'youtube' | 'vk' | 'twitch' | 'direct' | 'unsupported'."""
    if not url or not isinstance(url, str):
        return "unsupported"

    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return "unsupported"

    if parsed.scheme not in ("http", "https"):
        return "unsupported"

    host = (parsed.hostname or "").lower()
    if not host:
        return "unsupported"

    if host in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"):
        return "youtube"
    if host.endswith(".youtube.com") or host == "youtube.com" or host == "youtu.be":
        return "youtube"

    if host in ("vk.com", "www.vk.com", "m.vk.com", "vkvideo.ru", "www.vkvideo.ru"):
        return "vk"
    if host.endswith(".vk.com") or host.endswith(".vkvideo.ru"):
        return "vk"

    if host in ("twitch.tv", "www.twitch.tv", "m.twitch.tv", "clips.twitch.tv"):
        return "twitch"
    if host.endswith(".twitch.tv"):
        return "twitch"

    # Direct video URL: only if the path looks like a real video file, or
    # (checked later, over the network) the server reports a video
    # Content-Type. Path-based check here is a cheap first filter so we
    # don't even attempt an SSRF-checked connection for obvious junk.
    path = (parsed.path or "").lower()
    if any(path.endswith(ext) for ext in _DIRECT_VIDEO_EXT):
        return "direct"

    return "unsupported"


# ── SSRF guard ───────────────────────────────────────────────────────────

def _is_public_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    # Cloud metadata endpoint (AWS/GCP/Azure/most clouds use this).
    if ip_str == "169.254.169.254":
        return False
    return True


def assert_safe_url(url: str) -> None:
    """Raises UnsafeUrlError if the URL's scheme isn't http(s), or its host
    resolves (even partially) to a private/loopback/link-local/metadata
    address. Call this again after following any redirect."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError("scheme not allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("no host")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"cannot resolve host: {exc}") from exc

    if not infos:
        raise UnsafeUrlError("host did not resolve to any address")

    for info in infos:
        addr = info[4][0]
        if not _is_public_ip(addr):
            raise UnsafeUrlError(f"host resolves to a non-public address ({addr})")


# ── Filename helpers ─────────────────────────────────────────────────────

def _safe_filename(name: str, max_len: int = 200) -> str:
    safe = _FORBIDDEN_FILENAME.sub("_", name).strip(". ")
    if len(safe) > max_len:
        stem = Path(safe).stem[: max_len - 10]
        ext = Path(safe).suffix
        safe = stem + ext
    return safe or "video.mp4"


# ── yt-dlp path (youtube / vk / twitch) ─────────────────────────────────

_YTDLP_FORMAT = (
    "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
)


def probe(url: str, platform: str, mode: Optional[str] = None,
          max_duration_sec: Optional[int] = None) -> dict:
    """Metadata-only lookup (no download). Raises DownloadLimitExceeded if
    the video is already known to exceed the duration cap; raises
    DownloadFailed if the video can't be resolved at all.

    max_duration_sec: the caller's paid-plan cap (scripts/entitlements.py),
    which replaces the generic per-mode default -- e.g. a 2-hour plan must
    not be cut off by the 60-minute free default. It is only an early,
    platform-reported check; the authoritative one is ffprobe on the
    downloaded file."""
    assert_safe_url(url)
    duration_cap = max_duration_sec or _max_duration_sec(mode)

    if platform not in ("youtube", "vk", "twitch"):
        # 'direct' is probed via a lightweight HEAD/range request instead —
        # see _probe_direct().
        return _probe_direct(url, mode=mode)

    if not YT_DLP_AVAILABLE:
        raise DownloadFailed("yt-dlp is not installed")

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "http_headers": {"User-Agent": _UA},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        logger.warning("[url_ingest] probe_failed platform=%s url_host=%s: %s",
                        platform, urlparse(url).hostname, exc)
        raise DownloadFailed(f"probe failed: {exc}") from exc

    duration = info.get("duration")
    if duration and duration > duration_cap:
        raise DownloadLimitExceeded(
            f"Видео слишком длинное ({int(duration // 60)} мин). "
            f"Максимум — {duration_cap // 60} мин.",
            duration_sec=float(duration), cap_sec=int(duration_cap),
        )

    filesize = info.get("filesize") or info.get("filesize_approx")
    if filesize and filesize > _max_bytes(mode):
        raise DownloadLimitExceeded(
            f"Видео слишком большое ({filesize // (1024 * 1024)} МБ). "
            f"Максимум — {_max_bytes(mode) // (1024 * 1024)} МБ."
        )

    return {"title": info.get("title"), "duration": duration, "filesize": filesize}


def _probe_direct(url: str, mode: Optional[str] = None) -> dict:
    try:
        resp = requests.head(url, headers={"User-Agent": _UA}, timeout=30, allow_redirects=True)
        # Some servers don't support HEAD properly (405 / no content-length) —
        # fall back to a ranged GET of just the headers.
        if resp.status_code >= 400 or "content-length" not in resp.headers:
            resp = requests.get(url, headers={"User-Agent": _UA, "Range": "bytes=0-0"},
                                 timeout=30, stream=True, allow_redirects=True)
    except requests.RequestException as exc:
        raise DownloadFailed(f"probe failed: {exc}") from exc

    # Re-validate after redirects landed us on a possibly-different host.
    assert_safe_url(str(resp.url))

    if resp.status_code >= 400:
        raise DownloadFailed(f"server returned HTTP {resp.status_code}")

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith("video/") and content_type not in _DIRECT_VIDEO_CONTENT_TYPES:
        raise UnsupportedUrlError()

    content_length = resp.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            size = None
        if size and size > _max_bytes(mode):
            raise DownloadLimitExceeded(
                f"Файл слишком большой ({size // (1024 * 1024)} МБ). "
                f"Максимум — {_max_bytes(mode) // (1024 * 1024)} МБ."
            )

    return {"content_type": content_type}


# ── Download ─────────────────────────────────────────────────────────────

def download_video(
    url: str,
    platform: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    mode: Optional[str] = None,
) -> tuple[str, str]:
    """
    Downloads the video to a fresh temp file.

    Returns (local_path, extension). Caller owns the returned file and MUST
    remove it (try/finally) once done — this function never cleans up its
    own output on the success path.
    """
    assert_safe_url(url)

    tmp_dir = tempfile.mkdtemp(prefix="sonya_url_ingest_")

    if platform in ("youtube", "vk", "twitch"):
        return _download_via_ytdlp(url, platform, tmp_dir, progress_cb, mode=mode)
    if platform == "direct":
        return _download_direct(url, tmp_dir, progress_cb, mode=mode)
    raise UnsupportedUrlError()


def _download_via_ytdlp(url: str, platform: str, tmp_dir: str,
                         progress_cb: Optional[Callable[[float], None]],
                         mode: Optional[str] = None) -> tuple[str, str]:
    if not YT_DLP_AVAILABLE:
        raise DownloadFailed("yt-dlp is not installed")

    outtmpl = os.path.join(tmp_dir, f"{platform}_{uuid.uuid4().hex}.%(ext)s")

    def _hook(d: dict) -> None:
        if not progress_cb:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            if total:
                try:
                    progress_cb(min(99.0, 100.0 * downloaded / total))
                except Exception:
                    pass

    opts = {
        "format": _YTDLP_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
        "retries": 3,
        "socket_timeout": 30,
        "max_filesize": _max_bytes(mode),
        "http_headers": {"User-Agent": _UA},
        "progress_hooks": [_hook],
        # Deliberately no cookiefile / proxy / POT-provider — only public,
        # unauthenticated content is supported.
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        _rmtree(tmp_dir)
        logger.warning("[url_ingest] download_failed platform=%s: %s", platform, exc)
        raise DownloadFailed(str(exc)) from exc

    path = _find_downloaded_file(info, tmp_dir)
    if not path:
        _rmtree(tmp_dir)
        raise DownloadFailed("yt-dlp reported success but no output file was found")

    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass

    return path, Path(path).suffix or ".mp4"


def _find_downloaded_file(info: Optional[dict], tmp_dir: str) -> Optional[str]:
    if info and info.get("requested_downloads"):
        for dl in info["requested_downloads"]:
            path = dl.get("filepath") or dl.get("filename")
            if path and os.path.exists(path):
                return path
    candidates = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)] if os.path.isdir(tmp_dir) else []
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _download_direct(url: str, tmp_dir: str,
                      progress_cb: Optional[Callable[[float], None]],
                      mode: Optional[str] = None) -> tuple[str, str]:
    limit = _max_bytes(mode)
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in _DIRECT_VIDEO_EXT:
        ext = ".mp4"
    out_path = os.path.join(tmp_dir, f"direct_{uuid.uuid4().hex}{ext}")

    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, stream=True,
                             timeout=_network_timeout_sec(), allow_redirects=True)
        assert_safe_url(str(resp.url))
        resp.raise_for_status()

        total = resp.headers.get("content-length")
        total_bytes = int(total) if total and total.isdigit() else None
        if total_bytes and total_bytes > limit:
            raise DownloadLimitExceeded(
                f"Файл слишком большой ({total_bytes // (1024 * 1024)} МБ). "
                f"Максимум — {limit // (1024 * 1024)} МБ."
            )

        downloaded = 0
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > limit:
                    raise DownloadLimitExceeded(
                        f"Файл превышает лимит {limit // (1024 * 1024)} МБ."
                    )
                fh.write(chunk)
                if progress_cb and total_bytes:
                    try:
                        progress_cb(min(99.0, 100.0 * downloaded / total_bytes))
                    except Exception:
                        pass
    except (requests.RequestException, DownloadLimitExceeded):
        _rmtree(tmp_dir)
        raise
    except Exception as exc:
        _rmtree(tmp_dir)
        raise DownloadFailed(str(exc)) from exc

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        _rmtree(tmp_dir)
        raise DownloadFailed("empty download")

    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass

    return out_path, ext


def _rmtree(tmp_dir: str) -> None:
    try:
        for f in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, f))
            except OSError:
                pass
        os.rmdir(tmp_dir)
    except OSError:
        pass


def validate_downloaded_file(local_path: str, hint_name: str, mode: Optional[str] = None) -> tuple[bytes, str]:
    """
    Read a downloaded file from disk and run it through the exact same
    filename/size/magic-byte validation a browser-uploaded file goes
    through (scripts.upload_security.validate_video_bytes) — a URL
    download gets no special treatment, and can be rejected for the same
    reasons a direct upload would be.

    Returns (content_bytes, safe_filename). Raises fastapi.HTTPException
    on validation failure (same as the upload path).
    """
    from scripts.upload_security import validate_video_bytes  # local import: avoid a hard fastapi dep at module load for pure-unit tests

    size = os.path.getsize(local_path)
    if size > _max_bytes(mode):
        raise DownloadLimitExceeded(
            f"Файл слишком большой ({size // (1024 * 1024)} МБ). "
            f"Максимум — {_max_bytes(mode) // (1024 * 1024)} МБ."
        )

    with open(local_path, "rb") as fh:
        content = fh.read()

    safe_name = validate_video_bytes(content, hint_name, max_size_bytes=_max_bytes(mode))
    return content, safe_name


def cleanup(local_path: str) -> None:
    """Best-effort removal of a downloaded file and its parent temp dir."""
    try:
        tmp_dir = os.path.dirname(local_path)
        if os.path.exists(local_path):
            os.remove(local_path)
        if os.path.isdir(tmp_dir) and os.path.basename(tmp_dir).startswith("sonya_url_ingest_"):
            if not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
    except OSError as exc:
        logger.warning("[url_ingest] cleanup_failed path=%s: %s", local_path, exc)
