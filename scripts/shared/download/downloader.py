# -*- coding: utf-8 -*-
"""
downloader.py
=============
Video download module — ported from yuvelirochka.
PUBLIC API IS DISABLED in SONYA production.
Use only for internal file-path-based processing.
Do NOT call download() or get_info() from production code.

Reconstructed from: downloader.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary.

Confirmed recovered elements:
  - Class: VideoDownloader  (exact from binary: uVideoDownloader.*)
  - Methods: __init__, download, get_info, _extract_video_id_and_platform,
             _filter_formats, _get_audio_tracks, _get_language_name,
             _apply_pot_provider, _download_youtube_via_proxy,
             _download_kick_via_proxy, _download_via_any4k
  - Env vars: BOOSTA_SERVER_DOWNLOAD, BOOSTA_ACCESS_TOKEN, KICK_PROXY_URL,
              YOUTUBE_PROXY_URL, YOUTUBE_POT_PROVIDER_URL, COOKIES_FILE
  - Server endpoints: https://boosta.pro/api/youtube/download
                      https://boosta.pro/api/kick/metadata
  - Any4k endpoints: https://api.any4k.com/v1/dlp/check
                     https://api.any4k.com/v1/dlp/download
  - Format selectors: bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/...
  - Regex: [?&]v=([a-zA-Z0-9_-]+), youtu\\.be/([a-zA-Z0-9_-]+), [<>:"/\\|?*]
  - POT provider: youtubepot-bgutilhttp (YOUTUBE_POT_PROVIDER_URL)
  - aria2c detection via shutil.which
  - Audio track selection: best_original > best_clean > best_any
  - Languages: 18 languages in _get_language_name lang_map
  - Progress: _percent_str, _speed_str, _eta_str from yt-dlp hook dict
  - User-agents: Mac Chrome + Windows Chrome
  - Kick: curl_cffi impersonation with Chrome, m3u8 via ffmpeg
  - YouTube fallback: video_url + audio_url merged with ffmpeg, HLS variant

Pipeline:
    URL → VideoDownloader.download() → local .mp4
        → transcriber.py → analyzer.py → clipper.py → final clips
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dotenv
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# yt-dlp
# ---------------------------------------------------------------------------
YT_DLP_AVAILABLE = False
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    yt_dlp = None  # type: ignore

# ---------------------------------------------------------------------------
# Optional requests (for proxy/Any4k calls)
# ---------------------------------------------------------------------------
REQUESTS_AVAILABLE = False
try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore

__all__ = ["VideoDownloader", "download"]

# SONYA production: public download API is disabled.
# Calling download() or VideoDownloader.download() raises RuntimeError.
_PUBLIC_API_DISABLED = True

# ---------------------------------------------------------------------------
# Constants decoded from binary
# ---------------------------------------------------------------------------
_FORBIDDEN_WIN_CHARS  = re.compile(r'[<>:"/\\|?*]')
_RE_YT_WATCH          = re.compile(r'[?&]v=([a-zA-Z0-9_-]+)')
_RE_YT_SHORT          = re.compile(r'youtu\.be/([a-zA-Z0-9_-]+)')
_RE_YT_SHORTS_PATH    = re.compile(r'/shorts/([a-zA-Z0-9_-]+)')

_SERVER_YT_ENDPOINT   = "https://boosta.pro/api/youtube/download"
_SERVER_KICK_ENDPOINT = "https://boosta.pro/api/kick/metadata"
_ANY4K_CHECK          = "https://api.any4k.com/v1/dlp/check"
_ANY4K_DOWNLOAD       = "https://api.any4k.com/v1/dlp/download"
_ANY4K_BUNDLE         = "com.any4k.downloader"

_UA_MAC   = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
             "AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/120.0.0.0 Safari/537.36")
_UA_WIN   = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
             "AppleWebKit/537.36")

# Quality → yt-dlp format selector
_FORMAT_MAP: Dict[str, str] = {
    "best":         "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/bestvideo+bestaudio/best",
    "auto":         "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/bestvideo+bestaudio/best",
    "1080p":        "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p":         "bestvideo[vcodec^=avc1][height<=720]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p":         "bestvideo[vcodec^=avc1][height<=480]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p":         "bestvideo[height<=360]+bestaudio/best[height<=360]",
    "audio":        "bestaudio/best",
}
_FORMAT_MAP["Best Available"]       = _FORMAT_MAP["best"]
_FORMAT_MAP["Best Available (Auto)"] = _FORMAT_MAP["best"]
_FORMAT_MAP["Auto-detect"]          = _FORMAT_MAP["best"]

# Languages decoded from binary variable names
_LANG_MAP: Dict[str, str] = {
    "en": "English",  "ru": "Russian",   "ar": "Arabic",
    "zh": "Chinese",  "es": "Spanish",   "fr": "French",
    "de": "German",   "it": "Italian",   "ja": "Japanese",
    "ko": "Korean",   "pt": "Portuguese","tr": "Turkish",
    "hi": "Hindi",    "vi": "Vietnamese","id": "Indonesian",
    "pl": "Polish",   "nl": "Dutch",     "uk": "Ukrainian",
    "th": "Thai",
}


# ===========================================================================
# VideoDownloader
# ===========================================================================

class VideoDownloader:
    """Downloads video from URL using yt-dlp."""

    def __init__(
        self,
        ffmpeg_path:  str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        output_dir:   str = "downloads",
    ) -> None:
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg   = self._find_bin(ffmpeg_path,  script_dir)
        self.ffprobe  = self._find_bin(ffprobe_path, script_dir)
        self.output_dir = output_dir

        # Env-var driven config
        self.server_download     = os.environ.get("BOOSTA_SERVER_DOWNLOAD", "")
        self.access_token        = os.environ.get("BOOSTA_ACCESS_TOKEN", "")
        self.kick_proxy_url      = os.environ.get("KICK_PROXY_URL", "")
        self.youtube_proxy_url   = os.environ.get("YOUTUBE_PROXY_URL", "")
        self.pot_provider_url    = os.environ.get("YOUTUBE_POT_PROVIDER_URL", "")
        self.cookies_file        = self._resolve_cookies()

        # aria2c
        self.aria2c = shutil.which("aria2c") or ""
        if self.aria2c:
            print(f"[Downloader] aria2c found, enabling multi-connection download...")

        if not YT_DLP_AVAILABLE:
            print("[WARN] yt-dlp not installed — download will fail. "
                  "Run: pip install yt-dlp")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def download(
        self,
        url:               str,
        output_dir:        Optional[str] = None,
        quality:           str = "best",
        progress_callback: Optional[Callable] = None,
        audio_format_id:   Optional[str] = None,
        format_id:         Optional[str] = None,
    ) -> str:
        """
        Downloads video from URL using yt-dlp.
        Returns path to downloaded .mp4 file.
        """
        if not YT_DLP_AVAILABLE:
            raise RuntimeError(
                "yt-dlp is not installed. Run: pip install yt-dlp"
            )

        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        video_id, platform = self._extract_video_id_and_platform(url)

        if progress_callback:
            progress_callback({"status": "starting", "percent": 0.0})

        # Determine format string
        req_format = self._build_format_string(quality, format_id, audio_format_id)
        print(f"[Downloader] Final Format String: {req_format}")

        # Build outtmpl
        safe_title   = f"{platform}_{video_id}" if video_id else "video"
        outtmpl      = os.path.join(out_dir, f"{safe_title}_%(id)s.%(ext)s")

        # Build yt-dlp options
        ydl_opts = self._build_ydl_opts(
            outtmpl          = outtmpl,
            req_format       = req_format,
            progress_callback= progress_callback,
            url              = url,
            platform         = platform,
        )

        # ── Attempt 1: yt-dlp direct ────────────────────────────────────────
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                result_path = self._find_output_file(info, out_dir, video_id)
                if result_path:
                    if progress_callback:
                        progress_callback({"status": "finished", "percent": 100.0,
                                           "path": result_path})
                    print(f"[Downloader] Download completed: {result_path}")
                    return result_path
        except Exception as exc:
            print(f"[Downloader] yt-dlp error: {exc}")

        # ── Attempt 2: platform-specific proxy fallback ──────────────────────
        if platform == "youtube" and (self.server_download or self.youtube_proxy_url):
            print(f"[Downloader] Trying YouTube proxy fallback...")
            try:
                result = self._download_youtube_via_proxy(
                    url, out_dir, quality, progress_callback
                )
                if result:
                    if progress_callback:
                        progress_callback({"status": "finished", "percent": 100.0,
                                           "path": result})
                    return result
            except Exception as exc:
                print(f"[YouTubeProxy] Error: {exc}")

        if platform == "kick" and self.kick_proxy_url:
            print(f"[Downloader] Trying Kick proxy fallback...")
            try:
                result = self._download_kick_via_proxy(
                    url, out_dir, quality, progress_callback
                )
                if result:
                    if progress_callback:
                        progress_callback({"status": "finished", "percent": 100.0,
                                           "path": result})
                    return result
            except Exception as exc:
                print(f"[KickProxy] Error: {exc}")

        # ── Attempt 3: Any4k API ─────────────────────────────────────────────
        print(f"[Downloader] Falling back to Any4k...")
        try:
            result = self._download_via_any4k(url, out_dir, quality, progress_callback)
            if result:
                if progress_callback:
                    progress_callback({"status": "finished", "percent": 100.0,
                                       "path": result})
                return result
        except Exception as exc:
            print(f"[Any4k] Exception: {exc}")

        if progress_callback:
            progress_callback({"status": "error",
                               "error": "All download methods failed"})
        raise RuntimeError(
            f"All download methods failed for: {url}\n"
            "Check COOKIES_FILE, YOUTUBE_PROXY_URL, KICK_PROXY_URL env vars."
        )

    def get_info(self, url: str) -> Dict:
        """Fetches video metadata and available formats without downloading."""
        if not YT_DLP_AVAILABLE:
            raise RuntimeError("yt-dlp not installed")

        ydl_opts_base = self._build_base_opts(url=url, platform=None)
        ydl_opts_base.update({
            "quiet":     True,
            "no_warnings": True,
            "skip_download": True,
        })
        try:
            with yt_dlp.YoutubeDL(ydl_opts_base) as ydl:
                ydl_meta = ydl.extract_info(url, download=False)
                return ydl_meta or {}
        except Exception as exc:
            print(f"[Downloader] Error fetching info: {exc}")
            return {}

    # -----------------------------------------------------------------------
    # Platform detection
    # -----------------------------------------------------------------------

    def _extract_video_id_and_platform(self, url: str) -> Tuple[str, str]:
        """Extract (video_id, platform) from URL."""
        url_lower = url.lower()

        # YouTube
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            video_id = ""
            # Watch URL
            m = _RE_YT_WATCH.search(url)
            if m:
                video_id = m.group(1)
            # Short URL
            if not video_id:
                m = _RE_YT_SHORT.search(url)
                if m:
                    video_id = m.group(1)
            # Shorts
            if not video_id:
                m = _RE_YT_SHORTS_PATH.search(url)
                if m:
                    video_id = m.group(1)
            return video_id or "youtube_video", "youtube"

        # Kick
        if "kick.com" in url_lower:
            parts = [p for p in url.split("/") if p and "kick.com" not in p]
            video_id = parts[-1] if parts else "kick_video"
            return video_id, "kick"

        # Twitch
        if "twitch.tv" in url_lower:
            parts = url.rstrip("/").split("/")
            video_id = parts[-1] if parts else "twitch_video"
            return video_id, "twitch"

        # Generic
        return "video", "generic"

    # -----------------------------------------------------------------------
    # Format selection
    # -----------------------------------------------------------------------

    def _filter_formats(self, formats: List[Dict], quality: str) -> List[Dict]:
        """
        Filter and deduplicate format list by quality.
        We want to present clean options to the user (e.g. 1080p, 720p).
        """
        if not formats:
            return []

        height_map = {
            "1080p": 1080, "1080": 1080,
            "720p":  720,  "720":  720,
            "480p":  480,  "480":  480,
            "360p":  360,  "360":  360,
        }
        max_height = height_map.get(quality.lower())

        seen_resolutions: set = set()
        clean_formats: List[Dict] = []

        for fmt in sorted(formats, key=lambda f: (f.get("height") or 0), reverse=True):
            h = fmt.get("height") or 0
            if max_height and h > max_height:
                continue
            res_key = (h, fmt.get("vcodec", ""), fmt.get("ext", ""))
            if res_key in seen_resolutions:
                continue
            seen_resolutions.add(res_key)
            clean_formats.append(fmt)

        return clean_formats

    def _build_format_string(
        self,
        quality:       str,
        format_id:     Optional[str],
        audio_format_id: Optional[str],
    ) -> str:
        """Build the yt-dlp format string from quality/format_id."""
        # User provided explicit format
        if format_id:
            if audio_format_id:
                combined = f"{format_id}+{audio_format_id}"
                print(f"[Downloader] Using user-provided combined format: {combined}")
                return combined
            return format_id

        # Named quality
        q = quality.strip().lower()
        return _FORMAT_MAP.get(quality, _FORMAT_MAP.get(q, _FORMAT_MAP["best"]))

    # -----------------------------------------------------------------------
    # Audio tracks
    # -----------------------------------------------------------------------

    def _get_audio_tracks(self, info: Dict) -> List[Dict]:
        """
        Get available audio tracks from yt-dlp info dict.
        Prioritises: Original Audio > clean non-dub > any.
        """
        formats = info.get("formats", [])
        audio_formats = [
            f for f in formats
            if f.get("acodec") and f.get("acodec") != "none"
            and (not f.get("vcodec") or f.get("vcodec") == "none")
        ]
        if not audio_formats:
            return []

        audio_tracks: List[Dict] = []
        seen_languages: set = set()

        # Best 'Original Audio' track
        best_original: Optional[Dict] = None
        best_clean:    Optional[Dict] = None
        best_any:      Optional[Dict] = None
        original_candidates: List[Dict] = []
        clean_candidates:    List[Dict] = []

        for fmt in sorted(audio_formats,
                           key=lambda f: f.get("filesize", 0) or 0,
                           reverse=True):
            label     = (fmt.get("format_note") or "").lower()
            lang_code = fmt.get("language") or ""
            lang_name = self._get_language_name(lang_code)

            is_original = "original" in label
            is_dub      = "dub" in label or "dubbed" in label

            if is_original:
                original_candidates.append(fmt)
            elif not is_dub and lang_code not in ("", None):
                clean_candidates.append(fmt)

            if lang_code and lang_code not in seen_languages:
                seen_languages.add(lang_code)
                audio_tracks.append({
                    "format_id":  fmt.get("format_id", ""),
                    "language":   lang_code,
                    "lang_name":  lang_name,
                    "label":      fmt.get("format_note", lang_name),
                    "filesize":   fmt.get("filesize", 0),
                })

        # Select best
        if original_candidates:
            best_original = original_candidates[0]
            print(f"[Downloader] ✅ Selected ORIGINAL track: "
                  f"{best_original.get('format_note', '')} / "
                  f"{self._get_language_name(best_original.get('language', ''))}")
        elif clean_candidates:
            best_clean = clean_candidates[0]
            print(f"[Downloader] ⚠️  No 'Original' found. "
                  f"Selected clean track (non-dub/non-default): "
                  f"{best_clean.get('format_note', '')}")
        else:
            best_any = audio_formats[0] if audio_formats else None
            if best_any:
                print(f"[Downloader] ⚠️ Check failed. Fallback to best available audio: "
                      f"{best_any.get('format_note', '')}")

        return audio_tracks

    def _get_language_name(self, lang_code: str) -> str:
        """Convert language code to human-readable name."""
        if not lang_code:
            return "Unknown"
        code = lang_code.lower().split("-")[0]
        return _LANG_MAP.get(code, lang_code.upper())

    # -----------------------------------------------------------------------
    # POT provider (YouTube token)
    # -----------------------------------------------------------------------

    def _apply_pot_provider(self, opts: Dict) -> Dict:
        """Apply POT provider settings for YouTube if configured."""
        pot_url = self.pot_provider_url
        if not pot_url:
            return opts

        opts.setdefault("extractor_args", {})
        opts["extractor_args"].setdefault("youtube", {})
        opts["extractor_args"]["youtube"]["pot_provider"] = "youtubepot-bgutilhttp"
        opts["extractor_args"]["youtube"]["pot_url"]      = pot_url

        # Also configure player clients
        opts["extractor_args"]["youtube"].setdefault(
            "player_client", ["android", "web_creator"]
        )
        return opts

    # -----------------------------------------------------------------------
    # YouTube proxy (boosta.pro server)
    # -----------------------------------------------------------------------

    def _download_youtube_via_proxy(
        self,
        url:               str,
        output_dir:        str,
        quality:           str,
        progress_callback: Optional[Callable],
    ) -> Optional[str]:
        """
        Download YouTube video via Boosta proxy server.
        Proxy returns separate video+audio URLs, we merge with ffmpeg
        for high quality.
        """
        if not REQUESTS_AVAILABLE:
            return None

        video_id, _ = self._extract_video_id_and_platform(url)
        proxy_url   = self.server_download or _SERVER_YT_ENDPOINT
        headers     = {"User-Agent": _UA_WIN}

        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
            headers["X-Client"]      = "boosta"

        # Use proxy for metadata if YOUTUBE_PROXY_URL set
        if self.youtube_proxy_url:
            print(f"[Downloader] Using proxy for YouTube metadata...")
            headers["proxy"] = self.youtube_proxy_url

        payload = {
            "video_id": video_id,
            "url":      url,
            "quality":  quality,
        }
        print(f"[YouTubeProxy] Requesting server download: {proxy_url}")
        if progress_callback:
            progress_callback({"status": "downloading", "percent": 5.0})

        try:
            resp = _requests.post(proxy_url, json=payload, headers=headers, timeout=60)
            print(f"[YouTubeProxy] Proxy returned {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[YouTubeProxy] Proxy error: {exc}")
            return None

        # Parse response
        video_url    = data.get("video_url") or data.get("video")
        audio_url    = data.get("audio_url") or data.get("audio")
        hls_playlist = data.get("hls_playlist") or data.get("hls")
        combined_url = data.get("combined_url") or data.get("url")
        server_file  = data.get("file_url") or data.get("download_url")

        safe_name = f"yt_{video_id}_{quality}.mp4"
        out_path  = os.path.join(output_dir, safe_name)

        # Option A: server pre-merged file
        if server_file:
            print(f"[YouTubeProxy] Got server file URL")
            print(f"[YouTubeProxy] Downloading server file...")
            try:
                result = self._download_url_to_file(server_file, out_path, progress_callback)
                if result:
                    print(f"[YouTubeProxy] Download completed: {result}")
                    return result
                else:
                    print(f"[YouTubeProxy] Server file download failed:")
            except Exception as exc:
                print(f"[YouTubeProxy] Server file download failed: {exc}")

        # Option B: separate video + audio → ffmpeg merge
        if video_url and audio_url:
            print(f"[YouTubeProxy] Got URLs: video={video_url[:60]}..., audio=...")
            print(f"[YouTubeProxy] Downloading and merging video+audio with ffmpeg...")
            if progress_callback:
                progress_callback({"status": "downloading", "percent": 30.0})

            tmp_video = tempfile.mktemp(suffix="_video.mp4")
            tmp_audio = tempfile.mktemp(suffix="_audio.m4a")
            try:
                self._download_url_to_file(video_url, tmp_video)
                self._download_url_to_file(audio_url, tmp_audio)
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", tmp_video,
                    "-i", tmp_audio,
                    "-c:v", "copy", "-c:a", "aac",
                    "-loglevel", "error",
                    out_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                if result.returncode == 0 and os.path.exists(out_path):
                    print(f"[YouTubeProxy] Merge completed: {out_path}")
                    return out_path
                else:
                    print(f"[YouTubeProxy] ffmpeg failed: {result.stderr.decode()[:200]}")
            except Exception as exc:
                print(f"[YouTubeProxy] ffmpeg failed: {exc}")
            finally:
                for tmp in (tmp_video, tmp_audio):
                    try: os.remove(tmp)
                    except OSError: pass

        # Option C: combined URL (single stream)
        if combined_url:
            print(f"[Downloader] Using user-provided combined format: {combined_url[:60]}")
            result = self._download_url_to_file(combined_url, out_path, progress_callback)
            if result:
                return result

        # Option D: HLS / m3u8 via ffmpeg
        if hls_playlist:
            print(f"[YouTubeProxy] Downloading HLS manifest with ffmpeg...")
            if progress_callback:
                progress_callback({"status": "downloading", "percent": 20.0})
            try:
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", hls_playlist,
                    "-c", "copy",
                    "-bsf:a", "aac_adtstoasc",
                    "-loglevel", "error",
                    out_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=600)
                if result.returncode == 0 and os.path.exists(out_path):
                    print(f"[YouTubeProxy] HLS download completed: {out_path}")
                    return out_path
                else:
                    print(f"[YouTubeProxy] HLS ffmpeg failed: {result.stderr.decode()[:200]}")
                    print(f"Server HLS download failed")
            except Exception as exc:
                print(f"[YouTubeProxy] HLS ffmpeg failed: {exc}")

        print(f"Server download failed: no usable URLs in response")
        return None

    # -----------------------------------------------------------------------
    # Kick proxy (boosta.pro server + ffmpeg m3u8)
    # -----------------------------------------------------------------------

    def _download_kick_via_proxy(
        self,
        url:               str,
        output_dir:        str,
        quality:           str,
        progress_callback: Optional[Callable],
    ) -> Optional[str]:
        """
        Download Kick video using our proxy API to bypass Cloudflare.
        The proxy gets metadata, then we download m3u8 directly.
        """
        if not REQUESTS_AVAILABLE:
            return None

        video_id, _ = self._extract_video_id_and_platform(url)
        proxy_url   = self.kick_proxy_url or _SERVER_KICK_ENDPOINT
        headers     = {"User-Agent": _UA_WIN}

        # Try curl_cffi impersonation first (needed on servers without browser)
        try:
            import curl_cffi.requests as curl_req
            print(f"[Downloader] Using browser impersonation for Kick")
            kick_client = curl_req
        except ImportError:
            print(f"[Downloader] Warning: curl_cffi not installed, Kick may fail on servers")
            kick_client = None

        # Fetch metadata via proxy
        meta_url = f"{proxy_url}?video_id={video_id}"
        print(f"[KickProxy] Fetching metadata via proxy: {meta_url}")
        print(f"[Downloader] Fetching metadata for audio selection: {url}")

        try:
            if kick_client:
                resp = kick_client.get(meta_url, headers=headers,
                                       impersonate="chrome", timeout=30)
            else:
                resp = _requests.get(meta_url, headers=headers, timeout=30)
            print(f"[KickProxy] Proxy returned {resp.status_code}")
            resp.raise_for_status()
            meta = resp.json()
        except Exception as exc:
            print(f"[KickProxy] Proxy error: {exc}")
            return None

        source_url = meta.get("source_url") or meta.get("m3u8_url") or meta.get("url")
        if not source_url:
            print(f"[KickProxy] No source URL in response")
            return None

        # Check if m3u8 / HLS
        is_hls = ".m3u8" in source_url or "m3u8" in source_url.lower()
        if is_hls:
            print(f"[KickProxy] Got m3u8: {source_url[:80]}")
        else:
            print(f"[KickProxy] Got source: {source_url[:80]}")

        safe_name = self._safe_filename(f"kick_{video_id}.mp4")
        out_path  = os.path.join(output_dir, safe_name)

        print(f"[KickProxy] Downloading via ffmpeg...")
        if progress_callback:
            progress_callback({"status": "downloading", "percent": 10.0})
        try:
            cmd = [self.ffmpeg, "-y"]
            if is_hls:
                cmd += ["-i", source_url, "-c", "copy", "-bsf:a", "aac_adtstoasc"]
            else:
                cmd += ["-i", source_url, "-c", "copy"]
            cmd += ["-loglevel", "error", out_path]
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode == 0 and os.path.exists(out_path):
                if progress_callback:
                    progress_callback({"status": "finished", "percent": 100.0})
                print(f"[KickProxy] Download completed: {out_path}")
                return out_path
            else:
                err = result.stderr.decode(errors="replace")[:300]
                print(f"[KickProxy] ffmpeg failed: {err}")
        except Exception as exc:
            print(f"[KickProxy] Error: {exc}")
        return None

    # -----------------------------------------------------------------------
    # Any4k API fallback
    # -----------------------------------------------------------------------

    def _download_via_any4k(
        self,
        url:               str,
        output_dir:        str,
        quality:           str,
        progress_callback: Optional[Callable],
    ) -> Optional[str]:
        """
        Download video using Any4k API (v1/dlp).
        Documentation: https://any4k.com/api
        """
        if not REQUESTS_AVAILABLE:
            return None

        headers = {
            "User-Agent":    _UA_MAC,
            "Content-Type":  "application/json",
            "X-Client":      "boosta",
        }

        # Android device fingerprint (decoded from binary)
        dl_payload = {
            "BundleId": _ANY4K_BUNDLE,
            "AppVer":   "1.2.0",
            "DeviceId": str(uuid.uuid4()),
            "SysVer":   "Android 12",
            "url":      url,
            "quality":  quality,
        }

        # Step 1: check available formats
        print(f"[Any4k] Checking available formats for: {url}")
        try:
            check_resp = _requests.post(
                _ANY4K_CHECK, json=dl_payload, headers=headers, timeout=30
            )
        except Exception as exc:
            print(f"[Any4k] Exception: {exc}")
            return None

        if check_resp.status_code != 200:
            print(f"[Any4k] Check failed: HTTP {check_resp.status_code}")
            return None

        try:
            check_data = check_resp.json()
        except Exception:
            print(f"[Any4k] Check failed:")
            return None

        err_code = check_data.get("err_code") or check_data.get("error")
        if err_code:
            err_msg = check_data.get("err_msg") or check_data.get("message", "")
            print(f"[Any4k] API Error: {err_code} — {err_msg}")
            return None

        formats = check_data.get("formats") or check_data.get("data", {}).get("formats", [])
        if not formats:
            print(f"[Any4k] No downloadable formats found")
            return None

        # Pick best format by quality
        def get_res(fmt: Dict) -> int:
            h = fmt.get("height") or 0
            return int(h)

        selected_format = sorted(formats, key=get_res, reverse=True)[0]
        print(f"[Any4k] Selected format: {selected_format}")
        dl_payload["format_id"] = selected_format.get("format_id", "")

        # Step 2: request download stream (with retry)
        print(f"[Any4k] Requesting stream from API...")
        dl_resp = None
        max_retries = 5

        for attempt in range(max_retries):
            try:
                dl_resp = _requests.post(
                    _ANY4K_DOWNLOAD, json=dl_payload, headers=headers, timeout=60
                )
            except Exception as exc:
                print(f"[Any4k] Exception: {exc}")
                return None

            if dl_resp.status_code == 500:
                wait = 10 + attempt * 5
                print(f"[Any4k] Server busy (500) (attempt {attempt+1}/{max_retries}), "
                      f"waiting {wait}s...")
                time.sleep(wait)
                continue

            try:
                err_data = dl_resp.json()
                if isinstance(err_data, dict):
                    err_code = err_data.get("err_code") or err_data.get("error")
                    if err_code == "high_demand":
                        wait = 15 + attempt * 10
                        print(f"[Any4k] High demand (attempt {attempt+1}/{max_retries}), "
                              f"waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    if err_code:
                        err_msg = err_data.get("err_msg", "")
                        print(f"[Any4k] Download returned JSON Error: {err_code} — {err_msg}")
                        return None
            except Exception:
                pass  # Not JSON — might be binary stream, continue

            break

        if not dl_resp or dl_resp.status_code != 200:
            return None

        # Try to parse as JSON first (may contain download URL)
        try:
            dl_data = dl_resp.json()
            if isinstance(dl_data, dict):
                source_url = (dl_data.get("url") or dl_data.get("source_url")
                              or dl_data.get("download_url"))
                if source_url:
                    safe_name = self._safe_filename(f"any4k_{int(time.time())}.mp4")
                    out_path  = os.path.join(output_dir, safe_name)
                    result    = self._download_url_to_file(source_url, out_path, progress_callback)
                    if result:
                        print(f"[Any4k] Download complete: {result}")
                        return result
                print(f"[Any4k] Download returned JSON but couldn't parse it.")
                return None
        except Exception:
            pass

        # Binary stream response
        safe_name = self._safe_filename(f"any4k_{int(time.time())}.mp4")
        out_path  = os.path.join(output_dir, safe_name)
        try:
            with open(out_path, "wb") as fh:
                for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"[Any4k] Download complete: {out_path}")
                return out_path
        except Exception as exc:
            print(f"[Any4k] Exception: {exc}")

        return None

    # -----------------------------------------------------------------------
    # yt-dlp options builder
    # -----------------------------------------------------------------------

    def _build_base_opts(self, url: Optional[str], platform: Optional[str]) -> Dict:
        """Build base yt-dlp options (shared across download/info)."""
        opts: Dict = {
            "merge_output_format": "mp4",
            "noplaylist":          True,
            "quiet":               False,
            "no_warnings":         False,
            "overwrites":          True,
            "retries":             5,
            "http_chunk_size":     10 * 1024 * 1024,  # 10 MB
            "buffersize":          1024 * 1024,
            "concurrent_fragment_downloads": 4,
        }

        # ffmpeg location
        if os.path.exists(self.ffmpeg):
            opts["ffmpeg_location"] = os.path.dirname(self.ffmpeg)

        # cookies
        if self.cookies_file:
            print(f"[Downloader] Using cookies from: {self.cookies_file}")
            opts["cookiefile"] = self.cookies_file

        # YouTube proxy
        if platform == "youtube" and self.youtube_proxy_url:
            print(f"[Downloader] Using proxy for YouTube download...")
            opts["proxy"] = self.youtube_proxy_url

        # aria2c
        if self.aria2c:
            opts["external_downloader"]      = "aria2c"
            opts["external_downloader_args"] = {
                "aria2c": ["-x", "16", "-s", "16", "-k", "1M"]
            }

        return opts

    def _build_ydl_opts(
        self,
        outtmpl:           str,
        req_format:        str,
        progress_callback: Optional[Callable],
        url:               Optional[str],
        platform:          Optional[str],
    ) -> Dict:
        """Build full yt-dlp options dict for a download."""
        opts = self._build_base_opts(url=url, platform=platform)
        opts.update({
            "outtmpl":  outtmpl,
            "format":   req_format,
        })

        # Progress hook
        def progress_hook(d: Dict) -> None:
            if not progress_callback:
                return
            status = d.get("status", "")
            if status == "downloading":
                _percent_str = d.get("_percent_str", "0.0%").strip()
                try:
                    percent = float(_percent_str.replace("%", ""))
                except ValueError:
                    percent = 0.0
                _speed_str = d.get("_speed_str", "") or ""
                _eta_str   = d.get("_eta_str", "")   or ""
                progress_callback({
                    "status":           "downloading",
                    "percent":          percent,
                    "downloaded_bytes": d.get("downloaded_bytes", 0),
                    "total_bytes":      d.get("total_bytes") or d.get("total_bytes_estimate", 0),
                    "speed":            d.get("speed", 0),
                    "eta":              d.get("eta", 0),
                    "_speed_str":       _speed_str,
                    "_eta_str":         _eta_str,
                })
            elif status == "finished":
                print(f"[download] Finished: {d.get('filename', '')}")
                progress_callback({"status": "processing", "percent": 100.0})

        opts["progress_hooks"] = [progress_hook]

        # YouTube-specific
        if platform == "youtube":
            opts = self._apply_pot_provider(opts)
            # server-side download
            if self.server_download:
                print(f"[Downloader] Using server-side YouTube download...")

        return opts

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_cookies(self) -> str:
        """Resolve cookies file from env or default location."""
        cookies_path = os.environ.get("COOKIES_FILE", "")
        if cookies_path and os.path.exists(cookies_path):
            return cookies_path
        # Check default location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default    = os.path.join(script_dir, "cookies.txt")
        if os.path.exists(default):
            return default
        return ""

    def _find_output_file(
        self,
        info:       Optional[Dict],
        output_dir: str,
        video_id:   str,
    ) -> Optional[str]:
        """Find the actual downloaded file from yt-dlp info dict."""
        if info is None:
            return None

        # Direct filepath from info
        if info.get("requested_downloads"):
            for dl in info["requested_downloads"]:
                path = dl.get("filepath") or dl.get("filename")
                if path and os.path.exists(path):
                    return path

        # Look for .mp4 files in output_dir matching video_id
        potential_mp4: List[str] = []
        if os.path.isdir(output_dir):
            for fname in os.listdir(output_dir):
                if fname.endswith(".mp4") and (not video_id or video_id in fname):
                    potential_mp4.append(os.path.join(output_dir, fname))

        if potential_mp4:
            return max(potential_mp4, key=os.path.getmtime)

        return None

    def _safe_filename(self, name: str, max_len: int = 200) -> str:
        """Create a Windows-safe filename."""
        safe = _FORBIDDEN_WIN_CHARS.sub("_", name)
        safe = safe.strip(". ")
        if len(safe) > max_len:
            stem = Path(safe).stem[:max_len - 10]
            ext  = Path(safe).suffix
            safe = stem + ext
        return safe or "video.mp4"

    def _download_url_to_file(
        self,
        url:               str,
        out_path:          str,
        progress_callback: Optional[Callable] = None,
        timeout:           int = 300,
    ) -> Optional[str]:
        """Stream-download a direct URL to a file."""
        if not REQUESTS_AVAILABLE:
            return None
        try:
            headers    = {"User-Agent": _UA_WIN}
            stream_resp = _requests.get(url, headers=headers,
                                        stream=True, timeout=timeout)
            stream_resp.raise_for_status()
            total_size = int(stream_resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024

            with open(out_path, "wb") as fh:
                for chunk in stream_resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            pct = (downloaded / total_size) * 100
                            progress_callback({
                                "status":           "downloading",
                                "percent":          round(pct, 1),
                                "downloaded_bytes": downloaded,
                                "total_bytes":      total_size,
                            })

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
        except Exception as exc:
            print(f"Server file download failed: {exc}")
        return None

    @staticmethod
    def _find_bin(name: str, script_dir: str) -> str:
        """Find binary: local dir first, then PATH."""
        for candidate in (name, name + ".exe",
                          os.path.join(script_dir, name),
                          os.path.join(script_dir, name + ".exe")):
            if os.path.exists(candidate):
                return candidate
        return name


# ===========================================================================
# Module-level convenience
# ===========================================================================

def download(
    url:               str,
    output_dir:        str = "downloads",
    quality:           str = "best",
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Convenience wrapper around VideoDownloader.download().

    Args:
        url:               Video URL (YouTube, Kick, Twitch, generic).
        output_dir:        Directory to save the downloaded file.
        quality:           "best", "1080p", "720p", "480p", "360p", "audio".
        progress_callback: Called with progress dict during download.

    Returns:
        Path to the downloaded .mp4 file.
    """
    d = VideoDownloader(output_dir=output_dir)
    return d.download(url, output_dir=output_dir, quality=quality,
                      progress_callback=progress_callback)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import io
    import sys

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Boosta VideoDownloader — download video to local .mp4"
    )
    parser.add_argument("url",     help="Video URL (YouTube, Kick, Twitch, generic)")
    parser.add_argument("--output",  default="downloads", help="Output directory")
    parser.add_argument("--quality", default="best",
                        choices=["best", "1080p", "720p", "480p", "360p", "audio"],
                        help="Video quality")
    parser.add_argument("--info", action="store_true",
                        help="Print info/formats without downloading")
    args = parser.parse_args()

    def _progress(d: Dict) -> None:
        status = d.get("status", "")
        if status == "downloading":
            pct = d.get("percent", 0)
            spd = d.get("_speed_str", "")
            eta = d.get("_eta_str", "")
            print(f"\r[{pct:5.1f}%] {spd} ETA {eta}   ", end="", flush=True)
        elif status == "processing":
            print("\n[Processing...]")
        elif status == "finished":
            print(f"\n[Done] {d.get('path', '')}")
        elif status == "error":
            print(f"\n[Error] {d.get('error', '')}")

    print("=" * 60)
    print("Boosta VideoDownloader")
    print("=" * 60)
    d = VideoDownloader(output_dir=args.output)
    vid_id, platform = d._extract_video_id_and_platform(args.url)
    print(f"URL:       {args.url}")
    print(f"Platform:  {platform}")
    print(f"Video ID:  {vid_id}")
    print(f"Quality:   {args.quality}")
    print(f"Output:    {args.output}")
    print()

    if args.info:
        info = d.get_info(args.url)
        formats = info.get("formats", [])
        print(f"Title:   {info.get('title', 'N/A')}")
        print(f"Formats: {len(formats)}")
        for fmt in formats[-10:]:
            print(f"  {fmt.get('format_id','?'):12} {fmt.get('height','?'):5}p "
                  f"{fmt.get('vcodec','?'):12} {fmt.get('acodec','?'):12}")
        sys.exit(0)

    try:
        path = d.download(args.url, output_dir=args.output,
                          quality=args.quality, progress_callback=_progress)
        print(f"\nSaved: {path}")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
