"""
scripts/shared/source_normalize.py
==================================
The SONYA Desktop ingest normalization (sonya-desktop desktop/sonya_desktop/ingest/pipeline.py:
``summarize_probe`` / ``normalize_reasons`` / ``normalize_cmd``, copied unchanged) for the worker.

Desktop re-encodes a source before ANY mode runs on it when its video codec is not H.264/HEVC, its audio
codec is unusual, the frame rate is variable (phone recordings), it carries a rotation flag (phone video)
or the pixel format is not 4:2:0. Without this the worker produced a different BEEP file than the Mac mini
(AV1 copied through instead of H.264, beep_exile_180) and would decode phone uploads differently in Cut.

Encoder: the Desktop "quality" preset without a bitrate cap (NVENC p6 cq 20 / libx264 slow crf 19).
``SONYA_NORMALIZE=never`` switches it off (rollback).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FRIENDLY_VCODECS = {"h264", "hevc"}
FRIENDLY_ACODECS = {"aac", "mp3", "opus", None}
MODES = ("virality", "beep")          # the modes ported from the Mac mini (Desktop ASR_MODES)
LONG_SEC = int(os.environ.get("SONYA_LONG_SOURCE_SEC", "1200"))    # > 20 min: fast CPU fallback + watchdog (speedfix)
LONG_CPU_PRESET = os.environ.get("SONYA_NORMALIZE_LONG_CPU_PRESET", "ultrafast")
MSG_FAILED = "Не удалось подготовить видео. Попробуйте ещё раз."
MSG_NO_SPACE = "На сервере не хватило места для такого длинного видео. Попробуйте ещё раз через несколько минут."
MSG_TOO_SLOW = "Подготовка такого длинного видео заняла слишком много времени. Попробуйте ещё раз или выберите отрезок короче."


class NormalizeError(RuntimeError):
    """A re-encode that cannot finish: code + a message for the user (no hours of CPU libx264, no full disk)."""
    def __init__(self, code: str, message: str, detail: str = ""):
        super().__init__(f"{code}: {detail or message}")
        self.code, self.message = code, message


def _rate(s: Optional[str]) -> Optional[float]:
    try:
        f = Fraction(s or "0")
        return float(f) if f else None
    except (ValueError, ZeroDivisionError):
        return None


def summarize_probe(raw: Dict[str, Any]) -> Dict[str, Any]:
    streams = raw.get("streams", [])
    fmt = raw.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    try:
        duration = float(fmt.get("duration") or (v or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    rotation = 0
    if v:
        tags = v.get("tags") or {}
        if "rotate" in tags:
            rotation = int(float(tags["rotate"]))
        for sd in v.get("side_data_list") or []:
            if "rotation" in sd:
                rotation = int(float(sd["rotation"]))
    r_fps, avg_fps = _rate((v or {}).get("r_frame_rate")), _rate((v or {}).get("avg_frame_rate"))
    return {
        "duration": duration,
        "has_video": v is not None,
        "has_audio": a is not None,
        "vcodec": (v or {}).get("codec_name"),
        "acodec": (a or {}).get("codec_name"),
        "width": (v or {}).get("width"),
        "height": (v or {}).get("height"),
        "pix_fmt": (v or {}).get("pix_fmt"),
        "fps": avg_fps or r_fps,
        "vfr": bool(r_fps and avg_fps and abs(r_fps - avg_fps) / r_fps > 0.02),
        "rotation": rotation % 360,
    }


def normalize_reasons(summary: Dict[str, Any]) -> List[str]:
    reasons = []
    if summary["vcodec"] not in FRIENDLY_VCODECS:
        reasons.append(f"vcodec:{summary['vcodec']}")
    if summary["acodec"] not in FRIENDLY_ACODECS:
        reasons.append(f"acodec:{summary['acodec']}")
    if summary["vfr"]:
        reasons.append("vfr")
    if summary["rotation"]:
        reasons.append(f"rotation:{summary['rotation']}")
    if summary["pix_fmt"] and summary["pix_fmt"] not in ("yuv420p", "yuvj420p", "nv12"):
        reasons.append(f"pix_fmt:{summary['pix_fmt']}")
    return reasons


# Load test 2026-11-16: a 19-min 1440p VP9 YouTube source took 382 s to normalize (CPU VP9 decode + the clip preset p7 /
# 2-pass / lookahead, on a busy server). The normalized file is an INTERMEDIATE: decode on the GPU (NVDEC; ffmpeg falls back
# to software decode by itself when the card cannot) and a fast NVENC preset at a HIGHER quality target (cq 17 vs 19).
# SONYA_NORMALIZE_FAST=0 = the old command (clip preset, CPU decode).
def fast_enabled() -> bool:
    return os.environ.get("SONYA_NORMALIZE_FAST", "1").lower() not in ("0", "false", "off", "no")


def fast_nvenc_args() -> List[str]:
    preset = os.environ.get("SONYA_NORMALIZE_PRESET", "p4")
    return ["-c:v", "h264_nvenc", "-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", "17", "-b:v", "0",
            "-spatial-aq", "1", "-bf", "3", "-pix_fmt", "yuv420p"]


def normalize_cmd(ffmpeg: str, src: Path, dst: Path, summary: Dict[str, Any], encoder_args: List[str],
                  hw_decode: bool = False) -> List[str]:
    """CFR + yuv420p + AAC; ffmpeg autorotate bakes rotation into pixels. hw_decode: NVDEC (frames come back to system
    memory, so the fps filter / autorotate work unchanged)."""
    fps = summary.get("fps") or 30.0
    fps = min(max(round(fps, 3), 1.0), 120.0)
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            *(["-hwaccel", "cuda"] if hw_decode else []),
            "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", f"fps={fps}", *encoder_args,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(dst)]


def probe(path: str, ffprobe: str = "ffprobe") -> Dict[str, Any]:
    out = subprocess.run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr[-300:]}")
    return json.loads(out.stdout or "{}")


def normalize_source(path: str, work_dir: str, mode: str) -> Dict[str, Any]:
    """-> {"path": the file the mode should read, "reasons": [...], "normalized": bool}. Never raises for a
    source that needs nothing; a failed re-encode raises (Desktop: «Не удалось подготовить видео»)."""
    if mode not in MODES or os.environ.get("SONYA_NORMALIZE", "auto").lower() == "never":
        return {"path": path, "reasons": [], "normalized": False}
    ff = os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    fp = os.path.join(os.path.dirname(ff), "ffprobe") if os.path.dirname(ff) else "ffprobe"
    summary = summarize_probe(probe(path, fp))
    reasons = normalize_reasons(summary)
    if not reasons:
        return {"path": path, "reasons": [], "normalized": False}
    from scripts.shared.cut_profile import encoder_args, encoder_family, run_encode
    long_src = (summary.get("duration") or 0) > LONG_SEC
    cpu_preset = LONG_CPU_PRESET if long_src else None
    family = encoder_family(ff)
    args = encoder_args(family, bitrate=None, cpu_preset=cpu_preset)
    fast = fast_enabled() and family == "nvenc"
    if fast:
        args = fast_nvenc_args()
    dst = Path(work_dir) / (Path(path).stem + ".normalized.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # disk first: the re-encode writes about as much as the source (2026-11-15: a 3.7 GB VP9 filled a 30 GB disk)
    need = int(os.path.getsize(path) * 1.3) + (512 << 20) if os.path.exists(path) else 0
    free = shutil.disk_usage(dst.parent).free
    if need and free < need:
        raise NormalizeError("NORMALIZE_NO_SPACE", MSG_NO_SPACE, f"free {free >> 20} MB < need {need >> 20} MB")
    # watchdog for long sources only: CPU ultrafast is ~5-10x realtime; never hours on one job
    timeout = max(600.0, 0.5 * summary["duration"]) if long_src else None
    logger.info("[normalize] %s: re-encoding (%s)%s", Path(path).name, ", ".join(reasons),
                f" long={summary['duration']:.0f}s cpu_preset={cpu_preset} timeout={timeout:.0f}s" if long_src else "")
    r, used, sec = run_encode(lambda a: normalize_cmd(ff, Path(path), dst, summary, a, hw_decode=fast), args,
                              what="normalize", timeout=timeout, cpu_preset=cpu_preset)
    logger.info("[normalize] done in %.0f s (%s%s) rc=%s", sec, used, ", fast nvdec" if fast else "", r.returncode)
    if r.returncode != 0:
        err = (r.stderr or "")[-400:]
        dst.unlink(missing_ok=True)
        if r.returncode == 124:
            raise NormalizeError("NORMALIZE_TIMEOUT", MSG_TOO_SLOW, err)
        if "No space left" in err or shutil.disk_usage(dst.parent).free < (256 << 20):
            raise NormalizeError("NORMALIZE_NO_SPACE", MSG_NO_SPACE, err)
        raise NormalizeError("NORMALIZE_FAILED", MSG_FAILED, err)
    return {"path": str(dst), "reasons": reasons, "normalized": True}
