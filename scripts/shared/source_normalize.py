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
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FRIENDLY_VCODECS = {"h264", "hevc"}
FRIENDLY_ACODECS = {"aac", "mp3", "opus", None}
MODES = ("virality", "beep")          # the modes ported from the Mac mini (Desktop ASR_MODES)


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


def normalize_cmd(ffmpeg: str, src: Path, dst: Path, summary: Dict[str, Any], encoder_args: List[str]) -> List[str]:
    """CFR + yuv420p + AAC; ffmpeg autorotate bakes rotation into pixels."""
    fps = summary.get("fps") or 30.0
    fps = min(max(round(fps, 3), 1.0), 120.0)
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
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
    from scripts.shared.cut_profile import encoder_args, encoder_family
    args = encoder_args(encoder_family(ff), bitrate=None)
    dst = Path(work_dir) / (Path(path).stem + ".normalized.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[normalize] %s: re-encoding (%s)", Path(path).name, ", ".join(reasons))
    from scripts.shared import nvenc_guard            # GeForce NVENC session cap -> CPU, never an error
    r = nvenc_guard.run(normalize_cmd(ff, Path(path), dst, summary, args), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"NORMALIZE_FAILED: {r.stderr[-400:]}")
    return {"path": str(dst), "reasons": reasons, "normalized": True}
