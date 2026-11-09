"""
compose.py — single FFmpeg pass: base clip → [burn subtitles] → overlay layers with fade + short slide → file.

Uses only the `overlay`, `fade` and (for subtitles) `ass` filters: no drawtext, which some FFmpeg builds lack.
The audio stream is copied untouched.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.shared.crop.smart_crop_adapter import _env_encoder_args, ffmpeg_bin

from .themes import FADE_IN_SEC, FADE_OUT_SEC, INTENSITY, SLIDE_PX_1080

ASPECTS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}


@dataclass
class PlacedLayer:
    png: str
    x: int
    y: int
    start: float
    end: float
    slide_px: int


def probe(path: str) -> Dict[str, Any]:
    ffprobe = re.sub(r"ffmpeg$", "ffprobe", ffmpeg_bin())
    r = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or "25/1").split("/")
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": float(num) / float(den or 1) if float(num) else 25.0,
            "duration": float(info["format"]["duration"]), "has_audio": any(s["codec_type"] == "audio" for s in info["streams"])}


def render_base(src: str, out: str, start: float, duration: float, aspect: str) -> str:
    """Whole frame in the target aspect (16:9, 1:1): nothing is cropped away from a board or a slide."""
    W, H = ASPECTS[aspect]
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{max(0.0, start):.3f}", "-i", src,
           "-t", f"{duration:.3f}", "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"base render failed: {r.stderr[-300:]}")
    return out


def _graph(layers: List[PlacedLayer], ass_name: Optional[str], fonts_dir: Optional[str]) -> Tuple[str, str]:
    parts: List[str] = []
    label = "0:v"
    if ass_name:
        fd = (fonts_dir or "").replace("\\", "/").replace(":", "\\:").replace("'", "")
        parts.append(f"[0:v]ass=filename={ass_name}:fontsdir='{fd}'[subbed]")
        label = "subbed"
    for i, l in enumerate(layers):
        dur = l.end - l.start
        idx = i + 1
        fade_in = min(FADE_IN_SEC, dur / 3)
        fade_out = min(FADE_OUT_SEC, dur / 3)
        parts.append(f"[{idx}:v]format=rgba,fade=t=in:st=0:d={fade_in:.3f}:alpha=1,"
                     f"fade=t=out:st={dur - fade_out:.3f}:d={fade_out:.3f}:alpha=1,"
                     f"setpts=PTS+{l.start:.3f}/TB[l{i}]")
        if l.slide_px:
            y = f"{l.y}+{l.slide_px}*pow(max(0\\,1-(t-{l.start:.3f})/{fade_in:.3f})\\,3)"
        else:
            y = str(l.y)
        parts.append(f"[{label}][l{i}]overlay=x={l.x}:y='{y}':eof_action=pass:eval=frame[o{i}]")
        label = f"o{i}"
    if not parts:
        return "", "0:v"
    return ";".join(parts), label


def composite(base: str, out: str, layers: List[PlacedLayer], *, subtitle_path: Optional[str] = None,
              fonts_dir: Optional[str] = None, encoder_args: Optional[List[str]] = None) -> str:
    info = probe(base)
    cwd = None
    ass_name = None
    if subtitle_path:
        sub = Path(subtitle_path).resolve()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", sub.name):
            raise ValueError(f"unsafe subtitle file name: {sub.name}")
        cwd, ass_name = str(sub.parent), sub.name
    base_path = str(Path(base).resolve())
    out_path = str(Path(out).resolve())
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", base_path]
    for l in layers:
        cmd += ["-loop", "1", "-framerate", f"{info['fps']:.3f}", "-t", f"{l.end - l.start:.3f}", "-i", str(Path(l.png).resolve())]
    graph, last = _graph(layers, ass_name, fonts_dir)
    if graph:
        cmd += ["-filter_complex", graph, "-map", f"[{last}]"]
    else:
        cmd += ["-map", "0:v"]
    cmd += ["-map", "0:a?", *(encoder_args or _env_encoder_args() or ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]),
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"overlay composite failed: {r.stderr[-400:]}")
    return out


def slide_px(short_side: int, intensity: str = "normal") -> int:
    return int(SLIDE_PX_1080 * short_side / 1080 * INTENSITY.get(intensity, 1.0))
