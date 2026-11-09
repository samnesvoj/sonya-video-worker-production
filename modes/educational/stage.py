"""
stage.py — how a lecture frame becomes 9:16 in Educational v4.

    stage  the whole 16:9 frame (slide, board, screen) on the template's background with rounded corners: nothing the
           lecturer writes or shows is cropped away. Title above, subtitles below — on the stage, not on the picture.
    crop   a talking head big enough to fill a vertical frame: the existing smart crop (compose_vertical_clip).

choose() decides per clip from YuNet faces (scripts/shared/crop/clip_analysis): crop only when one large face is on
screen most of the time; a board, a slide or a small lecturer at the blackboard → stage.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

W, H = 1080, 1920
SIDE = 36                  # stage margin left/right at 1080
RADIUS = 28
VIDEO_TOP = 0.355          # share of the height where the picture starts (title above, subtitles below)
BIG_FACE = 0.2             # face height / frame height that counts as a talking head
TALKING_HEAD_SHARE = 0.6


def choose(src: str, start: float, duration: float, layout_mode: str = "unknown") -> Tuple[str, Dict[str, Any]]:
    if layout_mode == "preserve_screen":
        return "stage", {"why": "layout:screen"}
    if layout_mode == "person_focus":
        return "crop", {"why": "layout:person"}
    try:
        from scripts.shared.crop.clip_analysis import analyze_clip
        a = analyze_clip(src, start, min(duration, 60.0), face_fps=1.0)
    except Exception as exc:  # noqa: BLE001 — no detector: the whole frame is the safe choice for a lesson
        return "stage", {"why": f"no_face_analysis:{type(exc).__name__}"}
    h = float(a["meta"]["height"]) or 1.0
    samples = a.get("samples") or []
    big = sum(1 for s in samples if any(f[3] / h >= BIG_FACE for f in s.get("faces") or []))
    share = big / max(1, len(samples))
    return ("crop" if share >= TALKING_HEAD_SHARE else "stage"), {"why": "faces", "big_face_share": round(share, 2),
                                                                   "samples": len(samples)}


def video_rect(src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    w = W - 2 * SIDE
    h = int(round(w * src_h / max(1, src_w) / 2)) * 2
    h = min(h, int(H * 0.5))
    w2 = int(round(h * src_w / max(1, src_h) / 2)) * 2
    return (W - w2) // 2, int(H * VIDEO_TOP), w2, h


def _mask(color: Tuple[int, int, int], rect: Tuple[int, int, int, int], path: Path) -> str:
    """Stage-coloured frame with a transparent rounded hole for the picture (supersampled corners)."""
    k = 3
    x, y, w, h = rect
    im = Image.new("RGBA", (W * k, H * k), color + (255,))
    ImageDraw.Draw(im).rounded_rectangle((x * k, y * k, (x + w) * k - 1, (y + h) * k - 1), RADIUS * k,
                                         fill=(0, 0, 0, 0))
    im.resize((W, H), Image.LANCZOS).save(path)
    return str(path)


def render_stage(src: str, out: str, start: float, duration: float, color: Tuple[int, int, int],
                 work_dir: Optional[str] = None) -> Dict[str, Any]:
    from scripts.shared.crop.smart_crop_adapter import _env_encoder_args, ffmpeg_bin
    from modes.educational.overlays.compose import probe
    info = probe(src)
    rect = video_rect(info["width"], info["height"])
    x, y, w, h = rect
    work = Path(work_dir or Path(out).parent)
    work.mkdir(parents=True, exist_ok=True)
    mask = _mask(color, rect, work / (Path(out).stem + ".stage.png"))
    hexc = "0x%02X%02X%02X" % color
    graph = (f"color=c={hexc}:s={W}x{H}:r={info['fps']:.3f}[bg];"
             f"[0:v]scale={w}:{h}:flags=lanczos,setsar=1[v];"
             f"[bg][v]overlay={x}:{y}:shortest=1[s];[s][1:v]overlay=0:0,format=yuv420p[o]")
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{max(0.0, start):.3f}",
           "-i", src, "-loop", "1", "-i", mask, "-t", f"{duration:.3f}", "-filter_complex", graph, "-map", "[o]",
           "-map", "0:a?", *(_env_encoder_args() or ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]),
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    Path(mask).unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage render failed: {r.stderr[-300:]}")
    return {"layout": "stage", "video_rect": list(rect)}
