"""
quality.py - «Качество нарезки» (owner 2026-11-20, Presets app on the site).

  standard (default, unchanged): 1080x1920, the Desktop encoder recipe under an 8 Mb/s cap, source <= 1440p
                                 (> 20 min: 1080p H.264, speedfix).
  high («Высокое — 2K»):         1440x2560, the same quality encoder recipe under a 24 Mb/s cap (H.264 high: plays
                                 everywhere incl. Telegram; HEVC would not preview in every Telegram client), source up
                                 to 2160p for any length. Costs HIGH_UNITS processings (API: create_job_with_quota).

Only where the whole picture comes from the Cut renderer (smart_crop_adapter: crop graph + ASS subtitles, both
resolution-independent): Вирал, Сторителлинг, Хуки. Streamer layouts, Educational overlays and the Trailer are drawn in
1080-pixel coordinates - for them the job stays standard (the site does not offer 2K there; the API charges 1 unit).

The worker runs one job per process: ``apply(params)`` at the start of every job sets the output size and the
encoder cap for this process and puts both back for a standard job. SONYA_QUALITY_HIGH=off -> every job standard.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

STANDARD, HIGH = "standard", "high"
SIZES = {STANDARD: (1080, 1920), HIGH: (1440, 2560)}
HIGH_BITRATE = os.environ.get("SONYA_HIGH_BITRATE", "24M")
HIGH_SOURCE_HEIGHT = 2160
HIGH_UNITS = 3                      # same number as the API (scripts/quality.py there) and the site
HIGH_MODES = ("virality",)
HIGH_CLIP_TYPES = ("", "viral", "virality", "storytelling", "stories", "story", "hooks")

_BASE: Dict[str, Optional[str]] = {}          # SONYA_RENDER_ARGS as the process had it before the first apply()


def requested(params: Optional[Mapping[str, Any]]) -> str:
    """What the job asks for: params.quality, else params.settings.quality; anything unknown = standard."""
    p = params or {}
    s = p.get("settings") if isinstance(p.get("settings"), Mapping) else {}
    q = str(p.get("quality") or s.get("quality") or STANDARD).strip().lower()
    return HIGH if q in ("high", "2k", "hq") else STANDARD


def supported(mode: str, params: Optional[Mapping[str, Any]]) -> bool:
    clip = str((params or {}).get("clipType") or "").strip().lower()
    return mode in HIGH_MODES and clip in HIGH_CLIP_TYPES


def resolve(mode: str, params: Optional[Mapping[str, Any]]) -> str:
    """The quality this job renders at."""
    if os.environ.get("SONYA_QUALITY_HIGH", "on").strip().lower() in ("off", "0", "false", "no"):
        return STANDARD
    q = requested(params)
    return q if q == STANDARD or supported(mode, params) else STANDARD


def _swap_bitrate(args: List[str], rate: str) -> List[str]:
    """Encoder args with every -b:v / -maxrate cap set to ``rate`` and -bufsize to 2 s of it (``-b:v 0`` stays)."""
    from scripts.shared.cut_profile import _double
    out = list(args)
    for i, a in enumerate(out[:-1]):
        if a in ("-b:v", "-maxrate") and out[i + 1] != "0":
            out[i + 1] = rate
        elif a == "-bufsize":
            out[i + 1] = _double(rate)
    return out


def encoder_args(base: Optional[List[str]], quality: str) -> Optional[List[str]]:
    if quality != HIGH:
        return base
    if base is None:
        from scripts.shared.cut_profile import encoder_args as ea, encoder_family
        return ea(encoder_family(), HIGH_BITRATE)
    if "-maxrate" not in base and "-b:v" not in base:          # an uncapped operator recipe: add the cap
        from scripts.shared.cut_profile import _double
        return list(base) + ["-maxrate", HIGH_BITRATE, "-bufsize", _double(HIGH_BITRATE)]
    return _swap_bitrate(base, HIGH_BITRATE)


def apply(mode: str, params: Optional[Mapping[str, Any]]) -> str:
    """Set this process up for the job: output size (smart_crop_adapter) + encoder cap (SONYA_RENDER_ARGS).
    -> the quality used. A standard job restores exactly what the process had before."""
    from scripts.shared.crop import smart_crop_adapter as sca
    q = resolve(mode, params)
    if "args" not in _BASE:
        _BASE["args"] = os.environ.get("SONYA_RENDER_ARGS")
    sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT = SIZES[q]
    if q == HIGH:
        try:
            base = json.loads(_BASE["args"]) if _BASE["args"] else None
        except ValueError:
            base = None
        try:
            os.environ["SONYA_RENDER_ARGS"] = json.dumps(encoder_args(base, HIGH))
        except Exception as exc:          # noqa: BLE001 - no encoder probe: libx264 default of the renderer
            logger.warning("[quality] encoder args for 2K failed: %s", exc)
    elif _BASE["args"] is None:
        os.environ.pop("SONYA_RENDER_ARGS", None)
    else:
        os.environ["SONYA_RENDER_ARGS"] = _BASE["args"]
    if q != requested(params):
        logger.info("[quality] %s asked for %s: not available for mode=%s clipType=%s - standard",
                    "job", requested(params), mode, (params or {}).get("clipType"))
    logger.info("[quality] %s: %dx%d", q, *SIZES[q])
    return q


def size() -> Tuple[int, int]:
    from scripts.shared.crop import smart_crop_adapter as sca
    return sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT


def source_height(quality: str) -> Optional[int]:
    return HIGH_SOURCE_HEIGHT if quality == HIGH else None
