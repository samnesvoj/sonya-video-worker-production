"""
scripts/shared/loudness.py
==========================
Even loudness for Cut clips (modes-improve 2026-11-16).

Production clips came out anywhere from -24 LUFS (quiet street interview: the viewer has to turn the phone up)
to -7 LUFS (a mastered YouTube show: loud, flat, near clipping). Reels / Shorts / TikTok play speech at about
-14 LUFS, so every clip gets ONE linear gain to -14 LUFS, measured on the clip's own audio, plus a true-peak-safe
limiter (-1.5 dBFS). A linear gain (not ffmpeg's dynamic loudnorm) keeps the speech dynamics: shouts stay
louder than whispers, nothing "pumps".

  SONYA_CLIP_LUFS   target integrated loudness (default -14); "off" = no change (ops kill switch)

The hook piece of Hooks mode reuses its clip's gain (a 2 s scream normalised on its own would be turned down).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET = -14.0
MAX_BOOST_DB = 12.0          # a near-silent source is not pumped up into noise
MAX_CUT_DB = 12.0
MIN_CHANGE_DB = 0.5          # closer than that: leave the audio untouched
SILENCE_LUFS = -50.0         # gated loudness this low = no real sound to normalise
LIMITER = "alimiter=limit=0.84:level=false"     # 0.84 ≈ -1.5 dBFS: a boost never clips


def target_lufs(env: Optional[dict] = None) -> Optional[float]:
    raw = str((os.environ if env is None else env).get("SONYA_CLIP_LUFS", "")).strip().lower()
    if raw in ("off", "0", "false", "no", "none"):
        return None
    if not raw:
        return DEFAULT_TARGET
    try:
        v = float(raw)
    except ValueError:
        logger.warning("[loudness] SONYA_CLIP_LUFS=%r is not a number - using %s", raw, DEFAULT_TARGET)
        return DEFAULT_TARGET
    return v if -30.0 <= v <= -8.0 else DEFAULT_TARGET


def parse_integrated(stderr: str) -> Optional[float]:
    """The summary 'I: -17.3 LUFS' of ffmpeg's ebur128 filter (the last one = the summary)."""
    found = re.findall(r"I:\s+(-?[0-9.]+|-inf)\s+LUFS", stderr or "")
    if not found or found[-1] == "-inf":
        return None
    return float(found[-1])


def measure(ffmpeg: str, path: str, start: float = 0.0, duration: Optional[float] = None,
            timeout: float = 120.0) -> Optional[float]:
    """Integrated loudness (LUFS) of path[start, start+duration] (audio only), None when there is no audio."""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-ss", f"{max(0.0, start):.3f}", "-i", path]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-sn", "-af", "ebur128=framelog=quiet", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:     # noqa: BLE001 - loudness is a polish step, never a reason to lose a clip
        logger.warning("[loudness] measure failed: %s", exc)
        return None
    return parse_integrated(r.stderr)


def gain_db(measured: Optional[float], target: Optional[float]) -> Optional[float]:
    """The linear gain that brings `measured` to `target`; None = leave the audio alone."""
    if measured is None or target is None or measured <= SILENCE_LUFS:
        return None
    g = max(-MAX_CUT_DB, min(MAX_BOOST_DB, target - measured))
    return round(g, 2) if abs(g) >= MIN_CHANGE_DB else None


def audio_filter(gain: Optional[float]) -> Optional[str]:
    """ffmpeg -af chain for a gain (None -> no filter)."""
    if gain is None:
        return None
    return f"volume={gain:.2f}dB,{LIMITER}"


def clip_gain(ffmpeg: str, path: str, start: float, duration: float) -> Optional[float]:
    """Measure the clip's span of the source and return its gain (None = off / silent / already right)."""
    t = target_lufs()
    if t is None:
        return None
    return gain_db(measure(ffmpeg, path, start, duration), t)
