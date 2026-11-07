"""
audio_features.py
=================
Cheap audio signal for Cut ranking: loudness per second of the whole source, once.

One ffmpeg pass decodes mono 8 kHz PCM; RMS per 1 s window → dBFS; z-scored against the
video's own median/MAD (a loud podcast and a quiet interview are judged on their own scale).
Per candidate: `energy_peak` (the strongest 2 s inside it) and `energy_mean`. Laughter,
shouting and heated exchanges show up as peaks; a flat read-out stays near 0.

Cached as JSON (`cache_path`) — a 2.7 h source takes ~6 s on an M4.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import numpy as _np
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "audio-energy-v1"
RATE = 8000


def loudness_per_second(media: str, ffmpeg: str = "ffmpeg", cache_path: Optional[str] = None) -> List[float]:
    if cache_path and os.path.isfile(cache_path):
        try:
            d = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if d.get("version") == VERSION:
                return d["db"]
        except (ValueError, KeyError):
            pass
    p = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-i", media, "-vn", "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], stdout=subprocess.PIPE)
    out: List[float] = []
    assert p.stdout is not None
    while True:
        buf = p.stdout.read(RATE * 2)
        if not buf:
            break
        a = _np.frombuffer(buf[: len(buf) // 2 * 2], dtype=_np.int16).astype(_np.float64)
        if not a.size:
            break
        ms = float(_np.mean(a * a))
        out.append(round(10 * math.log10(ms / (32768.0 ** 2) + 1e-10), 2))
    p.wait()
    if p.returncode != 0 and not out:
        raise RuntimeError("audio decode failed")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({"version": VERSION, "db": out}), encoding="utf-8")
    return out


def zscores(db: List[float]) -> List[float]:
    """Robust z: (x - median) / (1.4826 * MAD), silence (< -60 dB) ignored for the scale."""
    live = sorted(x for x in db if x > -60)
    if len(live) < 10:
        return [0.0] * len(db)
    med = live[len(live) // 2]
    mad = sorted(abs(x - med) for x in live)[len(live) // 2] or 1.0
    return [round((x - med) / (1.4826 * mad), 3) for x in db]


def candidate_energy(z: List[float], start: float, end: float) -> Dict[str, float]:
    a, b = max(0, int(start)), min(len(z), int(math.ceil(end)))
    win = z[a:b]
    if not win:
        return {"energy_peak": 0.0, "energy_mean": 0.0}
    pairs = [(win[i] + win[i + 1]) / 2 for i in range(len(win) - 1)] or win
    return {"energy_peak": round(max(pairs), 3), "energy_mean": round(sum(win) / len(win), 3)}
