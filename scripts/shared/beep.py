"""
beep.py
=======
SONYA BEEP: censor intervals from profanity hits and the audio effect.

    hits (profanity.detect) ──► intervals (padding, merge, clamp)
                            ──► effect: mute | beep | custom asset  (separate from detection)
                            ──► one ffmpeg pass: video stream-copied (or re-encoded only to
                                burn masked subtitles), audio rebuilt
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .profanity import Hit

BEEP_VERSION = "beep-v1"
EFFECTS = ("beep", "mute", "custom")


@dataclass
class BeepConfig:
    effect: str = "beep"
    sound: str = "tv_beep"               # beep: a preset from scripts/shared/beep_sounds.py
    asset: Optional[str] = None          # custom: audio file played (from its start) over each interval
    categories: Tuple[str, ...] = ("mat", "rude")
    pad_pre: float = 0.05                # ASR word edges are ±50 ms: cover the whole word
    pad_post: float = 0.08
    merge_gap: float = 0.15              # two bleeps closer than this become one
    beep_freq: float = 1000.0
    beep_gain: float = 0.35              # linear, ~ -9 dBFS
    fade: float = 0.01                   # 10 ms edges: no clicks
    subtitles: str = "off"               # off | srt | burn  (masked text either way)

    def __post_init__(self) -> None:
        if self.effect not in EFFECTS:
            raise ValueError(f"effect must be one of {EFFECTS}")
        if self.effect == "custom" and not self.asset:
            raise ValueError("effect=custom needs an asset")
        from .beep_sounds import SOUNDS
        if self.sound not in SOUNDS:
            raise ValueError(f"sound must be one of {sorted(SOUNDS)}")
        if self.subtitles not in ("off", "srt", "burn"):
            raise ValueError("subtitles must be off|srt|burn")
        if min(self.pad_pre, self.pad_post, self.merge_gap) < 0:
            raise ValueError("pad/merge must be >= 0")
        self.categories = tuple(self.categories)


def censor_intervals(hits: Sequence[Hit], duration: float, cfg: BeepConfig) -> List[List[float]]:
    spans = sorted((max(0.0, h.start - cfg.pad_pre), min(duration, h.end + cfg.pad_post)) for h in hits)
    out: List[List[float]] = []
    for a, b in spans:
        if b <= a:
            continue
        if out and a - out[-1][1] <= cfg.merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def audio_filter(intervals: Sequence[Sequence[float]], cfg: BeepConfig, custom_input: int = 1) -> str:
    """filter_complex producing [aout]. The original is silenced inside the
    intervals; for beep/custom one short source per interval is mixed in."""
    if not intervals:
        return "[0:a]aresample=48000[aout]"
    gate = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)
    parts = [f"[0:a]aresample=48000,volume=0:enable='{gate}'[orig]"]
    if cfg.effect == "mute" or not intervals:
        return parts[0].replace("[orig]", "[aout]")
    n = len(intervals)
    labels = []
    if cfg.effect == "custom" or cfg.sound != "tv_beep":
        parts.append(f"[{custom_input}:a]aresample=48000,aformat=channel_layouts=stereo,asplit={n}"
                     + "".join(f"[c{i}]" for i in range(n)))
    for i, (a, b) in enumerate(intervals):
        d = b - a
        f = min(cfg.fade, d / 4)
        src = (f"sine=frequency={cfg.beep_freq:g}:sample_rate=48000:duration={d:.3f},aformat=channel_layouts=stereo,"
               f"volume={cfg.beep_gain:g}" if cfg.effect == "beep" and cfg.sound == "tv_beep"
               else f"[c{i}]atrim=0:{d:.3f},asetpts=PTS-STARTPTS")
        ms = int(round(a * 1000))
        parts.append(f"{src},afade=t=in:d={f:.3f},afade=t=out:st={d - f:.3f}:d={f:.3f},adelay={ms}|{ms}[e{i}]")
        labels.append(f"[e{i}]")
    parts.append(f"[orig]{''.join(labels)}amix=inputs={n + 1}:normalize=0:duration=first[aout]")
    return ";".join(parts)


def ffmpeg_bin() -> str:
    return os.environ.get("SONYA_FFMPEG") or "ffmpeg"


def render(input_video: str, output: str, intervals: Sequence[Sequence[float]], cfg: BeepConfig,
           subtitle_ass: Optional[str] = None, encoder_args: Optional[List[str]] = None) -> str:
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", input_video]
    preset_wav = None
    if cfg.effect == "custom" and intervals:
        cmd += ["-i", str(cfg.asset)]
    elif cfg.effect == "beep" and cfg.sound != "tv_beep" and intervals:
        from .beep_sounds import write_wav
        longest = max(b - a for a, b in intervals)
        preset_wav = str(Path(output).with_suffix(f".{cfg.sound}.wav"))
        write_wav(cfg.sound, preset_wav, max(4.0, longest + 0.5))
        cmd += ["-i", preset_wav]
    graph = audio_filter(intervals, cfg)
    cwd = None
    if subtitle_ass:
        sub = Path(subtitle_ass).resolve()
        cwd = str(sub.parent)
        graph = f"[0:v]ass=filename={sub.name}[vout];" + graph
        vmap = ["-map", "[vout]", *(encoder_args or ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]),
                "-pix_fmt", "yuv420p"]
    else:
        vmap = ["-map", "0:v:0", "-c:v", "copy"]          # picture untouched, bit for bit
    cmd += ["-filter_complex", graph, *vmap, "-map", "[aout]", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", output]
    try:
        from scripts.shared import nvenc_guard        # GeForce NVENC session cap -> CPU, never an error
        r = nvenc_guard.run(cmd, capture_output=True, text=True, cwd=cwd)
    finally:
        if preset_wav:
            Path(preset_wav).unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"beep render failed: {r.stderr[-600:]}")
    return output


def srt(cues) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return "\n".join(f"{i}\n{ts(c.start)} --> {ts(c.end)}\n" + "\n".join(c.lines) + "\n"
                     for i, c in enumerate(cues, 1))
