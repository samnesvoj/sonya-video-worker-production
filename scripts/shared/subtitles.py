"""
subtitles.py
============
Deterministic burn-in subtitles from ASR word timestamps.

    ASR words ──► words inside the clip's render interval (clip-local time)
              ──► cues (punctuation / pause / length / max duration)
              ──► ≤ 2 lines per cue, wrapped between words only
              ──► ASS file (PlayRes 1080x1920, safe zone above the bottom UI)

Only ASR text and timestamps are used — nothing is invented. Burn-in is
done by the existing crop render (smart_crop_adapter.render_tracked) as one
extra filter in the same ffmpeg pass.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .boundaries import Word, normalize_words

SUBTITLES_VERSION = "subs-v1"
_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)\]]*$")
_CLAUSE_END = re.compile(r"[,;:—–-]+$")


@dataclass
class SubtitleStyle:
    version: str = "style-sonya-v1"
    font: str = "Arial"               # has Cyrillic on macOS/Windows; libass falls back via fontconfig
    size: int = 68
    bold: bool = True
    primary: str = "&H00FFFFFF"       # ASS &HAABBGGRR
    outline_color: str = "&H00000000"
    outline: float = 5.0
    shadow: float = 0.0
    margin_v: int = 380               # px from bottom: clear of TikTok/Reels/Shorts UI
    margin_lr: int = 90
    max_chars_per_line: int = 24
    max_lines: int = 2
    max_cue_sec: float = 3.2
    min_cue_sec: float = 0.5
    pause_split_sec: float = 0.45


@dataclass
class Cue:
    start: float
    end: float
    lines: List[str]
    words: int


def clip_words(raw: Iterable[Dict[str, Any]], clip_start: float, clip_end: float) -> List[Word]:
    """ASR words overlapping [clip_start, clip_end], in clip-local time,
    clamped to the clip. A word cut by the edge is kept (never lost)."""
    words, _ = normalize_words(raw)
    dur = clip_end - clip_start
    out = []
    for w in words:
        if w.end > clip_start + 1e-6 and w.start < clip_end - 1e-6:
            s = max(0.0, w.start - clip_start)
            e = min(dur, w.end - clip_start)
            if e > s:
                out.append(Word(w.text.strip(), round(s, 3), round(e, 3)))
    return out


def wrap(text_words: List[str], max_chars: int) -> List[str]:
    """Greedy wrap between words; a single over-long word gets its own line."""
    lines: List[str] = []
    cur = ""
    for w in text_words:
        cand = f"{cur} {w}" if cur else w
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build_cues(words: List[Word], style: SubtitleStyle, clip_duration: float) -> List[Cue]:
    cues: List[Cue] = []
    group: List[Word] = []

    def flush() -> None:
        if group:
            cues.append(Cue(group[0].start, group[-1].end, wrap([w.text for w in group], style.max_chars_per_line),
                            len(group)))
            group.clear()

    for i, w in enumerate(words):
        if group:
            prev = group[-1]
            fits = len(wrap([g.text for g in group] + [w.text], style.max_chars_per_line)) <= style.max_lines
            too_long = w.end - group[0].start > style.max_cue_sec
            pause = w.start - prev.end >= style.pause_split_sec
            if not fits or too_long or pause or _SENTENCE_END.search(prev.text):
                flush()
            elif _CLAUSE_END.search(prev.text) and w.end - group[0].start > style.max_cue_sec * 0.6:
                flush()
        group.append(w)
    flush()
    # timing: keep cues on screen at least min_cue_sec, never overlapping, never past the clip
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else clip_duration
        c.end = round(min(max(c.end, c.start + style.min_cue_sec), nxt, clip_duration), 3)
    return cues


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    """ASS treats {...} as override blocks and \\ as an escape; make them literal."""
    return text.replace("\\", "⧵").replace("{", "｛").replace("}", "｝").replace("\n", " ")


def _in_any(c: Cue, intervals) -> bool:
    """Most of the cue falls inside one of the intervals."""
    return any(min(c.end, b) - max(c.start, a) >= 0.5 * (c.end - c.start) for a, b in intervals or [])


def placement(c: Cue, style: SubtitleStyle, play_res: Tuple[int, int], centered, avoid) -> str:
    """Position override for one cue: the seam during a split screen; moved off any
    burned-in text it would cover (above it, else below it, else top of frame)."""
    w, h = play_res
    lines = max(1, len(c.lines))
    band_h = lines * style.size * 1.25
    if _in_any(c, centered):
        top, bottom, tag = h / 2 - band_h / 2, h / 2 + band_h / 2, "{\\an5\\pos(%d,%d)}" % (w // 2, h // 2)
    else:
        bottom = h - style.margin_v
        top, tag = bottom - band_h, ""
    for a, b, ytop, ybot in avoid or []:
        if min(c.end, b) - max(c.start, a) < 0.3 * (c.end - c.start) or ybot < top - 20 or ytop > bottom + 20:
            continue
        if ytop - 30 - band_h >= 220:                    # room above the text
            return "{\\an2\\pos(%d,%d)}" % (w // 2, int(ytop - 30))
        if ybot + 30 + band_h <= h - 260:                # else below it
            return "{\\an8\\pos(%d,%d)}" % (w // 2, int(ybot + 30))
        return "{\\an8\\pos(%d,%d)}" % (w // 2, 240)
    return tag


def to_ass(cues: List[Cue], style: SubtitleStyle, play_res: Tuple[int, int] = (1080, 1920),
           centered: Optional[List[Tuple[float, float]]] = None,
           avoid: Optional[List[List[float]]] = None) -> str:
    """`centered`: clip intervals (split screen) where cues sit on the seam in the
    middle of the frame instead of low — low would cover the lower person's face."""
    w, h = play_res
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Sonya,{style.font},{style.size},{style.primary},&H000000FF,{style.outline_color},&H00000000,"
        f"{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},2,"
        f"{style.margin_lr},{style.margin_lr},{style.margin_v},204",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    ev = [f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Sonya,,0,0,0,,"
          + placement(c, style, play_res, centered, avoid)
          + r"\N".join(ass_escape(l) for l in c.lines) for c in cues]
    return "\n".join(head + ev) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(f".{os.getpid()}.partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def subtitles_for_clip(raw_words: Optional[Iterable[Dict[str, Any]]], clip_start: float, clip_duration: float,
                       out_path: Path, style: Optional[SubtitleStyle] = None,
                       centered: Optional[List[Tuple[float, float]]] = None,
                       avoid: Optional[List[List[float]]] = None, preset: Optional[str] = None,
                       keywords: Optional[List[str]] = None, hook: Optional[str] = None,
                       font: Optional[str] = None, promo: Optional[Dict[str, Any]] = None,
                       hook_top: Optional[int] = None) -> Dict[str, Any]:
    """Write <clip>.ass for one render interval. Returns metadata; no file
    when there are no words (status says why)."""
    style = style or SubtitleStyle()
    meta: Dict[str, Any] = {"subtitle_version": SUBTITLES_VERSION, "subtitle_style": style.version}
    words = clip_words(raw_words, clip_start, clip_start + clip_duration) if raw_words else []
    if not words:
        reason = "no_word_timestamps" if not raw_words else "no_words_in_interval"
        if promo:   # the ad and its legal marking never depend on speech being there
            from .subtitle_styles import to_ass as preset_ass
            write_atomic(out_path, preset_ass([], preset or "highlight", clip_duration, font=font, promo=promo))
            return dict(meta, subtitle_status="promo_only", subtitle_reason=reason, subtitle_path=str(out_path),
                        promo_version="promo-v1")
        return dict(meta, subtitle_status=reason)
    if promo and (not preset or preset == "classic"):
        preset = "highlight"        # the classic writer has no promo layer
    cues = build_cues(words, style, clip_duration)
    if preset and preset != "classic":   # trendy word-level styles (scripts/shared/subtitle_styles.py)
        from .subtitle_styles import PRESETS, to_ass as preset_ass
        ass_text = preset_ass(words, preset, clip_duration, centered=centered, avoid=avoid, base=style,
                              keywords=keywords, hook=hook, font=font, promo=promo,
                              **({"hook_top": hook_top} if hook_top is not None else {}))
        if promo:
            meta["promo_version"] = "promo-v1"
        if hook:
            meta["subtitle_hook"] = hook
        write_atomic(out_path, ass_text)
        from .subtitle_styles import glass_pills
        pills = glass_pills(ass_text, preset)
        if pills:   # sidecar for the renderer: where to blur + frost the video behind the text
            write_atomic(Path(out_path).with_suffix(".glass.json"), json.dumps(pills))
        meta["subtitle_style"] = f"preset:{preset if preset in PRESETS else 'highlight'}"
    else:
        write_atomic(out_path, to_ass(cues, style, centered=centered, avoid=avoid))
    meta["subtitle_centered_cues"] = sum(1 for c in cues if _in_any(c, centered))
    meta["subtitle_moved_for_text"] = sum(1 for c in cues if placement(c, style, (1080, 1920), centered, avoid)
                                          != placement(c, style, (1080, 1920), centered, None))
    return dict(meta, subtitle_status="ok", subtitle_path=str(out_path), subtitle_cues=len(cues),
                subtitle_words=len(words), subtitle_first_word=words[0].text, subtitle_last_word=words[-1].text)
