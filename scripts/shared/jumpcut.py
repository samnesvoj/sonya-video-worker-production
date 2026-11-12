"""
jumpcut.py
==========
Jump cuts for short clips (streamer v2, owner 2026-11-10: «убери паузы/тишину > 0.7 с внутри клипа, без обрыва фраз»).

A clip [start, end] becomes a list of *pieces* (source intervals played back to back):
  * a gap between two recognised words longer than ``max_gap`` is shortened to ``keep_gap`` (half of it stays
    after the last word, half before the next one: no word is ever clipped);
  * a gap is kept whole when the stream is loud there (``loud`` per-second z-score ≥ ``event_z``: a scream ASR
    did not catch, an explosion, a game event) — silence is cut, action is not;
  * piece edges sit on the frame grid (1/fps), so N pieces do not drift audio against video.

``to_output(t, pieces)`` maps a source time to the cut timeline; ``remap_words`` moves ASR words there (subtitles).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Piece = Tuple[float, float]          # (source start, duration)


def _grid(t: float, fps: float) -> float:
    return round(round(t * fps) / fps, 4)


def pieces(words: Sequence[Dict[str, Any]], start: float, end: float, *, max_gap: float = 0.7,
           keep_gap: float = 0.25, loud: Optional[Sequence[float]] = None, event_z: float = 1.0,
           fps: float = 30.0, min_piece: float = 0.4) -> List[Piece]:
    """-> [(src_start, dur), ...] covering [start, end] minus the dead air."""
    ws = sorted(((float(w["start"]), float(w["end"])) for w in words
                 if float(w["end"]) > start and float(w["start"]) < end), key=lambda x: x[0])
    cuts: List[Tuple[float, float]] = []            # source intervals to drop
    # leading / trailing silence inside the clip counts too (boundaries already snapped to words)
    edges = [(start, start)] + ws + [(end, end)]
    for (a0, a1), (b0, _b1) in zip(edges, edges[1:]):
        gap_a, gap_b = max(a1, start), min(b0, end)
        if gap_b - gap_a <= max_gap:
            continue
        if loud is not None and _event(loud, gap_a, gap_b, event_z):
            continue
        half = keep_gap / 2.0
        lo = gap_a + (half if a1 > start else 0.0)
        hi = gap_b - (half if b0 < end else 0.0)
        if hi - lo > 0.05:
            cuts.append((lo, hi))
    out: List[Piece] = []
    t = start
    for lo, hi in cuts:
        if lo > t:
            out.append((t, lo - t))
        t = hi
    if end > t:
        out.append((t, end - t))
    # frame grid + merge slivers into neighbours (a 0.2 s flash of picture looks like a glitch)
    g = [(_grid(a, fps), _grid(a + d, fps)) for a, d in out]
    g = [(a, b) for a, b in g if b - a > 0]
    merged: List[List[float]] = []
    for a, b in g:
        if merged and (b - a < min_piece or merged[-1][1] - merged[-1][0] < min_piece):
            merged[-1][1] = b if b > merged[-1][1] else merged[-1][1]
            continue
        merged.append([a, b])
    return [(round(a, 4), round(b - a, 4)) for a, b in merged]


def _event(loud: Sequence[float], a: float, b: float, z: float) -> bool:
    i0, i1 = max(0, int(math.floor(a))), min(len(loud), int(math.ceil(b)))
    return any(float(loud[i]) >= z for i in range(i0, i1))


def total(ps: Sequence[Piece]) -> float:
    return round(sum(d for _, d in ps), 4)


def to_output(t: float, ps: Sequence[Piece]) -> Optional[float]:
    """Source time → cut-timeline time (None when t falls in a removed gap)."""
    acc = 0.0
    for a, d in ps:
        if a - 1e-6 <= t <= a + d + 1e-6:
            return round(acc + min(max(t - a, 0.0), d), 4)
        acc += d
    return None


def remap_words(words: Sequence[Dict[str, Any]], ps: Sequence[Piece]) -> List[Dict[str, Any]]:
    """ASR words on the cut timeline (a word whose edge is in a removed gap is clamped to its piece)."""
    out = []
    acc = 0.0
    bounds = []
    for a, d in ps:
        bounds.append((a, a + d, acc))
        acc += d
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        for a, b, off in bounds:
            if s < b and e > a:
                ns, ne = off + max(s, a) - a, off + min(e, b) - a
                if ne - ns >= 0.03:
                    out.append(dict(w, start=round(ns, 3), end=round(ne, 3)))
                break
    return out


def cut_points(ps: Sequence[Piece]) -> List[float]:
    """Output times where one piece ends and the next begins."""
    acc, pts = 0.0, []
    for _, d in ps[:-1]:
        acc += d
        pts.append(round(acc, 4))
    return pts


def pick_insert(words_out: Sequence[Dict[str, Any]], ps: Sequence[Piece], *, at: float = 0.4,
                lo: float = 0.28, hi: float = 0.6) -> Optional[float]:
    """Where to insert a short ad on the cut timeline: a pause between phrases near ``at`` of the clip.
    Candidates: jump-cut points and gaps between words; a sentence end is preferred. None = no clean spot."""
    dur = total(ps)
    if dur <= 0:
        return None
    cands: List[Tuple[float, float]] = []           # (time, cost)
    cps = set(cut_points(ps))
    ws = list(words_out)
    for x, y in zip(ws, ws[1:]):
        gap = float(y["start"]) - float(x["end"])
        if gap < 0.12 and not any(float(x["end"]) - 0.01 <= c <= float(y["start"]) + 0.01 for c in cps):
            continue
        t = (float(x["end"]) + float(y["start"])) / 2.0
        if not lo * dur <= t <= hi * dur:
            continue
        txt = str(x.get("word") or x.get("text") or "").strip()
        sentence_end = txt[-1:] in ".!?…"
        cost = abs(t - at * dur) - (4.0 if sentence_end else 0.0) - min(gap, 0.6) * 4.0
        cands.append((round(t, 3), cost))
    if not cands:
        return None
    return min(cands, key=lambda c: c[1])[0]


# ── ad-slot presets (owner 2026-11-10 ~05:35) ───────────────────────────────
# suspense        — tension grows («сейчас… смотри… ну давай…», louder / faster) → the ad right BEFORE the payoff
# after_hook      — 3–6.5 s in: the viewer is already hooked
# mid_pause       — a pause between phrases near 40 % (the first version)
# before_punchline — right before the last punchline phrase
# Always on a gap between words (never inside a phrase), never in the first 3 s (the hook) nor the last 3 s, and
# the payoff itself always plays after the ad (the ad never eats it).
AD_SLOTS = ("suspense", "after_hook", "mid_pause", "before_punchline")
DEFAULT_AD_SLOT = "suspense"
SUSPENSE_RE = re.compile(r"^(сейчас|щас|смотри|смотрите|гляди|давай|давайте|подожди|погоди|тихо|внимание|"
                         r"готовы?|ждём|ждем|момент|секунду|сейча|ну)$")


def _gaps(words_out: Sequence[Dict[str, Any]], ps: Sequence[Piece]) -> List[Tuple[float, float, bool, int]]:
    """Clean insert points on the cut timeline: (time, gap, after a sentence end, index of the next word)."""
    cps = set(cut_points(ps))
    ws = list(words_out)
    out = []
    for i, (x, y) in enumerate(zip(ws, ws[1:])):
        gap = float(y["start"]) - float(x["end"])
        if gap < 0.12 and not any(float(x["end"]) - 0.01 <= c <= float(y["start"]) + 0.01 for c in cps):
            continue
        txt = str(x.get("word") or x.get("text") or "").strip()
        out.append((round((float(x["end"]) + float(y["start"])) / 2.0, 3), gap, txt[-1:] in ".!?…", i + 1))
    return out


def _word(w: Dict[str, Any]) -> str:
    return re.sub(r"[^\w]", "", str(w.get("word") or w.get("text") or "")).lower()


def suspense_target(words_out: Sequence[Dict[str, Any]], dur: float, peak: Optional[float] = None
                    ) -> Optional[float]:
    """No editor mark: the payoff = the loudness peak if it is past the hook, else the first sentence start after
    the last build-up word («сейчас / смотри / давай …») in the first 85 % of the clip."""
    if peak is not None and 4.0 <= peak <= dur - 1.5:
        return peak
    ws = list(words_out)
    last = None
    for i, w in enumerate(ws):
        if float(w["start"]) <= 0.85 * dur and SUSPENSE_RE.match(_word(w)):
            last = i
    if last is None:
        return None
    for j in range(last, len(ws) - 1):                       # the end of the build-up phrase
        if str(ws[j].get("word") or ws[j].get("text") or "").strip()[-1:] in ".!?…" or \
                float(ws[j + 1]["start"]) - float(ws[j]["end"]) >= 0.3:
            return float(ws[j + 1]["start"])
    return None


def last_phrase_start(words_out: Sequence[Dict[str, Any]], dur: float) -> Optional[float]:
    ws = list(words_out)
    for j in range(len(ws) - 2, -1, -1):
        t = float(ws[j + 1]["start"])
        if t < 0.5 * dur:
            break
        if str(ws[j].get("word") or ws[j].get("text") or "").strip()[-1:] in ".!?…" and t <= dur - 2.0:
            return t
    return None


def pick_slot(words_out: Sequence[Dict[str, Any]], ps: Sequence[Piece], *, slot: str = DEFAULT_AD_SLOT,
              payoff: Optional[float] = None, punchline: Optional[float] = None, peak: Optional[float] = None,
              at: float = 0.4, hook_sec: float = 3.0) -> Tuple[Optional[float], str, str]:
    """Insert time on the cut timeline for an ad-slot preset. payoff / punchline / peak are cut-timeline seconds
    (the editor's marks or the loudness peak). -> (time or None, slot actually used, how it was found).
    A preset without a usable target falls back to mid_pause (reported)."""
    dur = total(ps)
    if dur <= 0:
        return None, slot, "empty"
    gaps = [g for g in _gaps(words_out, ps) if hook_sec <= g[0] <= dur - 3.0]
    slot = slot if slot in AD_SLOTS else DEFAULT_AD_SLOT

    def before(target: float, back: float) -> Optional[float]:
        # the latest clean gap right before the target phrase (≤ 0.15 s past its first word's start)
        c = [g for g in gaps if target - back <= g[0] <= target + 0.15]
        if not c:
            return None
        # a sentence end within 1 s of the latest is preferred
        best = max(c, key=lambda g: g[0])
        se = [g for g in c if g[2] and g[0] >= best[0] - 1.0]
        return (max(se, key=lambda g: g[0]) if se else best)[0]

    if slot == "after_hook":
        c = [g for g in gaps if g[0] <= 6.5]
        if c:
            return min(c, key=lambda g: abs(g[0] - 4.0) - (1.0 if g[2] else 0.0) - min(g[1], 0.6))[0], slot, "gap 3–6.5 s"
    elif slot == "suspense":
        how = "editor payoff"
        target = payoff if payoff is not None and payoff > hook_sec else None
        if target is None:
            target, how = suspense_target(words_out, dur, peak), "local build-up/peak"
        if target is not None:
            t = before(target, 3.0)
            if t is not None:
                return t, slot, how
    elif slot == "before_punchline":
        how = "editor punchline"
        target = punchline if punchline is not None and punchline > hook_sec else None
        if target is None:
            target, how = last_phrase_start(words_out, dur), "last phrase"
        if target is not None:
            t = before(target, 2.5)
            if t is not None:
                return t, slot, how
    t = pick_insert(words_out, ps, at=at)
    return t, "mid_pause", ("pause near %d %%" % round(at * 100)) if slot == "mid_pause" else f"{slot} fallback"
