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
