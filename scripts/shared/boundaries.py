"""
boundaries.py
=============
Deterministic clip boundary engine: turns a *semantic* interval (chosen by
a mode / Gemini / heuristics) into a *render* interval that sits on real
ASR word boundaries.

It never decides WHAT is a good moment and never invents timestamps:
without word timestamps the semantic interval is kept as-is and marked
`boundary_status = "no_word_timestamps"`.

Steps per interval:
  1. words = normalize_words(asr words)       sort, drop malformed, de-overlap
  2. sentences = group_sentences(words)        punctuation + pauses
  3. snap to the words the interval touches    a word cut by the interval
                                               edge is included whole
  4. extend to the enclosing sentence edges    only if ≤ max_extend_sec away
  5. min/max duration                          add whole words / trim at the
                                               last sentence (or word) end
  6. pre/post-roll                             never into a neighbour word
Across clips, resolve_overlaps() keeps the higher-priority clip (earlier in
the list) and trims or drops the later one on word boundaries.

Everything is pure and deterministic (ties broken by index/start/text).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VERSION = "boundaries-v1"
_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)\]]*$")
EPS = 1e-6


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass
class BoundaryConfig:
    pre_roll: float = 0.15          # seconds of air before the first word
    post_roll: float = 0.30         # seconds after the last word
    pause_sec: float = 0.70         # silence that ends a sentence
    max_extend_sec: float = 3.0     # max stretch to reach a sentence edge
    min_duration: float = 5.0
    max_duration: float = 90.0
    overlap_tolerance: float = 0.5  # seconds of overlap tolerated between clips

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Refined:
    semantic_start: float
    semantic_end: float
    start: float
    end: float
    status: str                     # snapped | no_word_timestamps | no_speech_in_interval | dropped_overlap
    reasons: List[str] = field(default_factory=list)
    first_word: Optional[int] = None
    last_word: Optional[int] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_meta(self, cfg: BoundaryConfig) -> Dict[str, Any]:
        return {"semantic_start": round(self.semantic_start, 3), "semantic_end": round(self.semantic_end, 3),
                "render_start": round(self.start, 3), "render_end": round(self.end, 3),
                "boundary_status": self.status, "boundary_version": VERSION,
                "boundary_reasons": list(self.reasons), "boundary_params": cfg.to_dict()}


# ── 1. normalization ────────────────────────────────────────────────────

def normalize_words(raw: Iterable[Dict[str, Any]]) -> Tuple[List[Word], Dict[str, int]]:
    """Clean ASR words. Returns (words, stats).

    Drops entries without finite start/end, with end <= start or empty
    text; sorts by (start, end, text); trims a word that starts inside the
    previous one to the previous end (drops it if nothing is left).
    """
    stats = {"input": 0, "dropped_malformed": 0, "dropped_overlap": 0, "trimmed_overlap": 0}
    cand: List[Word] = []
    for w in raw or []:
        stats["input"] += 1
        try:
            text = str(w.get("word", w.get("text", ""))).strip()
            s, e = float(w.get("start")), float(w.get("end"))
        except (TypeError, ValueError, AttributeError):
            stats["dropped_malformed"] += 1
            continue
        if not text or not (math.isfinite(s) and math.isfinite(e)) or s < 0 or e <= s:
            stats["dropped_malformed"] += 1
            continue
        cand.append(Word(text, s, e))
    cand.sort(key=lambda w: (w.start, w.end, w.text))
    out: List[Word] = []
    for w in cand:
        if out and w.start < out[-1].end - EPS:
            if w.end <= out[-1].end + EPS:
                stats["dropped_overlap"] += 1
                continue
            w = Word(w.text, out[-1].end, w.end)
            stats["trimmed_overlap"] += 1
        out.append(w)
    return out, stats


# ── 2. sentences ────────────────────────────────────────────────────────

def group_sentences(words: Sequence[Word], pause_sec: float = 0.7) -> List[Tuple[int, int]]:
    """(first_idx, last_idx) per sentence. A sentence ends on terminal
    punctuation or when the gap to the next word is >= pause_sec."""
    groups: List[Tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        gap = words[i + 1].start - w.end if not last else 0.0
        if last or _SENTENCE_END.search(w.text) or gap >= pause_sec - EPS:
            groups.append((start, i))
            start = i + 1
    return groups


def _sentence_of(groups: Sequence[Tuple[int, int]], idx: int) -> Tuple[int, int]:
    for g in groups:
        if g[0] <= idx <= g[1]:
            return g
    return (idx, idx)


def cuts_inside_word(start: float, end: float, words: Sequence[Word]) -> bool:
    """True if either edge falls strictly inside a word."""
    return any(w.start + EPS < t < w.end - EPS for w in words for t in (start, end))


# ── 3–6. refine one interval ────────────────────────────────────────────

def _with_rolls(first: int, last: int, words: Sequence[Word], cfg: BoundaryConfig,
                reasons: List[str]) -> Tuple[float, float]:
    s = words[first].start - cfg.pre_roll
    if first > 0 and s < words[first - 1].end:
        s = words[first - 1].end
        reasons.append("pre_roll_limited_by_previous_word")
    e = words[last].end + cfg.post_roll
    if last + 1 < len(words) and e > words[last + 1].start:
        e = words[last + 1].start
        reasons.append("post_roll_limited_by_next_word")
    return max(0.0, s), e


def refine_interval(sem_start: float, sem_end: float, words: Sequence[Word],
                    groups: Sequence[Tuple[int, int]], cfg: BoundaryConfig) -> Refined:
    if sem_end < sem_start:
        sem_start, sem_end = sem_end, sem_start
    if not words:
        return Refined(sem_start, sem_end, sem_start, sem_end, "no_word_timestamps")
    touched = [i for i, w in enumerate(words) if w.end > sem_start + EPS and w.start < sem_end - EPS]
    if not touched:
        return Refined(sem_start, sem_end, sem_start, sem_end, "no_speech_in_interval")
    reasons: List[str] = []
    first, last = touched[0], touched[-1]
    if words[first].start < sem_start - EPS:
        reasons.append("start_was_inside_word")
    if words[last].end > sem_end + EPS:
        reasons.append("end_was_inside_word")

    gs, _ = _sentence_of(groups, first)
    if gs < first and words[first].start - words[gs].start <= cfg.max_extend_sec + EPS:
        first = gs
        reasons.append("start_extended_to_sentence")
    _, ge = _sentence_of(groups, last)
    if ge > last and words[ge].end - words[last].end <= cfg.max_extend_sec + EPS:
        last = ge
        reasons.append("end_extended_to_sentence")

    def dur(f: int, l: int) -> float:
        s, e = _with_rolls(f, l, words, cfg, [])
        return e - s

    # min duration: grow forward by whole words, then backward
    while dur(first, last) < cfg.min_duration - EPS:
        if last + 1 < len(words):
            last += 1
            if "extended_to_min_duration" not in reasons:
                reasons.append("extended_to_min_duration")
        elif first > 0:
            first -= 1
            if "extended_to_min_duration" not in reasons:
                reasons.append("extended_to_min_duration")
        else:
            reasons.append("shorter_than_min_duration_no_more_words")
            break

    # max duration: end at the last sentence end that fits, else last word that fits
    if dur(first, last) > cfg.max_duration + EPS:
        ends = [g[1] for g in groups if first <= g[1] <= last and dur(first, g[1]) <= cfg.max_duration + EPS]
        fit = [i for i in range(first, last + 1) if dur(first, i) <= cfg.max_duration + EPS]
        if ends:
            last = ends[-1]
            reasons.append("trimmed_to_max_duration_at_sentence")
        elif fit:
            last = fit[-1]
            reasons.append("trimmed_to_max_duration_at_word")
        else:
            reasons.append("first_word_longer_than_max_duration")

    s, e = _with_rolls(first, last, words, cfg, reasons)
    return Refined(sem_start, sem_end, s, e, "snapped", reasons, first, last)


def resolve_overlaps(refined: List[Refined], words: Sequence[Word], cfg: BoundaryConfig) -> List[Refined]:
    """Earlier entries win (they are higher ranked). A later clip that
    overlaps a kept one by more than overlap_tolerance is trimmed to the
    words outside it (keeping the longer side) or dropped if what remains is
    shorter than min_duration."""
    kept: List[Refined] = []
    for r in refined:
        if r.first_word is None:  # not snapped (no words/speech): left exactly as the mode chose
            kept.append(r)
            continue
        clash = [k for k in kept if k.status == "snapped"
                 and min(k.end, r.end) - max(k.start, r.start) > cfg.overlap_tolerance + EPS]
        if not clash:
            kept.append(r)
            continue
        idx = [i for i in range(r.first_word, r.last_word + 1)
               if not any(words[i].end > k.start and words[i].start < k.end for k in clash)]
        # longest contiguous run of free words (ties → earliest)
        runs: List[Tuple[int, int]] = []
        for i in idx:
            if runs and runs[-1][1] == i - 1:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((i, i))
        best = max(runs, key=lambda t: (words[t[1]].end - words[t[0]].start, -t[0]), default=None)
        if best is not None:
            reasons = r.reasons + ["trimmed_for_overlap"]
            s, e = _with_rolls(best[0], best[1], words, cfg, reasons)
            for k in clash:  # rolls must not re-enter a kept clip
                if s < k.end <= e and k.start <= s:
                    s = k.end
                if s <= k.start < e:
                    e = k.start
            if e - s >= cfg.min_duration - EPS:
                kept.append(Refined(r.semantic_start, r.semantic_end, s, e, "snapped", reasons, best[0], best[1]))
                continue
        r.status = "dropped_overlap"
        r.reasons.append("overlaps_higher_ranked_clip")
        kept.append(r)
    return kept


def refine_segments(segments: Sequence[Dict[str, Any]], raw_words: Iterable[Dict[str, Any]],
                    cfg: Optional[BoundaryConfig] = None) -> List[Dict[str, Any]]:
    """Mode-facing entry point. segments: [{start, duration|end, ...}] in
    priority order. Returns copies with start/duration replaced by the
    render interval and boundary metadata added; dropped clips are kept in
    the list with boundary_status=dropped_overlap so callers can skip them."""
    cfg = cfg or BoundaryConfig()
    words, stats = normalize_words(raw_words)
    groups = group_sentences(words, cfg.pause_sec)
    refined = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg["end"]) if "end" in seg else s + float(seg.get("duration", 0.0))
        refined.append(refine_interval(s, e, words, groups, cfg))
    refined = resolve_overlaps(refined, words, cfg)
    out = []
    for seg, r in zip(segments, refined):
        new = dict(seg)
        new.pop("end", None)
        new["start"], new["duration"] = round(r.start, 3), round(r.duration, 3)
        new.update(r.to_meta(cfg))
        new["asr_word_stats"] = stats
        out.append(new)
    return out
