"""
Trailer edit plan (pure): the measured voice → timeline; every phrase → the shots that SHOW it.

Timeline: a short lead-in, the segments back to back (0.2 s apart; after the hook a 0.45 s beat with a flash + whoosh),
the last word → 0.5 s hold → a hard cut to the small SONYA end card.

Shots of a segment: cut lengths by rhythm (1.8–2.8 s, the hook's first shot ~1.5 s, 1.5–2.2 s in the last third), cut
points snapped to the gaps between words. Every shot is a cut-free window of the source, chosen by
  writer's pick (the shots Gemini saw on the collages for this phrase)  +1.6
  the moment the phrase retells (its transcript lines ± 4 s)              +1.0
  near that moment (± 30 s)                                               +0.0
  anywhere in the storyline                                               −0.8
plus the picture score (big faces, motion; dark/empty/static lose). No window is used twice; neighbours never come from
the same source shot.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .shots import Scan

FPS = 30
LEAD = 0.12
GAP = 0.2
HOOK_BEAT = 0.45
HOLD = 0.5
END_CARD = 1.3
PAD_CUT = 0.12          # never closer than this to a source cut
PAD_USED = 0.3


def q(t: float) -> float:
    return round(round(t * FPS) / FPS, 4)


def timeline(segs: Sequence[Dict[str, Any]], voice: Dict[int, Dict[str, Any]], *, lead: float = LEAD,
             hook_beat: bool = True) -> Dict[str, Any]:
    """Absolute placement of every voiced segment + its words. -> {"segments": [{n, fn, t0, t1, wav, words}], "voice_end",
    "total", "end_card": (a, b)}"""
    t = lead
    out = []
    for k, s in enumerate(segs):
        v = voice[s["n"]]
        t0 = q(t)
        t1 = t0 + v["sec"]
        out.append({"n": s["n"], "fn": s["fn"], "t0": t0, "t1": round(t1, 4), "wav": v["wav"], "text": s["text"],
                    "src": s.get("src") or [], "shots": s.get("shots") or [],
                    "words": [(w, round(t0 + a, 3), round(t0 + b, 3)) for w, a, b in v["words"]]})
        t = t1 + (HOOK_BEAT if (k == 0 and hook_beat and s["fn"] == "hook" and len(segs) > 1) else GAP)
    voice_end = out[-1]["t1"]
    card0 = q(voice_end + HOLD)
    total = q(card0 + END_CARD)
    return {"segments": out, "voice_end": round(voice_end, 3), "end_card": (card0, total), "total": total}


def rhythm(progress: float, first: bool) -> Tuple[float, float]:
    if first:
        return 1.3, 1.8
    if progress < 0.35:
        return 1.9, 2.8
    if progress < 0.7:
        return 1.7, 2.5
    return 1.5, 2.2


def split_span(a: float, b: float, total: float, first: bool, word_gaps: Sequence[float]) -> List[float]:
    """Cut points inside [a, b] (excluding a and b): shots inside the rhythm range, each cut snapped to the nearest gap
    between words within ±0.35 s (cuts on the beat of the speech)."""
    cuts: List[float] = []
    t = a
    while True:
        lo, hi = rhythm(t / max(total, 0.1), first and not cuts)
        rest = b - t
        if rest <= hi + 0.15:
            break
        n_left = max(1, math.ceil(rest / hi))
        want = max(lo, min(hi, rest / n_left))
        c = t + want
        i = bisect.bisect_left(word_gaps, c)
        near = [g for g in word_gaps[max(0, i - 2):i + 2] if abs(g - c) <= 0.35 and t + lo * 0.8 <= g <= b - 1.0]
        if near:
            c = min(near, key=lambda g: abs(g - c))
        if b - c < 1.0:
            break
        cuts.append(q(c))
        t = c
    return cuts


def word_gaps(tl: Dict[str, Any]) -> List[float]:
    g = []
    for s in tl["segments"]:
        ws = s["words"]
        for x, y in zip(ws, ws[1:]):
            g.append(round((x[2] + y[1]) / 2, 3))
        g.append(s["t1"])
    return sorted(g)


def src_span(seg: Dict[str, Any], by_id: Dict[int, Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    ts = [(by_id[i]["start"], by_id[i]["end"]) for i in seg.get("src") or [] if i in by_id]
    if not ts:
        return None
    return min(a for a, _ in ts), max(b for _, b in ts)


def choose(sc: Scan, regions: Sequence[Tuple[float, float, float]], d: float, used: List[Tuple[float, float]],
           crop_w: int, avoid_shot: Optional[Tuple[float, float]] = None, step: float = 0.25,
           first_bonus: float = 0.0) -> Optional[Dict[str, Any]]:
    """Best cut-free unused window of length d over (lo, hi, bonus) regions."""
    best = None
    seen = set()
    for lo, hi, bonus in regions:
        s = lo
        while s + d <= hi + 1e-6:
            key = round(s, 2)
            if key in seen:
                s += step
                continue
            seen.add(key)
            e = s + d
            if sc.cuts_in(s - PAD_CUT, e + PAD_CUT):
                s += step
                continue
            if any(not (e + PAD_USED <= u0 or s >= u1 + PAD_USED) for u0, u1 in used):
                s += step
                continue
            if avoid_shot and avoid_shot[0] - 0.01 <= s <= avoid_shot[1] + 0.01:
                s += step
                continue
            x0, _ = sc.crop_x(s, e, crop_w)
            f = sc.window(s, e, (x0, crop_w))
            if f["n"]:
                score = sc.score(f) + bonus + first_bonus
                if best is None or score > best["score"]:
                    best = {"src_start": round(s, 3), "src_end": round(e, 3), "score": round(score, 3), "feat": f,
                            "bonus": bonus}
            s += step
    return best


def shot_of(sc: Scan, t: float) -> Tuple[float, float]:
    """The source shot (cut to cut) around time t."""
    i = bisect.bisect_right(sc.cuts, t)
    a = sc.cuts[i - 1] if i > 0 else sc.t[0]
    b = sc.cuts[i] if i < len(sc.cuts) else sc.t[-1]
    return a, b


def plan(tl: Dict[str, Any], sc: Scan, by_id: Dict[int, Dict[str, Any]], k_shots: Dict[str, Dict[str, Any]],
         episode: Tuple[float, float], crop_w: int, hook_t: Optional[float] = None) -> Dict[str, Any]:
    """Shots covering [0, end_card start) exactly. The end card shows the last shot's last frame (render)."""
    a_ep, b_ep = episode
    total = tl["end_card"][0]
    gaps = word_gaps(tl)
    segs = tl["segments"]
    shots: List[Dict[str, Any]] = []
    used: List[Tuple[float, float]] = []
    warns: List[str] = []
    prev_shot: Optional[Tuple[float, float]] = None
    for si, seg in enumerate(segs):
        a = 0.0 if si == 0 else seg["t0"]
        b = segs[si + 1]["t0"] if si + 1 < len(segs) else total
        cuts = split_span(a, b, tl["total"], si == 0, gaps)
        bounds = [a] + cuts + [b]
        span = src_span(seg, by_id)
        picks = [k_shots[k] for k in seg.get("shots") or [] if k in k_shots]
        for j in range(len(bounds) - 1):
            t0, t1 = bounds[j], bounds[j + 1]
            d = round(t1 - t0, 4)
            regions: List[Tuple[float, float, float]] = []
            for p in picks:
                regions.append((p["start"], p["end"], 1.6))
            if span:
                regions.append((max(a_ep, span[0] - 4.0), min(b_ep, span[1] + 4.0), 1.0))
                regions.append((max(a_ep, span[0] - 30.0), min(b_ep, span[1] + 30.0), 0.0))
            if si == 0 and hook_t is not None:
                regions.insert(0, (max(a_ep, hook_t - 6.0), min(b_ep, hook_t + 8.0), 1.4))
            regions.append((a_ep, b_ep, -0.8))
            best = choose(sc, regions, d, used, crop_w, prev_shot, first_bonus=0.0)
            if best is None:
                best = choose(sc, [(a_ep, b_ep, -0.8)], d, [], crop_w)
                warns.append(f"S{seg['n']} shot {j + 1}: reused material")
            if best is None:
                mid = (span[0] + span[1]) / 2 if span else (a_ep + b_ep) / 2
                s0 = min(max(a_ep, mid - d / 2), b_ep - d)
                best = {"src_start": round(s0, 3), "src_end": round(s0 + d, 3), "score": -9.0, "feat": sc.window(s0, s0 + d),
                        "bonus": -9}
                warns.append(f"S{seg['n']} shot {j + 1}: no cut-free window")
            s0, s1 = best["src_start"], best["src_end"]
            used.append((s0, s1))
            prev_shot = shot_of(sc, (s0 + s1) / 2)
            x0, x1 = sc.crop_x(s0, s1, crop_w)
            fx, fy = sc.face_rel(s0, s1, (x0 + x1) / 2.0, crop_w)
            f = best["feat"]
            static = sc.is_static(f)
            shots.append({"i": len(shots) + 1, "seg": seg["n"], "fn": seg["fn"], "t0": round(t0, 4), "t1": round(t1, 4),
                          "src_start": s0, "src_end": s1, "score": best["score"],
                          "why": {1.6: "writer pick", 1.4: "hook moment", 1.0: "the moment it retells", 0.0: "near it",
                                  -0.8: "storyline"}.get(best["bonus"], "fallback"),
                          "face": round(f["face"], 3), "presence": round(f["presence"], 2), "motion": round(f["motion"], 3),
                          "push_in": 0.05 if static else 0.0, "crop_x0": x0, "crop_x1": x1, "fx": round(fx, 3),
                          "fy": round(fy, 3)})
    return {"shots": shots, "warnings": warns, "stats": stats(shots, total)}


def stats(shots: Sequence[Dict[str, Any]], total: float) -> Dict[str, Any]:
    d = [s["t1"] - s["t0"] for s in shots]
    if not d:
        return {}
    why: Dict[str, int] = {}
    for s in shots:
        why[s["why"]] = why.get(s["why"], 0) + 1
    return {"shots": len(d), "min": round(min(d), 2), "max": round(max(d), 2), "avg": round(sum(d) / len(d), 2),
            "faces": round(sum(1 for s in shots if s["presence"] >= 0.5) / len(d), 2), "why": why,
            "push_in": sum(1 for s in shots if s["push_in"])}


def sub_timeline(tl: Dict[str, Any], keep: Sequence[int], shots: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """A short version: only the segments `keep` (in that order), re-timed back to back, each with ITS shots of the main
    plan shifted (the voice of a segment is the same, so are its shots)."""
    by_n = {s["n"]: s for s in tl["segments"]}
    segs = [by_n[n] for n in keep if n in by_n]
    voice = {s["n"]: {"sec": s["t1"] - s["t0"], "wav": s["wav"], "words": [(w, a - s["t0"], b - s["t0"]) for w, a, b in s["words"]]}
             for s in segs}
    sub = timeline([{"n": s["n"], "fn": "hook" if k == 0 else s["fn"], "text": s["text"], "src": s["src"], "shots": s["shots"]}
                    for k, s in enumerate(segs)], voice)
    out: List[Dict[str, Any]] = []
    total = sub["end_card"][0]
    for k, s in enumerate(sub["segments"]):
        orig = by_n[s["n"]]
        a_new = 0.0 if k == 0 else s["t0"]
        b_new = sub["segments"][k + 1]["t0"] if k + 1 < len(sub["segments"]) else total
        own = [x for x in shots if x["seg"] == s["n"]]
        if not own:
            continue
        shift = s["t0"] - orig["t0"]
        pieces = []
        for x in own:
            t0, t1 = x["t0"] + shift, x["t1"] + shift
            t0, t1 = max(t0, a_new), min(t1, b_new)
            if t1 - t0 > 0.2:
                pieces.append(dict(x, t0=round(t0, 4), t1=round(t1, 4)))
        if not pieces:
            continue
        # stretch the first/last piece to fill the segment span exactly (lead-in / gaps differ by a few frames)
        pieces[0]["t0"] = round(a_new, 4)
        pieces[-1]["t1"] = round(b_new, 4)
        for p in pieces:
            # the source window follows the screen time (start kept, length = screen length)
            p["src_end"] = round(p["src_start"] + (p["t1"] - p["t0"]), 3)
        out += pieces
    for i, x in enumerate(out):
        x["i"] = i + 1
    return sub, out
