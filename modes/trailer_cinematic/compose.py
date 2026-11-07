"""
compose.py — Timeline Composer of trailer_cinematic v2 (pure functions + the voice-track writer).

The voice is synthesised FIRST; the picture is built to fit the measured voice, never the other way round:

  voice_timeline()   measured sentence durations + pauses → absolute time of every sentence / segment
  compose()          Director's shots per segment → contiguous shots covering [0, total) exactly on the 30 fps grid;
                     durations scaled to each segment's span, boundaries snapped to sentence ends, material that
                     runs out is replaced from the Director's alternates / neutral unused windows
  write_voice()      the ONLY audio of the final video: the narrator's sentences + silence
  stability()        per-sentence loudness spread (stable volume check input)
"""
from __future__ import annotations

import wave
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .recap import MAX_SHOT, MIN_SHOT, PAUSE_AFTER, SENT_PAUSE, SENT_PAUSE_Q, sentences_of, shot_cap

FPS = 30
LEAD_IN = 0.5          # picture before the first word
TAIL = 1.2             # picture after the last word (fade-out sits inside it)
SNAP = 0.7             # a shot cut may move this far to land on a sentence end
CUT_MARGIN = 0.15      # never use material closer than this to a scene cut of the source


def q(t: float) -> float:
    """Snap to the frame grid."""
    return round(round(t * FPS) / FPS, 4)


# ── voice ───────────────────────────────────────────────────────────────────

def voice_timeline(script: Dict[str, Any], sent_sec: Dict[str, List[float]], *, lead_in: float = LEAD_IN,
                   tail: float = TAIL, pause_scale: float = 1.0, pauses: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """sent_sec[segment_id] = measured seconds of each sentence. Returns segments with absolute times.
    pauses (style): {"sent": s, "sent_q": s, "after": {short|normal|long: s}}; default = the calm documentary pauses."""
    sp = (pauses or {}).get("sent", SENT_PAUSE)
    sq = (pauses or {}).get("sent_q", SENT_PAUSE_Q)
    after = (pauses or {}).get("after", PAUSE_AFTER)
    t = lead_in
    segs = []
    for i, seg in enumerate(script["segments"]):
        sents = sentences_of(seg["text"])
        secs = sent_sec[seg["id"]]
        assert len(sents) == len(secs), (seg["id"], len(sents), len(secs))
        rows, t0 = [], t
        for j, (text, d) in enumerate(zip(sents, secs)):
            rows.append({"text": text, "t0": round(t, 3), "t1": round(t + d, 3), "sec": round(d, 3)})
            t += d
            if j < len(sents) - 1:
                t += (sq if text.endswith("?") else sp) * pause_scale
        segs.append({"id": seg["id"], "t0": round(t0, 3), "t1": round(t, 3), "sentences": rows,
                     "pause_after": 0.0 if i == len(script["segments"]) - 1 else after[seg["pause_after"]] * pause_scale})
        t += segs[-1]["pause_after"]
    voice_end = segs[-1]["t1"]
    return {"segments": segs, "voice_end": round(voice_end, 3), "total": q(voice_end + tail), "lead_in": lead_in,
            "voice_sec": round(sum(r["sec"] for s in segs for r in s["sentences"]), 3)}


def segment_spans(vt: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Picture span of every segment: from its first word (the first one from 0) to the next segment's first word
    (the last one to the end of the video)."""
    segs = vt["segments"]
    out = {}
    for i, s in enumerate(segs):
        a = 0.0 if i == 0 else s["t0"]
        b = vt["total"] if i == len(segs) - 1 else segs[i + 1]["t0"]
        out[s["id"]] = (q(a), q(b))
    return out


def segment_seconds(vt: Dict[str, Any]) -> Dict[str, float]:
    return {k: round(b - a, 2) for k, (a, b) in segment_spans(vt).items()}


def write_voice(vt: Dict[str, Any], sent_wavs: Dict[Tuple[str, int], str], out_wav: str, sr: int = 48000) -> str:
    import numpy as np
    buf = np.zeros(int((vt["total"] + 0.5) * sr), dtype=np.float32)
    for s in vt["segments"]:
        for j, r in enumerate(s["sentences"]):
            with wave.open(sent_wavs[(s["id"], j)], "rb") as w:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
                if w.getframerate() != sr:
                    idx = (np.arange(int(len(data) * sr / w.getframerate())) * w.getframerate() / sr).astype(int)
                    data = data[np.clip(idx, 0, len(data) - 1)]
            a = int(round(r["t0"] * sr))
            b = min(len(buf), a + len(data))
            buf[a:b] += data[:b - a]
    pcm = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_wav


def stability(vt: Dict[str, Any], voice_wav: str) -> Dict[str, Any]:
    """RMS (dBFS) of every sentence on the placed voice track; spread = max − min."""
    import numpy as np
    with wave.open(voice_wav, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    vals = []
    for s in vt["segments"]:
        for r in s["sentences"]:
            seg = x[int(r["t0"] * sr):int(r["t1"] * sr)]
            if len(seg):
                vals.append(20 * np.log10(max(1e-6, float(np.sqrt(np.mean(seg ** 2))))))
    return {"per_sentence_db": [round(v, 1) for v in vals], "spread_db": round(max(vals) - min(vals), 2) if vals else 0.0}


# ── picture ─────────────────────────────────────────────────────────────────

def material(c: Dict[str, Any]) -> Tuple[float, float]:
    """Usable source range of a candidate: its window, extendable inside its own scene shot (no cut crossing)."""
    return (max(c["start"], c["shot_start"] + CUT_MARGIN) if c["shot_start"] > 0 else c["start"],
            min(c["end"], c["shot_end"] - CUT_MARGIN))


def extendable(c: Dict[str, Any]) -> Tuple[float, float]:
    return (c["shot_start"] + CUT_MARGIN if c["shot_start"] > 0 else 0.0, c["shot_end"] - CUT_MARGIN)


def fit_source(c: Dict[str, Any], dur: float) -> Tuple[float, float]:
    """Source interval of length ≤ dur for candidate c: the analysed window first, grown symmetrically inside the
    same scene shot only if needed. May return less than dur when the shot is short."""
    a, b = material(c)
    if b - a >= dur:
        mid = (a + b) / 2          # centred in the analysed window: its frames are what the model saw
        s = max(a, min(b - dur, mid - dur / 2))
        return round(s, 3), round(s + dur, 3)
    lo, hi = extendable(c)
    s = max(lo, min(a, (a + b) / 2 - dur / 2))
    e = min(hi, s + dur)
    s = max(lo, e - dur)
    return round(s, 3), round(e, 3)


def _scale_durations(base: Sequence[float], span: float, caps: Optional[Sequence[float]] = None) -> List[float]:
    """Scale to sum exactly to span with each in [MIN_SHOT, cap] (iterative clamp-redistribute)."""
    n = len(base)
    caps = list(caps) if caps else [MAX_SHOT] * n
    out = [float(b) for b in base]
    free = set(range(n))
    for _ in range(10):
        fixed = sum(out[i] for i in range(n) if i not in free)
        cur = sum(out[i] for i in free)
        if not free or cur <= 0:
            break
        k = (span - fixed) / cur
        changed = False
        for i in list(free):
            v = out[i] * k
            if v < MIN_SHOT or v > caps[i]:
                out[i] = min(caps[i], max(MIN_SHOT, v))
                free.discard(i)
                changed = True
            else:
                out[i] = v
        if not changed:
            break
    return out


def _snap_bounds(t0: float, durs: List[float], sentence_ends: Sequence[float]) -> List[float]:
    """Cumulative cut times inside a segment, each snapped to a nearby sentence end when both neighbours stay ≥ MIN_SHOT."""
    cuts, t = [], t0
    for d in durs[:-1]:
        t += d
        cuts.append(t)
    snapped = []
    prev = t0
    for i, c in enumerate(cuts):
        near = [e for e in sentence_ends if abs(e - c) <= SNAP and e - prev >= MIN_SHOT]
        v = min(near, key=lambda e: abs(e - c)) if near else c
        snapped.append(v)
        prev = v
    return snapped


def _who(c: Dict[str, Any]) -> set:
    return {str(p.get("who") or "").lower() for p in c.get("people") or [] if p.get("who")}


def relevance(c: Dict[str, Any], refs: Sequence[Dict[str, Any]]) -> float:
    """How well window c can stand in for the director's own picks `refs` of a segment: the same people on screen and
    the same story events weigh most, then closeness in the source. (Observed: the spare for "federal agents on the
    yacht" was the most "interesting" window — the hero's wife — because the pool was ordered by interest only.)"""
    if not refs:
        return 0.0
    who = set().union(*(_who(r) for r in refs))
    ev = set().union(*(set(r.get("event_ids") or []) for r in refs))
    dt = min(abs(c["start"] - r["start"]) for r in refs)
    return 3.0 * len(_who(c) & who) + 2.0 * len(set(c.get("event_ids") or []) & ev) - min(dt, 1800.0) / 600.0


def compose(plan: Dict[str, Any], cands: Dict[str, Dict[str, Any]], vt: Dict[str, Any]) -> Dict[str, Any]:
    """Contiguous shot list covering [0, total). Raises if a segment cannot be filled even from fallbacks."""
    spans = segment_spans(vt)
    seg_by = {s["id"]: s for s in vt["segments"]}
    used = {sh["candidate_id"] for p in plan["segments"] for sh in p["shots"]}
    key = lambda c: -(c["visual_interest"] * 10 + c["event_weight"])                 # noqa: E731
    # fillers: silent / listening first, then "uncertain" (a long film is mostly talking heads and the silent pool
    # runs dry); windows that probably speak are the very last resort and are flagged in the warnings
    pool = (sorted((c for c in cands.values() if c["usable"] and c["id"] not in used
                    and c["speaking"]["state"] in ("not_speaking", "likely_listening")), key=key)
            + sorted((c for c in cands.values() if c["usable"] and c["id"] not in used
                      and c["speaking"]["state"] == "uncertain"), key=key))
    last_resort = sorted((c for c in cands.values() if c["usable"] and c["id"] not in used
                          and c["speaking"]["state"] == "likely_speaking"), key=key)
    shots, warnings = [], []
    for p in plan["segments"]:
        a, b = spans[p["segment_id"]]
        span = b - a
        order = [dict(s) for s in p["shots"] if cands[s["candidate_id"]]["usable"]]   # usable may flip after planning
        alts = [x for x in p.get("alternates", []) if x in cands and cands[x]["usable"]]
        shown = {sh["candidate_id"] for sh in shots}
        for o in order:
            # the director may mark a repeat reuse_ok; the same footage twice in one minute reads as a glitch (observed:
            # "Jordan crawling" under two different lines) → the closest unused window in meaning, if there is one
            if o["candidate_id"] in shown:
                refs0 = [cands[o["candidate_id"]]]
                pick = next((x for x in alts if x not in shown and x not in {q["candidate_id"] for q in order}), None)
                if pick is None:
                    pick = next((c["id"] for c in sorted(pool, key=lambda c: -relevance(c, refs0))
                                 if c["id"] not in shown), None)
                if pick is not None:
                    warnings.append(f"{p['segment_id']}: repeat of {o['candidate_id']} → {pick}")
                    pool = [x for x in pool if x["id"] != pick]
                    alts = [x for x in alts if x != pick]
                    o.update(candidate_id=pick, fallback=True, reuse_ok=False)
        refs = [cands[o["candidate_id"]] for o in p["shots"] if o["candidate_id"] in cands]
        # fillers for THIS segment: the director's alternates, then neutral windows closest in meaning to its own picks
        # (speaking state stays the primary tier — a muted talking head is worse than a near-miss in meaning)
        tier = {"not_speaking": 0, "likely_listening": 0, "uncertain": 1}
        pool.sort(key=lambda c: (tier.get(c["speaking"]["state"], 2), -relevance(c, refs), key(c)))
        last_resort.sort(key=lambda c: -relevance(c, refs))
        # too many shots for the span → drop from the end; too few for MAX_SHOT → add from alternates / pool
        while len(order) > 1 and span < MIN_SHOT * len(order):
            order.pop()
        capof = lambda o: shot_cap(cands[o["candidate_id"]])          # noqa: E731
        while span > sum(capof(o) for o in order) + 1e-6:
            nxt = next((x for x in alts if x not in {o["candidate_id"] for o in order}), None)
            if nxt is None and pool:
                nxt = pool.pop(0)["id"]
            if nxt is None:
                raise RuntimeError(f"{p['segment_id']}: not enough material for {span:.1f}s")
            order.append({"candidate_id": nxt, "seconds": 3.0, "why": "fill (alternate/neutral)", "fallback": True})
            warnings.append(f"{p['segment_id']}: filled with {nxt}")
        durs = _scale_durations([o["seconds"] for o in order], span, [capof(o) for o in order])
        ends = [r["t1"] for r in seg_by[p["segment_id"]]["sentences"]]
        cuts = _snap_bounds(a, durs, ends)
        bounds = [a] + [q(c) for c in cuts] + [b]
        for i, o in enumerate(order):
            d = round(bounds[i + 1] - bounds[i], 4)
            c = cands[o["candidate_id"]]
            s, e = fit_source(c, d)
            if e - s < d - 1e-3:
                # shot too short: swap for an alternate/neutral window that can supply the whole duration
                for alt in [x for x in alts if x != c["id"]] + [x["id"] for x in pool] + [x["id"] for x in last_resort]:
                    cc = cands[alt]
                    s2, e2 = fit_source(cc, d)
                    if e2 - s2 >= d - 1e-3 and cc["usable"] and alt not in {x["candidate_id"] for x in shots}:
                        warnings.append(f"{p['segment_id']}: {c['id']} too short ({e - s:.1f}s < {d:.1f}s) → {alt}")
                        pool = [x for x in pool if x["id"] != alt]
                        last_resort = [x for x in last_resort if x["id"] != alt]
                        c, s, e, o = cc, s2, e2, dict(o, candidate_id=alt, fallback=True)
                        break
                else:
                    raise RuntimeError(f"{p['segment_id']}: no material for a {d:.1f}s shot")
            shots.append({"n": len(shots) + 1, "segment_id": p["segment_id"], "candidate_id": c["id"],
                          "src_start": s, "src_end": round(s + d, 3), "t0": bounds[i], "t1": bounds[i + 1],
                          "duration": d, "why": o.get("why", ""), "fallback": bool(o.get("fallback")),
                          "scene_shot": c["shot_id"], "speaking": c["speaking"]["state"]})
    return {"shots": shots, "total": vt["total"], "fps": FPS, "warnings": warnings}


def check_plan(tl: Dict[str, Any]) -> List[str]:
    """Contiguity / coverage problems of a composed timeline (empty = fine)."""
    bad, t = [], 0.0
    for s in tl["shots"]:
        if abs(s["t0"] - t) > 1e-3:
            bad.append(f"shot {s['n']}: starts at {s['t0']} but previous ended at {t}")
        if s["t1"] <= s["t0"]:
            bad.append(f"shot {s['n']}: empty")
        t = s["t1"]
    if abs(t - tl["total"]) > 1e-3:
        bad.append(f"timeline ends at {t}, voice track is {tl['total']}")
    return bad


def reuse_report(tl: Dict[str, Any]) -> Dict[str, Any]:
    ids = [s["candidate_id"] for s in tl["shots"]]
    back = [b["n"] for a, b in zip(tl["shots"], tl["shots"][1:]) if a["scene_shot"] == b["scene_shot"]]
    return {"shots": len(ids), "reused_candidates": sorted({i for i in ids if ids.count(i) > 1}),
            "same_scene_back_to_back": back}


# ── source verification: no hidden scene cut inside a shot ──────────────────

def cut_free_window(cuts: Sequence[float], lo: float, hi: float, s: float, d: float, margin: float = 0.12) -> Optional[Tuple[float, float]]:
    """Window of length d inside [lo, hi] that contains none of `cuts` (± margin), closest to start s; None if no span
    is long enough. `cuts` are absolute source times found by a frame-accurate detector."""
    pts = [lo] + sorted(c for c in cuts if lo < c < hi) + [hi]
    best = None
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        a2 = a + (margin if i else 0.0)
        b2 = b - (margin if i < len(pts) - 2 else 0.0)
        if b2 - a2 + 1e-6 < d:
            continue
        st = max(a2, min(b2 - d, s))
        cand = (abs(st - s), round(st, 3))
        if best is None or cand < best:
            best = cand
    return (best[1], round(best[1] + d, 3)) if best else None


def verify_sources(tl: Dict[str, Any], cands: Dict[str, Dict[str, Any]], cuts_in, duration: float, spare: Sequence[str] = (),
                   reach: float = 2.0) -> Dict[str, Any]:
    """Frame-accurate check of every shot's source window (cuts_in(a, b) -> absolute cut times in [a, b]).
    A hidden cut → move the window to a cut-free spot nearby; impossible → swap in a spare candidate that is clean.
    Boundaries / durations of the timeline never change."""
    notes = []
    used = {s["candidate_id"] for s in tl["shots"]}
    spare = [x for x in spare if x not in used]
    out_shots: List[Dict[str, Any]] = []

    def clean(cid: str, s0: float, d: float):
        lo, hi = max(0.0, s0 - reach), min(duration, s0 + d + reach)
        return cut_free_window(cuts_in(lo, hi), lo, hi, s0, d)

    def swap_for(d: float, avoid: Sequence[Tuple[float, float]] = ()):
        """(candidate id, window) of a spare that is clean for d seconds, else None."""
        for alt in list(spare):
            c = cands[alt]
            s2, e2 = fit_source(c, d)
            if e2 - s2 < d - 1e-3 or d > shot_cap(c) + 1e-6:
                continue
            w = clean(alt, s2, d)
            if w is not None and not any(w[0] < b and a < w[1] for a, b in avoid):
                return alt, w
        return None

    for sh in tl["shots"]:
        d = sh["duration"]
        ref = cands.get(sh["candidate_id"])
        if ref is not None:                       # a swap should stay close in meaning to the shot it replaces
            spare.sort(key=lambda x: -relevance(cands[x], [ref]))
        w = clean(sh["candidate_id"], sh["src_start"], d)
        if w is not None:
            if abs(w[0] - sh["src_start"]) > 1e-3:
                notes.append(f"shot {sh['n']}: window moved {sh['src_start']:.2f}→{w[0]:.2f} (hidden cut)")
                sh["src_start"], sh["src_end"] = w
            out_shots.append(sh)
            continue
        hit = swap_for(d)
        if hit is not None:
            alt, w = hit
            c = cands[alt]
            notes.append(f"shot {sh['n']}: {sh['candidate_id']} has hidden cuts → {alt}")
            spare.remove(alt)
            sh.update(candidate_id=alt, src_start=w[0], src_end=w[1], scene_shot=c["shot_id"], fallback=True,
                      speaking=c["speaking"]["state"])
            out_shots.append(sh)
            continue
        # a fast-cut film has few clean 4–5 s stretches: split the slot in two shots of ≥ MIN_SHOT, each from a clean
        # window (own candidate first, then a spare). The slot's boundaries do not change.
        if d >= 2 * MIN_SHOT - 1e-6:
            d1 = q(d / 2)
            if d1 < MIN_SHOT:
                d1 = q(MIN_SHOT)
            d2 = round(d - d1, 4)
            w1 = clean(sh["candidate_id"], sh["src_start"], d1)
            first = (sh["candidate_id"], w1) if w1 is not None else swap_for(d1)
            if first is not None and d2 >= MIN_SHOT - 1e-6:
                if first[0] != sh["candidate_id"]:
                    spare.remove(first[0])
                second = swap_for(d2, avoid=[first[1]] if first[0] in spare else [])
                own = clean(sh["candidate_id"], sh["src_start"] + d1, d2)
                if second is None and own is not None and first[0] == sh["candidate_id"] and own[0] >= first[1][1] - 1e-6:
                    second = (sh["candidate_id"], own)
                if second is not None:
                    notes.append(f"shot {sh['n']}: no clean {d:.1f}s window → split into {first[0]} + {second[0]}")
                    if second[0] in spare:
                        spare.remove(second[0])
                    t_mid = round(sh["t0"] + d1, 4)
                    for (cid, ww), t0_, t1_, dd in ((first, sh["t0"], t_mid, d1), (second, t_mid, sh["t1"], d2)):
                        c = cands[cid]
                        out_shots.append(dict(sh, candidate_id=cid, src_start=ww[0], src_end=ww[1], t0=t0_, t1=t1_, duration=dd,
                                              scene_shot=c["shot_id"], fallback=True, speaking=c["speaking"]["state"]))
                    continue
        raise RuntimeError(f"shot {sh['n']}: no cut-free source window for {d:.1f}s")
    for i, sh in enumerate(out_shots):
        sh["n"] = i + 1
    tl["shots"] = out_shots
    tl["warnings"] = list(tl.get("warnings", [])) + notes
    return tl
