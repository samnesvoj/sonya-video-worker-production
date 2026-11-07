"""
timeline.py — deterministic half of SONYA Trailer: no LLM, no ffmpeg, pure functions (tested).

  sentences()        ASR words → sentences S1..Sn (boundaries.group_sentences)
  shots()            scene cuts → shots C1..Cn
  chunks()           source → chunks for the map step (sentences / shots inside each)
  beat_to_fragment() Director beat → fragment (dialogue lines = sentences inside the candidate window)
  refine()           fragment → exact render interval: whole words / sentences, scene cuts,
                     never a flash of the next shot, never mid-word
  dedupe()           drop overlapping / near-identical neighbours
  fit_to_target()    drop the weakest middle fragments until the sum fits the target
  voice_rooms()      seconds available for a voice line per fragment (narration: the shot; both: the
                     longest pause between words) → Writer budget via the calibrated TTS model
  place_narration()  measured TTS length → fragment length (extend on natural bounds) + line offsets
  mix_plan()         per-fragment audio plan (dialogue 0 dB, no voice; narration/both: original −14 dB
                     under the voice only)
  duck_expr()        ffmpeg volume expression built from the mix plan
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.shared.boundaries import Word, group_sentences, normalize_words

VERSION = "trailer-timeline-v2"
DUCK_DB = -14.0                  # spec: original −12…−16 dB under narration
EPS = 1e-6


# ── inputs ──────────────────────────────────────────────────────────────────

def sentences(raw_words: Sequence[Dict[str, Any]]) -> Tuple[List[Word], List[Dict[str, Any]]]:
    words, _ = normalize_words(raw_words or [])
    out = []
    for i, (a, b) in enumerate(group_sentences(words, pause_sec=0.7)):
        text = " ".join(w.text for w in words[a:b + 1]).strip()
        out.append({"id": f"S{i + 1}", "first": a, "last": b, "start": round(words[a].start, 2),
                    "end": round(words[b].end, 2), "text": text})
    return words, out


def shots(cuts: Sequence[float], duration: float) -> List[Dict[str, Any]]:
    pts = [0.0] + sorted(c for c in cuts if 0.2 < c < duration - 0.2) + [duration]
    return [{"id": f"C{i + 1}", "start": round(a, 2), "end": round(b, 2)}
            for i, (a, b) in enumerate(zip(pts, pts[1:])) if b - a > 0.05]


def chunks(duration: float, sents: List[Dict[str, Any]], shot_list: List[Dict[str, Any]],
           chunk_sec: float = 1080.0, overlap: float = 15.0) -> List[Dict[str, Any]]:
    """Chunks of ~chunk_sec (spec: 15–20 min) with a small overlap; edges moved to the nearest sentence
    start so no sentence is split. A source up to 1.25 × chunk_sec is one chunk."""
    n = max(1, int(round(duration / chunk_sec)))
    edges = [0.0]
    for k in range(1, n):
        t = duration * k / n
        gaps = [s["start"] for s in sents if abs(s["start"] - t) < 60]
        edges.append(min(gaps, key=lambda g: abs(g - t)) if gaps else t)
    edges.append(duration)
    out = []
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        a0 = max(0.0, a - overlap) if i else a
        out.append({"index": i, "total": n, "start": a0, "end": b,
                    "sentences": [s for s in sents if a0 - EPS <= s["start"] < b],
                    "shots": [c for c in shot_list if c["end"] > a0 and c["start"] < b]})
    return out


def beat_to_fragment(beat: Dict[str, Any], sents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """dialogue / both → the sentences whose middle lies in the candidate window (at least one: the most
    overlapping); narration → the window as is (refine() moves it off words and onto cuts)."""
    s, e = beat["start_candidate"], beat["end_candidate"]
    f = dict(beat, id=beat["beat_id"], audio=beat["audio_owner"], start=s, end=e, lines=[])
    if beat["audio_owner"] in ("dialogue", "both"):
        inside = [x for x in sents if s - EPS <= (x["start"] + x["end"]) / 2 <= e + EPS]
        if not inside:
            ov = [(min(e, x["end"]) - max(s, x["start"]), x) for x in sents]
            ov = [o for o in ov if o[0] > 0]
            inside = [max(ov, key=lambda o: o[0])[1]] if ov else []
        if inside:
            f["lines"] = [x["id"] for x in inside]
        else:
            f["audio"] = "narration"
            f["downgraded"] = f"{beat['audio_owner']}→narration: no speech in {s:.1f}-{e:.1f}"
    return f


# ── boundaries ──────────────────────────────────────────────────────────────

def _word_at(t: float, words: Sequence[Word]) -> Optional[Word]:
    for w in words:
        if w.start - EPS < t < w.end + EPS and w.start < t - 0.02 and t < w.end - 0.02:
            return w
    return None


def _no_word_across(t: float, words: Sequence[Word]) -> bool:
    return _word_at(t, words) is None


def refine(frag: Dict[str, Any], words: Sequence[Word], sents: Dict[str, Dict[str, Any]],
           cuts: Sequence[float], duration: float, *, max_dialogue: float = 12.0,
           narration_len: Tuple[float, float] = (2.0, 8.0)) -> Dict[str, Any]:
    """Returns a copy with render start/end and the reasons for every move."""
    f = dict(frag)
    reasons: List[str] = []
    s, e = float(f["start"]), float(f["end"])
    cuts = sorted(cuts)
    if f["audio"] in ("dialogue", "both") and f.get("lines"):
        ss = [sents[k] for k in f["lines"]]
        s, e = min(x["start"] for x in ss), max(x["end"] for x in ss)
        if e - s > max_dialogue:                       # keep the first sentences that fit
            keep = [x for x in sorted(ss, key=lambda x: x["start"]) if x["end"] - s <= max_dialogue]
            if keep:
                e = keep[-1]["end"]
                f["lines"] = [x["id"] for x in keep]
            else:
                e = s + max_dialogue
            reasons.append("dialogue_trimmed_to_max")
        first = next((w for w in words if w.end > s + EPS), None)
        last = [w for w in words if w.start < e - EPS]
        prev_end = max((w.end for w in words if w.end <= s + EPS), default=0.0)
        next_start = min((w.start for w in words if w.start >= e - EPS), default=duration)
        s = max(prev_end + 0.02, s - 0.15, 0.0) if first else s
        e = min(next_start - 0.02, e + 0.35, duration) if last else e
        reasons.append("snapped_to_sentences")
    else:
        lo, hi = narration_len
        if e - s < lo:
            e = s + lo
        if e - s > hi:
            e = s + hi
        # start on the shot cut if one is near (cleaner than a random frame)
        near = [c for c in cuts if abs(c - s) <= 1.0 and c < e - lo + 0.5]
        if near:
            s = min(near, key=lambda c: abs(c - s))
            reasons.append("start_on_shot_cut")
        w = _word_at(s, words)
        if w:
            s = w.end + 0.02 if w.end - s < 0.6 else w.start - 0.02
            reasons.append("start_moved_off_word")
        w = _word_at(e, words)
        if w:
            e = w.end + 0.05 if w.end - e < 0.8 else max(s + lo, w.start - 0.02)
            reasons.append("end_moved_off_word")
    # never flash the next / previous shot for a few frames
    for c in cuts:
        if s < c < s + 0.35 and _no_word_across(c, words):
            s = c
            reasons.append("start_moved_past_flash_cut")
        if e - 0.35 < c < e and c - s > 1.0:
            # a dialogue fragment keeps its last word: move only if no speech ends after the cut
            speech_after = any(c < x.end <= e + EPS for x in words)
            if f["audio"] == "narration" or not speech_after:
                e = c - 0.04
                reasons.append("end_before_flash_cut")
    s, e = max(0.0, s), min(duration, e)
    f.update({"render_start": round(s, 3), "render_end": round(e, 3), "boundary_reasons": reasons,
              "semantic_start": frag["start"], "semantic_end": frag["end"],
              "cuts_inside_word": (not _no_word_across(s, words)) or (not _no_word_across(e, words))})
    return f


def dedupe(frags: List[Dict[str, Any]], min_gap: float = 1.0) -> Tuple[List[Dict[str, Any]], List[str]]:
    kept: List[Dict[str, Any]] = []
    dropped = []
    for f in frags:
        a, b = f["render_start"], f["render_end"]
        clash = None
        for k in kept:
            ov = min(b, k["render_end"]) - max(a, k["render_start"])
            if ov > 0.3 * min(b - a, k["render_end"] - k["render_start"]):
                clash = f"overlaps {k['id']}"
                break
        if not clash and kept:
            p = kept[-1]
            if f["event_ids"][:1] == p["event_ids"][:1] and f["audio"] == p["audio"] \
                    and abs(a - p["render_end"]) < min_gap:
                clash = f"near-identical neighbour of {p['id']}"
        if clash:
            dropped.append(f"{f['id']}: {clash}")
        else:
            kept.append(f)
    return kept, dropped


def total(frags: Sequence[Dict[str, Any]]) -> float:
    return round(sum(f["render_end"] - f["render_start"] for f in frags), 3)


def fit_to_target(frags: List[Dict[str, Any]], target: float, tol: float = 0.12) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Over target → drop the lowest-priority middle fragment (hook and tease stay), repeat."""
    frags = list(frags)
    dropped = []
    while total(frags) > target * (1 + tol) and len(frags) > 3:
        middle = list(range(1, len(frags) - 1))
        i = min(middle, key=lambda j: (frags[j]["priority"], -frags[j].get("spoiler_level", 0), -j))
        dropped.append(f"{frags[i]['id']}: over target")
        frags.pop(i)
    return frags, dropped


def speech_gaps(a: float, b: float, words: Sequence[Word], min_gap: float = 0.0) -> List[Tuple[float, float]]:
    """Silent intervals (no ASR word) inside [a, b]."""
    inside = sorted((w for w in words if w.end > a and w.start < b), key=lambda w: w.start)
    gaps, t = [], a
    for w in inside:
        if w.start - t >= min_gap:
            gaps.append((t, w.start))
        t = max(t, w.end)
    if b - t >= min_gap:
        gaps.append((t, b))
    return [(round(x, 3), round(y, 3)) for x, y in gaps if y - x > EPS]


def voice_rooms(frags: List[Dict[str, Any]], words: Sequence[Word], lead: float = 0.3, tail: float = 0.35,
                max_extend: float = 2.5, min_both_gap: float = 1.6) -> List[str]:
    """voice_room_sec per fragment. narration: the shot (+ half the allowed extension) minus lead/tail.
    both: the longest pause between words (the voice must not cover a line); too short → dialogue."""
    notes = []
    for f in frags:
        f.pop("voice_room_sec", None)
        if f["audio"] == "narration":
            f["voice_room_sec"] = round(f["render_end"] - f["render_start"] + max_extend * 0.5 - lead - tail, 2)
        elif f["audio"] == "both":
            gaps = speech_gaps(f["render_start"], f["render_end"], words)
            best = max(gaps, key=lambda g: g[1] - g[0], default=None)
            if best and best[1] - best[0] >= min_both_gap:
                f["voice_gap"] = best
                f["voice_room_sec"] = round(best[1] - best[0] - 0.4, 2)
            else:
                f["audio"] = "dialogue"
                notes.append(f"{f['id']}: both→dialogue (longest pause "
                             f"{(best[1] - best[0]) if best else 0:.1f}s < {min_both_gap}s)")
    return notes


def place_narration(frags: List[Dict[str, Any]], tts_sec: Dict[str, float], words: Sequence[Word],
                    cuts: Sequence[float], duration: float, lead: float = 0.3, tail: float = 0.35,
                    max_extend: float = 2.5) -> List[str]:
    """Make every narration fragment long enough for its measured voice line: extend the end
    (to a cut or a word end when one is near), at most max_extend; record what could not fit."""
    issues = []
    taken = [(f["render_start"], f["render_end"]) for f in frags]
    for i, f in enumerate(frags):
        sec = tts_sec.get(f["id"])
        if sec is None:
            continue
        if f["audio"] == "both":
            ga, gb = f["voice_gap"]
            if sec > gb - ga - 0.2:
                issues.append(f"{f['id']}: voice {sec:.1f}s does not fit the pause {gb - ga:.1f}s — line dropped")
                f["voice_dropped"] = True
                continue
            f["voice_offset"] = round(ga - f["render_start"] + (gb - ga - sec) / 2, 3)
            f["voice_sec"] = round(sec, 3)
            continue
        need = lead + sec + tail
        have = f["render_end"] - f["render_start"]
        if need > have + EPS:
            want = f["render_start"] + min(need, have + max_extend)
            nxt = [a for j, (a, b) in enumerate(taken) if j != i and a >= f["render_end"] - EPS]
            limit = min([duration] + nxt)
            new_end = min(want, limit)
            near_cut = [c for c in cuts if new_end < c <= new_end + 0.6 and c <= limit]
            if near_cut:
                new_end = near_cut[0] - 0.04
            w = _word_at(new_end, words)
            if w and w.end <= limit:
                new_end = w.end + 0.05
            f["render_end"] = round(new_end, 3)
            f.setdefault("boundary_reasons", []).append("extended_for_voice")
            if f["render_end"] - f["render_start"] < need - 0.05:
                issues.append(f"{f['id']}: voice {sec:.1f}s > room {f['render_end'] - f['render_start'] - lead - tail:.1f}s")
        f["voice_offset"] = lead
        f["voice_sec"] = round(sec, 3)
    return issues


def layout(frags: List[Dict[str, Any]]) -> float:
    """Trailer-time position of every fragment (t0/t1); returns the total length."""
    t = 0.0
    for f in frags:
        f["t0"] = round(t, 3)
        t += f["render_end"] - f["render_start"]
        f["t1"] = round(t, 3)
    return round(t, 3)


def voice_windows(frags: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    return [(round(f["t0"] + f["voice_offset"], 3), round(f["t0"] + f["voice_offset"] + f["voice_sec"], 3))
            for f in frags if f.get("voice_sec") and not f.get("voice_dropped")]


def mix_plan(frags: Sequence[Dict[str, Any]], duck_db: float = DUCK_DB) -> Dict[str, Any]:
    """What every fragment does with sound (trailer time). dialogue: original 0 dB, no voice;
    narration / both: voice in its window, original ducked to duck_db only under it."""
    segs = []
    for f in frags:
        v = None
        if f.get("voice_sec") and not f.get("voice_dropped"):
            v = [round(f["t0"] + f["voice_offset"], 3), round(f["t0"] + f["voice_offset"] + f["voice_sec"], 3)]
        segs.append({"id": f["id"], "audio_owner": f["audio"], "t0": f["t0"], "t1": f["t1"],
                     "original_db": 0.0, "original_under_voice_db": duck_db if v else None, "voice": v})
    voiced = sum(s["voice"][1] - s["voice"][0] for s in segs if s["voice"])
    length = segs[-1]["t1"] if segs else 0.0
    return {"duck_db": duck_db, "segments": segs, "voice_sec": round(voiced, 2),
            "voice_share": round(voiced / length, 3) if length else 0.0}


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def duck_expr(windows: Sequence[Tuple[float, float]], depth_db: float = DUCK_DB, ramp: float = 0.25,
              pre: float = 0.15, post: float = 0.3) -> str:
    """volume='…' for ffmpeg (eval=frame): 1.0 outside the voice, depth_db under it, linear ramps."""
    if not windows:
        return "1"
    depth = db_to_gain(depth_db)
    terms = []
    for a, b in windows:
        a, b = a - pre, b + post
        terms.append(f"clip(min((t-{a - ramp:.3f})/{ramp},({b + ramp:.3f}-t)/{ramp}),0,1)")
    d = terms[0]
    for t in terms[1:]:
        d = f"max({d},{t})"
    return f"1-{1 - depth:.4f}*{d}"


def subtitle_words(frags: Sequence[Dict[str, Any]], lines: Dict[str, str]):
    """Narration subtitles only (spec): each line spread over its measured voice window by characters."""
    from scripts.shared.subtitles import Word as SubWord
    out = []
    for f in frags:
        text = lines.get(f["id"])
        if not text or not f.get("voice_sec") or f.get("voice_dropped"):
            continue
        a = f["t0"] + f["voice_offset"]
        b = a + f["voice_sec"]
        toks = text.split()
        total_c = sum(len(x) + 1 for x in toks)
        t = a
        for x in toks:
            d = (b - a) * (len(x) + 1) / total_c
            out.append(SubWord(x, round(t, 3), round(t + d, 3)))
            t += d
    out.sort(key=lambda w: w.start)
    return out
