"""
visual.py — Visual Analyst of trailer_cinematic v2 (docs/TRAILER_V2_RECAP_SPEC.md §5).

Two-stage selection, the source is NEVER sent frame by frame:
  1. LOCAL (no LLM): scene shots + story events + ASR → candidate windows; 3 frames per candidate;
     cheap features (motion between frames, dHash for near-duplicate removal, sharpness, speech ratio).
  2. VISION (LLM, ONE shared client: scripts/shared/llm_provider.chat_json, Gemini is vision-capable):
     a bounded set of candidates, 3 small JPEG frames each, in batches → who / action / shot type /
     on-screen text / mouth state. Answers are validated; every call goes through budget.LLM
     (forecast + hard stop + response cache).

"Is he speaking?" is never decided from one still: the model sees several frames of the window and the result
is FUSED with ASR speech coverage of that window (fuse_speaking) — insufficient evidence → "uncertain".
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .budget import LLMStop, SchemaError

VISION_VERSION = "trailer-vision-v1"
FRAME_W = 640
FRAMES_PER_CAND = 3
SHOT_TYPES = ("closeup", "medium", "wide", "insert", "graphic")
KINDS = ("main_interview", "reaction", "cutaway", "insert_clip", "advertisement", "title_graphic")
MOUTH = ("open_changing", "closed", "unclear")
SPEAKING = ("likely_speaking", "likely_listening", "uncertain")


# ── stage 1: candidates (pure, tested) ──────────────────────────────────────

def speech_ratio(words: Sequence[Any], a: float, b: float) -> float:
    """Share of [a, b] covered by ASR words (gaps < 0.35 s between words count as speech)."""
    if b <= a:
        return 0.0
    iv = sorted((max(a, w.start), min(b, w.end + 0.35)) for w in words if w.end > a and w.start < b)
    covered, cur = 0.0, a
    for s, e in iv:
        s = max(s, cur)
        if e > s:
            covered += e - s
            cur = e
    return round(min(1.0, covered / (b - a)), 3)


def event_weight(events: Sequence[Dict[str, Any]], a: float, b: float) -> int:
    """Importance of the best story event overlapping [a, b] (0 = none)."""
    return max((int(e.get("importance", 1)) for e in events if e["end"] > a and e["start"] < b), default=0)


def candidate_windows(shot_list: Sequence[Dict[str, Any]], duration: float, *, win: float = 4.5, min_len: float = 1.8,
                      per_shot: int = 3, edge: float = 0.2) -> List[Dict[str, Any]]:
    """Windows INSIDE one shot each (never across a cut). Long shots give up to per_shot spread windows."""
    out = []
    for sh in shot_list:
        a0, b0 = sh["start"] + edge, min(sh["end"], duration) - edge
        L = b0 - a0
        if L < min_len:
            continue
        n = 1 if L <= win * 1.3 else min(per_shot, int(L // win))
        w = min(win, L) if n == 1 else win
        for k in range(n):
            s = a0 if n == 1 else a0 + (L - w) * k / (n - 1)
            out.append({"shot_id": sh["id"], "start": round(s, 2), "end": round(min(b0, s + w), 2),
                        "shot_start": sh["start"], "shot_end": sh["end"]})
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def select_candidates(cands: List[Dict[str, Any]], events: Sequence[Dict[str, Any]], words: Sequence[Any],
                      max_n: int = 56, min_hash_dist: int = 6) -> List[Dict[str, Any]]:
    """Rank by story relevance (event importance, silence is a bonus: nobody talks over the narration), keep
    visually distinct ones (dHash), at most max_n. `dhash` must already be on each candidate."""
    for c in cands:
        c["event_weight"] = event_weight(events, c["start"], c["end"])
        c["speech_ratio"] = speech_ratio(words, c["start"], c["end"])
        c["rank"] = c["event_weight"] * 2 + 1.0 + (1.0 - c["speech_ratio"]) * 1.5
    kept: List[Dict[str, Any]] = []
    for c in sorted(cands, key=lambda x: (-x["rank"], x["start"])):
        if len(kept) >= max_n:
            break
        if any(hamming(c["dhash"], k["dhash"]) < min_hash_dist and abs(c["start"] - k["start"]) < 120 for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda x: x["start"])
    for i, c in enumerate(kept):
        c["id"] = f"V{i + 1}"
    return kept


def fuse_speaking(mouth_states: Sequence[str], model_verdict: str, ratio: float) -> Dict[str, Any]:
    """Mouth evidence (several frames) + ASR coverage → speaking state; honest about uncertainty.
    A single still never decides. Silence in the window (ASR) means nobody speaks (certain)."""
    if ratio < 0.12:
        return {"state": "not_speaking", "basis": "asr_silence", "confidence": "high"}
    seen = [m for m in mouth_states if m in MOUTH and m != "unclear"]
    if len(seen) < 2:
        return {"state": "uncertain", "basis": "mouth_unclear_or_single_frame", "confidence": "low"}
    opened = sum(1 for m in seen if m == "open_changing")
    if opened >= 2 and model_verdict == "likely_speaking" and ratio >= 0.4:
        return {"state": "likely_speaking", "basis": "mouth_motion_multi_frame+asr", "confidence": "medium"}
    if opened == 0 and model_verdict == "likely_listening" and ratio >= 0.4:
        return {"state": "likely_listening", "basis": "mouth_closed_multi_frame+asr_speech", "confidence": "medium"}
    return {"state": "uncertain", "basis": "evidence_conflicts", "confidence": "low"}


# ── frames + local features (cv2/numpy, ffmpeg) ─────────────────────────────

def _ffmpeg() -> str:
    return os.environ.get("SONYA_FFMPEG") or "ffmpeg"


def frame_times(c: Dict[str, Any]) -> List[float]:
    a, b = c["start"], c["end"]
    return [round(a + (b - a) * f, 2) for f in (0.12, 0.5, 0.88)]


def extract_frame(src: str, t: float, out: str, width: int = FRAME_W) -> str:
    """One JPEG. A 3-hour source can hold a damaged frame: try fast input seek, then a decode-accurate seek,
    then a nearby time; only then fail (the caller drops that window instead of the whole run)."""
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return out
    err = ""
    for pre, post, dt in (([ "-ss", f"{t:.2f}"], [], 0.0), ([], ["-ss", f"{t:.2f}"], 0.0),
                          (["-ss", f"{t + 0.5:.2f}"], [], 0.5)):
        r = subprocess.run([_ffmpeg(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1", *pre, "-i", src,
                            *post, "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", out], capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        err = r.stderr[-200:]
    raise RuntimeError(f"frame extract failed at {t}: {err}")


CLIP_W = 360
CLIP_FPS = 8


def extract_clip(src: str, a: float, b: float, out: str, width: int = CLIP_W, fps: int = CLIP_FPS) -> str:
    """Short silent low-res fragment of one window (≈ 100–300 KB) for Gemini video input."""
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return out
    r = subprocess.run([_ffmpeg(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{a:.2f}", "-t",
                        f"{max(0.5, b - a):.2f}", "-i", src, "-an", "-vf", f"scale={width}:-2,fps={fps}", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"clip extract failed at {a}: {r.stderr[-200:]}")
    return out


def video_part(path: str) -> Dict[str, Any]:
    """Gemini (via Timeweb) takes a video as image_url with a data:video/mp4 URL (verified live 2026-11-04)."""
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{b}"}}


_FACE = None
_FACE_LOCK = threading.Lock()   # one CascadeClassifier shared by the frame workers is NOT thread-safe: counts varied
                                # between runs (1 vs 3 faces on the same frames) → different director prompts, lost cache


def count_faces(grays) -> int:
    """Most frontal faces seen in any frame (OpenCV Haar, local, free). 0 → nobody to call "the hero in frame": a model
    can describe a face on a plain elevator door (observed). Profile faces are missed, so this only VETOES, never proves."""
    global _FACE
    import cv2
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best = 0
    with _FACE_LOCK:
        for g in grays:
            found = _FACE.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, g.shape[1] // 22),) * 2)
            best = max(best, len(found))
    return best


def local_features(paths: Sequence[str]) -> Dict[str, Any]:
    import cv2
    import numpy as np
    grays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    small = [cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA) for g in grays]
    mid = small[len(small) // 2]
    bits = (mid[:, 1:] > mid[:, :-1]).flatten()
    dh = int("".join("1" if b else "0" for b in bits), 2)
    thumbs = [cv2.resize(g, (64, 28), interpolation=cv2.INTER_AREA).astype(np.float32) for g in grays]
    motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(thumbs, thumbs[1:])])) if len(thumbs) > 1 else 0.0
    return {"dhash": dh, "faces": count_faces(grays), "motion": round(motion, 2), "sharpness": round(float(cv2.Laplacian(grays[len(grays) // 2], cv2.CV_64F).var()), 1),
            "brightness": round(float(grays[len(grays) // 2].mean()), 1)}


def prepare_clips(src: str, cands: List[Dict[str, Any]], frames_dir: Path, sha: str, workers: int = 4) -> None:
    """vision="clips": one small silent fragment per chosen window (only for the windows that go to the model)."""
    def one(c):
        c["clip"] = extract_clip(src, c["start"], c["end"], str(frames_dir / f"{sha[:10]}_{c['start']:.2f}_{c['end']:.2f}.mp4"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, cands))


def prepare_candidates(src: str, cands: List[Dict[str, Any]], frames_dir: Path, sha: str, workers: int = 4) -> List[Dict[str, Any]]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    def one(c):
        try:
            ps = [extract_frame(src, t, str(frames_dir / f"{sha[:10]}_{t:.2f}.jpg")) for t in frame_times(c)]
            c["frames"] = ps
            c.update(local_features(ps))
        except (RuntimeError, OSError, ValueError):       # unreadable window: skip it, the others are enough
            return None
        return c
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [c for c in pool.map(one, cands) if c is not None]


# ── stage 2: vision LLM ─────────────────────────────────────────────────────

VISION_SYSTEM = (
    "Ты — визуальный аналитик монтажёра. Ты ВИДИШЬ кадры (по три кадра из каждого окна, по порядку времени) "
    "из длинного подкаст-интервью. Описывай только то, что реально видно. Не выдумывай имена: если не уверен, "
    "пиши «неизвестный». По неподвижному кадру нельзя утверждать, что человек говорит: сравни рот на ТРЁХ кадрах "
    "окна (open_changing — рот заметно меняется/открыт на нескольких кадрах; closed — закрыт на всех; unclear — "
    "лица не видно или не ясно). Рекламные вставки, плашки, экраны приложений и титры отмечай отдельно. "
    "Отвечай только JSON.")


def image_part(path: str) -> Dict[str, Any]:
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}


def vision_messages(batch: Sequence[Dict[str, Any]], characters: Sequence[Dict[str, Any]],
                    speech_text: Dict[str, str], vision: str = "frames") -> List[Dict[str, Any]]:
    chars = [f"{c['name']}: {c.get('description', '')[:140]}" for c in characters][:8]
    parts: List[Dict[str, Any]] = [{"type": "text", "text": (
        "ГЕРОИ ИСТОРИИ (из расшифровки; внешность неизвестна — опирайся на контекст, а не на догадки):\n"
        + "\n".join(chars) + "\n\nНиже окна V<n>: три кадра каждого + что звучит в этот момент (ASR). Для КАЖДОГО окна верни "
        "объект. Схема:\n{\"windows\": [{\"id\": \"V1\", \"shot_type\": \"" + "|".join(SHOT_TYPES) + "\", "
        "\"people\": [{\"who\": \"имя героя из списка / ведущий / неизвестный\", \"confidence\": 0..1, "
        "\"mouth\": \"" + "|".join(MOUTH) + "\"}], \"action\": \"что делают, 1 короткая фраза\", \"setting\": \"где\", "
        "\"composition\": \"кратко: план, кто слева/справа\", \"kind\": \"" + "|".join(KINDS) + "\", "
        "\"speaking\": \"" + "|".join(SPEAKING) + "\", \"on_screen_text\": {\"present\": bool, \"text\": \"…\", "
        "\"cut_by_vertical_center_crop\": bool}, \"visual_interest\": 1..5, \"good_for\": [\"темы/ситуации, "
        "которые этот кадр может иллюстрировать (3–5 слов каждая)\"], \"problems\": [\"…\"]}]}\n")}]
    for c in batch:
        parts.append({"type": "text", "text": f"\n=== {c['id']} ({c['start']:.1f}–{c['end']:.1f} с). ASR в окне: "
                                              f"{speech_text.get(c['id']) or '(тишина)'}"})
        if vision == "clips" and c.get("clip"):
            parts.append({"type": "text", "text": f"{c['id']} видеофрагмент без звука ({c['end'] - c['start']:.1f} с):"})
            parts.append(video_part(c["clip"]))
            continue
        for k, p in enumerate(c["frames"]):
            parts.append({"type": "text", "text": f"{c['id']} кадр {k + 1}/{len(c['frames'])}:"})
            parts.append(image_part(p))
    system = VISION_SYSTEM if vision != "clips" else VISION_SYSTEM.replace(
        "по три кадра из каждого окна, по порядку времени", "короткий видеофрагмент без звука на каждое окно").replace(
        "сравни рот на ТРЁХ кадрах окна", "следи за ртом на протяжении фрагмента")
    return [{"role": "system", "content": system}, {"role": "user", "content": parts}]


def validate_vision(data: Any, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("windows"), list):
        raise SchemaError("vision: {'windows': [...]} expected")
    out: Dict[str, Dict[str, Any]] = {}
    for w in data["windows"]:
        if not isinstance(w, dict) or w.get("id") not in ids:
            continue
        st = w.get("shot_type") if w.get("shot_type") in SHOT_TYPES else "medium"
        kind = w.get("kind") if w.get("kind") in KINDS else "main_interview"
        people = []
        for p in (w.get("people") or [])[:4]:
            if isinstance(p, dict):
                try:
                    conf = max(0.0, min(1.0, float(p.get("confidence", 0))))
                except (TypeError, ValueError):
                    conf = 0.0
                people.append({"who": str(p.get("who") or "неизвестный")[:60], "confidence": conf,
                               "mouth": p.get("mouth") if p.get("mouth") in MOUTH else "unclear"})
        ost = w.get("on_screen_text") if isinstance(w.get("on_screen_text"), dict) else {}
        try:
            vi = max(1, min(5, int(round(float(w.get("visual_interest", 3))))))
        except (TypeError, ValueError):
            vi = 3
        out[w["id"]] = {"shot_type": st, "kind": kind, "people": people,
                        "action": str(w.get("action") or "")[:200], "setting": str(w.get("setting") or "")[:120],
                        "composition": str(w.get("composition") or "")[:160],
                        "model_speaking": w.get("speaking") if w.get("speaking") in SPEAKING else "uncertain",
                        "on_screen_text": {"present": bool(ost.get("present")), "text": str(ost.get("text") or "")[:80],
                                           "cut_by_vertical_center_crop": bool(ost.get("cut_by_vertical_center_crop"))},
                        "visual_interest": vi,
                        "good_for": [str(x)[:60] for x in (w.get("good_for") or [])][:5],
                        "problems": [str(x)[:100] for x in (w.get("problems") or [])][:4]}
    missing = [i for i in ids if i not in out]
    if missing:
        raise SchemaError(f"vision: no analysis for {missing}")
    return out


CROP_TEXT_MAX_CHARS = 4


def text_would_be_cut(ost: Dict[str, Any]) -> bool:
    """The model's own "cut_by_vertical_center_crop" is not reliable (observed: the caption «October 19 1987» across a
    16:9 frame was answered "false" and came out as «ctober 19 198»). A 9:16 crop keeps ~32 % of a 16:9 width and
    follows faces, not text, so any caption longer than a few characters is treated as cut — the window is not used.
    Set dressing the model reports WITHOUT text (present=false) stays allowed."""
    if not ost.get("present"):
        return False
    if ost.get("cut_by_vertical_center_crop"):
        return True
    return len(re.sub(r"\s+", "", str(ost.get("text") or ""))) > CROP_TEXT_MAX_CHARS


def finalize(c: Dict[str, Any], v: Dict[str, Any]) -> Dict[str, Any]:
    """Merge model output + local features into the public analysis record (no raw paths of frames needed)."""
    mouths = [p["mouth"] for p in v["people"]]
    # several people in a window: mouth evidence per person is not frame-aligned, so use the strongest signal
    states = mouths if len(mouths) >= 2 else mouths * FRAMES_PER_CAND if mouths else []
    sp = fuse_speaking(states, v["model_speaking"], c["speech_ratio"])
    # set dressing (a neon sign in the room) is fine; text that the vertical centre crop would cut, ads, graphics and
    # app screens are not
    usable = (v["kind"] not in ("advertisement", "title_graphic") and v["shot_type"] != "graphic"
              and not text_would_be_cut(v["on_screen_text"]))
    return {"id": c["id"], "start": c["start"], "end": c["end"], "shot_id": c["shot_id"],
            "shot_start": c["shot_start"], "shot_end": c["shot_end"], "event_weight": c["event_weight"],
            "speech_ratio": c["speech_ratio"], "faces": c.get("faces"), "motion": c["motion"], "sharpness": c["sharpness"], "dhash": c["dhash"],
            "frame_times": frame_times(c), **v, "speaking": sp, "usable": usable}


def prefilter(raw: List[Dict[str, Any]], events: Sequence[Dict[str, Any]], duration: float, n: int,
              bucket_sec: float = 600.0) -> List[Dict[str, Any]]:
    """Cheap pre-filter before frame extraction: the n most story-relevant windows, SPREAD over the whole source.
    A global sort by (event weight, start) filled the quota from the first minutes of a long film; here every
    bucket_sec bucket gets its own share (≥ 1), ties inside a bucket are spread evenly in time."""
    for c in raw:
        c["_w"] = event_weight(events, c["start"], c["end"])
    if len(raw) <= n:
        return sorted(raw, key=lambda c: c["start"])
    nb = max(1, int(duration // bucket_sec) + (1 if duration % bucket_sec > bucket_sec / 3 else 0))
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(nb)]
    for c in raw:
        buckets[min(nb - 1, int(c["start"] // (duration / nb)))].append(c)
    quota = max(1, n // nb)
    chosen: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for b in buckets:
        b.sort(key=lambda c: -c["_w"])
        top = b[:quota * 3]                                   # best weights, then thin them out evenly in time
        top.sort(key=lambda c: c["start"])
        step = len(top) / quota if len(top) > quota else 1.0
        pick = [top[int(i * step)] for i in range(min(quota, len(top)))]
        chosen += pick
        ids = {id(x) for x in pick}
        rest += [c for c in b if id(c) not in ids]
    if len(chosen) < n:                                       # unused quota → best remaining overall
        rest.sort(key=lambda c: (-c["_w"], c["start"]))
        chosen += rest[:n - len(chosen)]
    return sorted(chosen[:n], key=lambda c: c["start"])


def analyze(llm, src: str, shot_list, words, events, characters, duration: float, sha: str, frames_dir: Path,
            *, max_candidates: int = 56, batch: int = 6, speech_by_window=None, log=print,
            vision: str = "frames") -> Dict[str, Any]:
    """Stage 1 + stage 2. `llm` is budget.LLM (its chat is llm_provider.chat_json)."""
    raw = candidate_windows(shot_list, duration)
    # cheap pre-filter before frame extraction: story-relevant first, evenly spread, cap 3× final size
    pre = prefilter(raw, events, duration, max_candidates * 3)
    prepared = prepare_candidates(src, pre, frames_dir, sha)
    chosen = select_candidates(prepared, events, words, max_n=max_candidates)
    if vision == "clips":
        prepare_clips(src, chosen, frames_dir, sha)
    out: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    speech_text = speech_by_window or (lambda c: "")

    def ask(tag: str, b: List[Dict[str, Any]]):
        ids = [c["id"] for c in b]
        st = {c["id"]: speech_text(c) for c in b}
        return llm.ask(tag, vision_messages(b, characters, st, vision), VISION_VERSION,
                       lambda d, ids=ids: validate_vision(d, ids), temperature=0.0, max_out=4000)
    for i in range(0, len(chosen), batch):
        b = chosen[i:i + batch]
        try:
            res = ask(f"vision_{i // batch + 1}", b)
        except LLMStop:
            raise
        except RuntimeError as e:                    # a batch the provider will not answer: window by window
            log(f"vision batch {i // batch + 1} failed ({str(e)[:80]}) → one window at a time")
            res = {}
            for c in b:
                try:
                    res.update(ask(f"vision_{i // batch + 1}_{c['id']}", [c]))
                except LLMStop:
                    raise
                except RuntimeError as e2:
                    failed.append({"id": c["id"], "start": c["start"], "error": str(e2)[:160]})
        out += [finalize(c, res[c["id"]]) for c in b if c["id"] in res]
        log(f"vision batch {i // batch + 1}/{(len(chosen) + batch - 1) // batch}")
    return {"version": VISION_VERSION, "stage1": {"raw_windows": len(raw), "prepared": len(prepared),
                                                  "sent_to_vision": len(chosen), "frames_per_window": FRAMES_PER_CAND, "vision_input": vision,
                                                  "failed_windows": failed},
            "candidates": out}


def supplement(llm, src: str, shot_list, words, focus_events: Sequence[Dict[str, Any]], characters, duration: float,
               sha: str, frames_dir: Path, have: Sequence[Dict[str, Any]], *, per_event: int = 2, max_new: int = 30,
               batch: int = 6, speech_by_window=None, log=print, vision: str = "frames", dry_run: bool = False) -> Dict[str, Any]:
    """Targeted second pass (punchy style): the first pass picked windows for the documentary map, so the wildest
    moments the new script lists (a salvo) may have no picture at all. Here: up to per_event NEW windows inside each
    focus event (not overlapping an analysed one), visually distinct, ≤ max_new in total → the same vision prompt.
    Ids are X1…; cost is bounded by max_new / batch calls (each one budget-guarded)."""
    taken = [(c["start"], c["end"]) for c in have]
    raw = [c for c in candidate_windows(shot_list, duration)
           if not any(c["start"] < b and a < c["end"] for a, b in taken)]
    pre: List[Dict[str, Any]] = []
    for e in focus_events:
        inside = [c for c in raw if c["start"] >= e["start"] - 1 and c["end"] <= e["end"] + 1 and c not in pre]
        step = max(1, len(inside) // (per_event * 3))
        pre += inside[::step][:per_event * 3]
    if not pre:
        return {"uncached_batches": 0, "batches": 0} if dry_run else {"candidates": [], "stage1": {"supplement_windows": 0}}
    prepared = prepare_candidates(src, pre, frames_dir, sha)
    chosen: List[Dict[str, Any]] = []
    for e in focus_events:
        mine = [c for c in prepared if c["start"] >= e["start"] - 1 and c["end"] <= e["end"] + 1 and c not in chosen]
        chosen += select_candidates(mine, [e], words, max_n=per_event)
    chosen = sorted(chosen[:max_new], key=lambda c: c["start"])
    for i, c in enumerate(chosen):
        c["id"] = f"X{i + 1}"
    if vision == "clips":
        prepare_clips(src, chosen, frames_dir, sha)
    speech_text = speech_by_window or (lambda c: "")
    out: List[Dict[str, Any]] = []
    uncached = 0
    for i in range(0, len(chosen), batch):
        b = chosen[i:i + batch]
        ids = [c["id"] for c in b]
        st = {c["id"]: speech_text(c) for c in b}
        if dry_run:                  # forecast only: which batches would really be sent (not cached, not marked failed)
            mk = frames_dir / ("vision_extra_failed-" + "_".join(f"{c['start']:.2f}" for c in b) + ".txt")
            cp = llm._cache_path(f"vision_extra_{i // batch + 1}", vision_messages(b, characters, st, vision), VISION_VERSION, 0.0, 4000)
            uncached += int(not mk.is_file() and not (cp is not None and cp.is_file()))
            continue
        # a batch the model answered with a broken schema twice is not paid for again on the next run
        mark = frames_dir / ("vision_extra_failed-" + "_".join(f"{c['start']:.2f}" for c in b) + ".txt")
        if mark.is_file():
            log(f"vision extra batch {i // batch + 1} skipped (failed before: {mark.name[:40]}…)")
            continue
        try:
            res = llm.ask(f"vision_extra_{i // batch + 1}", vision_messages(b, characters, st, vision), VISION_VERSION,
                          lambda d, ids=ids: validate_vision(d, ids), temperature=0.0, max_out=4000)
        except LLMStop:
            raise
        except RuntimeError as e:
            log(f"vision extra batch {i // batch + 1} failed ({str(e)[:80]}) — skipped")
            mark.write_text(str(e)[:300], encoding="utf-8")
            continue
        out += [finalize(c, res[c["id"]]) for c in b if c["id"] in res]
        log(f"vision extra batch {i // batch + 1}/{(len(chosen) + batch - 1) // batch}")
    if dry_run:
        return {"uncached_batches": uncached, "batches": (len(chosen) + batch - 1) // batch}
    return {"candidates": out, "stage1": {"supplement_windows": len(chosen)}}
