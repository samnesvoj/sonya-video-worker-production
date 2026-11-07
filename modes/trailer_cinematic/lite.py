"""
lite.py — trailer_cinematic ECONOMY path (owner 2026-11-05: a trailer of one fragment must cost 2.5–4.5 ₽, hard stop 5 ₽).

    one fragment of a long video  →  local ASR cache → transcript squeezed locally (no fillers / repeats, key lines
    with timecodes) → 12–20 small frames (≤ 384 px) chosen locally (cuts, faces, motion, strong words)
    → ONE Story+Writer call (digest + frames → story points + script + frame plan)
    → free deterministic checks → [at most ONE text-only fix, only if the checks found a defect]
    → TTS → local shot plan (no Director, no Judge) → render → BEEP → technical validator.

Gemini never sees a video clip: only still frames. Calls go through budget.LLM (forecast BEFORE the call, hard stop at
LITE_STOP_RUB in code, the daily Timeweb ledger is untouched in size). The default for the energetic style when the
source is a fragment (`--window`), see recap_runner.main.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import budget, censor, compose, media, recap, subs, timeline as tl, validator, visual
from .budget import SchemaError
from .runner import ROOT, _desktop_media_env, _dump, _load_llm_env, _sha256, run_asr

LITE_VERSION = "trailer-lite-v1"
LITE_STOP_RUB = 5.0                 # hard stop in code: forecast before the call + the ledger of this video
LITE_FRAMES = (12, 20)              # how many stills Gemini gets
FRAME_MAX_PX = 384                  # longest side of a still sent to the model
DIGEST_CHARS = 9500                 # target size of the squeezed transcript
EVENT_SEC = 28.0                    # one digest event ≈ this many seconds of the fragment
MAX_OUT_TOKENS = 3500
FIX_OUT_TOKENS = 2600               # the text-only fix: the script alone is ≈ 1.5–2k tokens

FILLERS = {"ну", "вот", "э", "ээ", "эм", "м", "мм", "ммм", "типа", "короче", "значит", "как", "бы", "блин", "это", "самое",
           "так", "же", "ладно", "аа", "а-а", "угу", "ага", "да-да", "ой", "ах", "эх", "ага.", "ну,", "вот,"}
FILLER_PHRASES = re.compile(r"\b(как бы|это самое|в общем|так сказать|на самом деле)\b", re.I)
STRONG = re.compile(r"(!|\?{2}|ха-?ха|смех|кринж|пиздец|блять|бляд|нахуй|хуй|сука|еб[а-я]+|дроч|жоп|бород|нюха|убью|орёт|орет)", re.I)


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


# ── 1. squeeze the transcript (local, free) ─────────────────────────────────

def clean_text(text: str) -> str:
    """Fillers and stutters out, words kept in order."""
    text = FILLER_PHRASES.sub(" ", text)
    out: List[str] = []
    for tok in text.split():
        bare = re.sub(r"[^\w-]", "", tok).lower()
        if bare in FILLERS:
            continue
        if out and re.sub(r"[^\w-]", "", out[-1]).lower() == bare and bare:
            continue                                              # «да да да» / stutter
        out.append(tok)
    return re.sub(r"\s+", " ", " ".join(out)).strip(" ,")


def sentence_score(text: str) -> float:
    n = len(text.split())
    return min(n, 14) * 0.4 + 3.0 * len(STRONG.findall(text)) + (1.0 if n >= 4 else -2.0)


def digest(raw_words: Sequence[Dict[str, Any]], a: float, b: float, *, target_chars: int = DIGEST_CHARS,
           event_sec: float = EVENT_SEC) -> List[Dict[str, Any]]:
    """Events E1..En over [a, b]: each holds the strongest cleaned lines (with m:ss relative to a). Total ≈ target_chars."""
    inside = [w for w in raw_words if w.get("end", 0) > a and w.get("start", 0) < b]
    _, sents = tl.sentences(inside)
    n_ev = max(1, int(math.ceil((b - a) / event_sec)))
    per_event = max(80, target_chars // n_ev - 14)
    events: List[Dict[str, Any]] = []
    for k in range(n_ev):
        lo, hi = a + k * (b - a) / n_ev, a + (k + 1) * (b - a) / n_ev
        rows = []
        for s in sents:
            if lo <= s["start"] < hi:
                t = clean_text(s["text"])
                if len(t.split()) >= 2 or "!" in t:
                    rows.append({"t": s["start"], "text": t[:150], "score": sentence_score(t)})
        keep, used = [], 0
        for r in sorted(rows, key=lambda r: -r["score"]):
            cost = len(r["text"]) + 8
            if used + cost > per_event:
                continue
            keep.append(r)
            used += cost
        keep.sort(key=lambda r: r["t"])
        events.append({"id": f"E{k + 1}", "start": round(lo, 2), "end": round(hi, 2),
                       "text": " | ".join(f"[{mmss(r['t'] - a)}] {r['text']}" for r in keep)})
    return events


def digest_text(events: Sequence[Dict[str, Any]], a: float) -> str:
    return "\n".join(f"{e['id']} ({mmss(e['start'] - a)}–{mmss(e['end'] - a)}): {e['text'] or '(речи почти нет)'}" for e in events)


# ── 2. frames chosen locally ────────────────────────────────────────────────

def strong_times(raw_words: Sequence[Dict[str, Any]], a: float, b: float) -> List[float]:
    out = []
    for w in raw_words:
        if a <= w.get("start", 0) < b and (STRONG.search(w.get("word", "")) or "!" in w.get("word", "")):
            out.append(float(w["start"]))
    return out


def select_frames(times: Sequence[float], diff: Sequence[float], bright: Sequence[float], faces: Dict[int, int],
                  strong: Sequence[float], n: int, a: float, b: float) -> List[int]:
    """Pure: indexes (into `times`) of n frames, one per equal time bin (so the plan covers the fragment in order), each
    the best of its bin by cut/motion + faces + strong words nearby. Near-black frames are skipped."""
    n = max(1, n)
    picks: List[int] = []
    span = (b - a) / n
    dmax = max(diff) if diff and max(diff) > 0 else 1.0
    for k in range(n):
        lo, hi = a + k * span, a + (k + 1) * span
        best, best_s = None, -1e9
        for i, t in enumerate(times):
            if not lo <= t < hi or bright[i] < 28:
                continue
            s = diff[i] / dmax + (1.0 if faces.get(i, 0) >= 1 else 0.0) + (0.3 if faces.get(i, 0) >= 2 else 0.0)
            s += 0.8 if any(abs(t - x) < 1.5 for x in strong) else 0.0
            if s > best_s:
                best, best_s = i, s
        if best is not None:
            picks.append(best)
    return picks


def scan_fragment(src: str, a: float, b: float, fps: float = 2.0, w: int = 128, h: int = 72):
    """Whole fragment as tiny gray frames at `fps` (one ffmpeg pass, ~free) → (times, diff-to-previous, brightness)."""
    import numpy as np
    cmd = [media.ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}",
           "-i", src, "-an", "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8)
    n = len(fr) // (w * h)
    fr = fr[: n * w * h].reshape(n, w * h).astype(np.float32)
    diff = [0.0] + [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, n)]
    return [a + i / fps for i in range(n)], diff, [float(f.mean()) for f in fr]


def extract_still(src: str, t: float, out: str, px: int = FRAME_MAX_PX) -> str:
    if os.path.isfile(out) and os.path.getsize(out) > 500:
        return out
    def grab(tol: bool, tt: float):
        pre = ["-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts"] if tol else []       # damaged stretch of a long film
        return subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", *pre, "-ss", f"{tt:.2f}", "-i", src,
                               "-frames:v", "1", "-vf", f"scale={px}:{px}:force_original_aspect_ratio=decrease", "-q:v", "5", out],
                              capture_output=True, text=True)
    r = grab(False, t)
    if r.returncode != 0 or not os.path.isfile(out):
        for tt in (t, t + 0.6, t - 0.6):
            r = grab(True, tt)
            if r.returncode == 0 and os.path.isfile(out):
                break
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"still at {t:.1f}s failed: {r.stderr[-200:]}")
    return out


def choose_frames(src: str, raw_words, a: float, b: float, n: int, work: Path) -> List[Dict[str, Any]]:
    times, diff, bright = scan_fragment(src, a, b)
    strong = strong_times(raw_words, a, b)
    # faces only on a short list (top frames of each bin by motion), local Haar detector from visual.py
    import cv2
    shortlist = select_frames(times, diff, bright, {}, strong, n * 3, a, b)
    faces: Dict[int, int] = {}
    for i in shortlist:
        p = extract_still(src, times[i], str(work / f"cand_{int(times[i] * 10)}.jpg"))
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        faces[i] = visual.count_faces([g]) if g is not None else 0
    picks = select_frames(times, diff, bright, faces, strong, n, a, b)
    out = []
    for k, i in enumerate(picks):
        p = extract_still(src, times[i], str(work / f"F{k + 1}.jpg"))
        out.append({"id": f"F{k + 1}", "t": round(times[i], 2), "rel": mmss(times[i] - a), "path": p, "faces": faces.get(i, 0)})
    return out


# ── 3. ONE Story+Writer call ────────────────────────────────────────────────

LITE_SYSTEM_EXTRA = (
    "\n\nЭТО РЕЖИМ «ЭКОНОМ»: перед тобой не весь фильм, а ОДИН фрагмент (спектакль/сцена) — выжимка расшифровки и кадры. "
    "Ты одновременно разбираешь, что там происходит, и пишешь текст. Описать нужно ВСЁ, что происходит, по порядку: "
    "сюжет сцены → как участники реагируют на подсказки → как всё разваливается. Кадры подписаны F1…; расшифровка — "
    "событиями E1…. В кадре ищи, кто что делает; чего нет ни в кадрах, ни в расшифровке — не пиши.")


def lite_messages(events: Sequence[Dict[str, Any]], frames: Sequence[Dict[str, Any]], a: float, b: float, lo_c: int, hi_c: int,
                  rate: Dict[str, Any], target_sec: float, forbid: Sequence[str] = ()) -> List[Dict[str, Any]]:
    lo_n, hi_n = recap.STYLES["punchy"]["segments"]
    task = (
        f"ЗАДАЧА: трейлер ≈ {target_sec:.0f} секунд БЫСТРОЙ речи по фрагменту длиной {mmss(b - a)}. Голос измерен: "
        f"≈ {rate['chars_per_sec']:.1f} знаков в секунду. ВЕСЬ text (сумма по сегментам, с пробелами) — от {lo_c} до "
        f"{hi_c} знаков (≈ {lo_c // 7}–{hi_c // 7} слов). Считай знаки. МЕНЬШЕ {lo_c} знаков — брак (ролик выйдет короче "
        f"минуты): пиши много, на сегмент в среднем {lo_c // 11}–{hi_c // 10} знаков, залп — до 320.\n\n"
        "ВЫЖИМКА РАСШИФРОВКИ (единственный источник фактов вместе с кадрами; время от начала фрагмента, мат/ошибки "
        "распознавания возможны — не цитируй слова, пересказывай суть):\n" + digest_text(events, a) +
        "\n\nКАДРЫ: " + ", ".join(f"{f['id']}={f['rel']}" for f in frames) + " (картинки ниже, по порядку).\n\n"
        f"Сценарий — {lo_n + 2}–{hi_n - 1} сегментов (не меньше {lo_n + 2}!): hook (удар) → основная часть ПО ПОРЯДКУ событий фрагмента (event_ids "
        "идут по возрастанию времени) → ровно один salvo (залп: 6–9 самых угарных вещей через запятую, они могут быть "
        "из любого места) → turn/climax → punchline последним. Каждый сегмент 2–5 коротких предложений, в среднем ≤ 7 слов. ИМЁН нет совсем: ни полных, ни уменьшительных, ни "
        "кличек (Саня, Ваня, Лёха…) — только роли («этот парень», «ведущий», «царь», «леший», «Баба-Яга», «богатырь»). "
        "Для каждого сегмента: event_ids (только E…), frame_ids (1–3 кадра F…, лучше всего подходящих к ЭТОМУ тексту; "
        "для salvo — 3 кадра), visual_intent (5–12 слов), pause_after (short|normal|long), gags (выдуманные приёмы: "
        "[{\"text\": \"точная цитата из text\", \"device\": \"" + "|".join(recap.GAG_DEVICES) + "\"}] или []).\n"
        "Также верни characters — реальные ИМЕНА людей из фрагмента (ведущие, участники; для запрета в тексте) и "
        "points — 5–8 коротких фактов-опор вида [{\"event_id\": \"E…\", \"fact\": \"…\"}].\n"
        "Верни JSON: {\"title\": \"…\", \"characters\": [{\"role\": \"…\", \"name\": \"…\"}], \"points\": [...], "
        "\"segments\": [{\"id\": \"S1\", \"function\": \"" + "|".join(recap.PUNCHY_FUNCTIONS) + "\", \"text\": \"…\", "
        "\"event_ids\": [\"E…\"], \"frame_ids\": [\"F…\"], \"visual_intent\": \"…\", \"pause_after\": \"short|normal|long\", "
        "\"gags\": []}]}.")
    if forbid:
        task += "\n\nЗАПРЕЩЁННЫЕ СЛОВА (имена): " + ", ".join(x.rstrip("$") for x in forbid) + "."
    parts: List[Dict[str, Any]] = [{"type": "text", "text": task}]
    for f in frames:
        parts.append({"type": "text", "text": f"{f['id']} ({f['rel']}):"})
        parts.append(visual.image_part(f["path"]))
    return [{"role": "system", "content": recap.WRITER_SYSTEM_PUNCHY + LITE_SYSTEM_EXTRA}, {"role": "user", "content": parts}]


def fix_messages(events, a: float, script: Dict[str, Any], defects: Sequence[str], lo_c: int, hi_c: int, rate) -> List[Dict[str, Any]]:
    """Text-only repair (no frames): the previous script + the list of defects found by the free checks."""
    task = (f"Исправь сценарий трейлера. Общая длина text: {lo_c}–{hi_c} знаков (голос ≈ {rate['chars_per_sec']:.1f} зн/с).\n"
            "ДЕФЕКТЫ (исправь все, остальное сохрани):\n- " + "\n- ".join(defects) +
            "\n\nПРЕДЫДУЩИЙ ВАРИАНТ (сегменты можно добавлять и переписывать, JSON тот же):\n" + json.dumps(script, ensure_ascii=False) +
            "\n\nВЫЖИМКА (для сверки фактов):\n" + digest_text(events, a)[:6000] +
            "\n\nВерни тот же JSON целиком (segments с event_ids, frame_ids, visual_intent, pause_after, gags).")
    return [{"role": "system", "content": recap.WRITER_SYSTEM_PUNCHY + LITE_SYSTEM_EXTRA}, {"role": "user", "content": task}]


def free_checks(script: Dict[str, Any], frames: Sequence[Dict[str, Any]], events: Sequence[Dict[str, Any]],
                tolerance_sec: float = 90.0) -> List[str]:
    """Deterministic, free: frame ids exist, the main body goes in the order of the fragment, the frames cover it."""
    bad: List[str] = []
    fids = {f["id"] for f in frames}
    ev_t = {e["id"]: (e["start"] + e["end"]) / 2 for e in events}
    prev = None
    for s in script["segments"]:
        if not s.get("frame_ids"):
            bad.append(f"{s['id']}: no valid frame_ids (use F… from the list)")
        if s["function"] in ("hook", "salvo", "punchline"):
            continue
        t = min(ev_t[e] for e in s["event_ids"])
        if prev is not None and t < prev - tolerance_sec:
            bad.append(f"{s['id']}: events go back in time ({mmss(t)} after {mmss(prev)}) — the main part must follow the fragment in order")
        prev = max(prev or t, t)
    used = {x for s in script["segments"] for x in s.get("frame_ids", [])}
    if len(frames) >= 8 and len(used) < max(5, len(frames) // 2):
        bad.append(f"only {len(used)} different frames used of {len(frames)}: spread the frame_ids over the whole story")
    return bad


ROLE_CAPS = {"Леший", "Лешего", "Лешему", "Лешим", "Баба-Яга", "Бабу-Ягу", "Бабе-Яге", "Бабы-Яги", "Бабой-Ягой", "Яга", "Ягу", "Кощей",
             "Иван-дурак", "Ивана-дурака", "Ивану-дураку", "Василиса", "Василису", "Прекрасная", "Прекрасную", "Русь"}


NICKNAMES = re.compile(r"\b(Саня|Саня|Саш[аеуи]|Ван[яеюи]|Вас[яеюи]|Пет[яеюи]|Коль[яеюи]|Миш[аеуи]|Лёх[аеуи]|Лех[аеуи]|Дим[аеуы]|Серёг[аеуи]|"
                       r"Серег[аеуи]|Жен[яеюи]|Макс[аеуи]?|Паш[аеуи]|Костя|Андрюх[аеуи]|Толик|Ген[аеуы]|Стёп[аеуы]|Степ[аеуы])\b")


def stray_capitals(text: str) -> List[str]:
    """Capitalised words in the MIDDLE of a sentence that are not fairy-tale roles: probably a name or a nickname."""
    out = [m.group(0) for m in NICKNAMES.finditer(text)]
    for sent in recap.sentences_of(text):
        toks = sent.split()
        for tok in toks[1:]:
            w = re.sub(r"^[^\w]+|[^\w-]+$", "", tok)
            if len(w) > 2 and w[0].isupper() and not w.isupper() and w not in ROLE_CAPS:
                out.append(w)
    return out


def quick_defects(data: Any, lo_c: int, hi_c: int, n_range: Tuple[int, int]) -> List[str]:
    """All the cheap numbers at once (validate_script reports only the first problem): segments, length, stray names."""
    segs = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segs, list):
        return ["no segments[] in the answer"]
    bad: List[str] = []
    chars = sum(len(str(x.get("text", ""))) for x in segs if isinstance(x, dict))
    if not n_range[0] <= len(segs) <= n_range[1]:
        bad.append(f"сегментов {len(segs)}, нужно {n_range[0]}–{n_range[1]}")
    if not lo_c * 0.88 <= chars <= hi_c * 1.12:
        bad.append(f"всего {chars} знаков в text, нужно {lo_c}–{hi_c} ({'допиши' if chars < lo_c else 'сократи'} примерно на "
                   f"{abs(chars - (lo_c + hi_c) // 2)} знаков; на сегмент ≈ {(lo_c + hi_c) // 2 // max(1, n_range[0] + 2)} знаков)")
    names = sorted({w for x in segs if isinstance(x, dict) for w in stray_capitals(str(x.get("text", "")))})
    if names:
        bad.append(f"имена/клички в тексте: {', '.join(names[:6])} — замени на роль («этот парень», «ведущий», «царь»)")
    return bad


def validate_lite(data: Any, sm: Dict[str, Any], frames: Sequence[Dict[str, Any]], lo_c: int, hi_c: int,
                  slack: float = 0.12) -> Dict[str, Any]:
    """recap.validate_script (names, digits, topics, pace, length) + frame ids. Raises SchemaError on a defect."""
    clean = recap.validate_script(data, sm, lo_c, hi_c, slack, style="punchy")
    fids = {f["id"] for f in frames}
    raw = data["segments"]
    for i, s in enumerate(clean["segments"]):
        ids = [x for x in (raw[i].get("frame_ids") or []) if x in fids] if isinstance(raw[i], dict) else []
        s["frame_ids"] = ids[:4]
    clean["characters"] = [c for c in (data.get("characters") or []) if isinstance(c, dict)][:12]
    clean["points"] = [p for p in (data.get("points") or []) if isinstance(p, dict)][:10]
    return clean


# ── 4. forecast / stop ──────────────────────────────────────────────────────

def forecast_rub(messages: List[Dict[str, Any]], model: str, max_out: int = MAX_OUT_TOKENS) -> Dict[str, Any]:
    """Worst case of one call (whole output cap, 1100 tokens per image); in roubles at the pessimistic rate."""
    usd = budget.forecast_usd(messages, max_out, model)
    return {"usd": round(usd, 5), "rub": budget.rub(usd)}


def gate(messages: List[Dict[str, Any]], model: str, spent_rub: float, stop_rub: float = LITE_STOP_RUB,
         max_out: int = MAX_OUT_TOKENS) -> Dict[str, Any]:
    """Raises BudgetExceeded BEFORE the request if spent + forecast would pass the stop."""
    fc = forecast_rub(messages, model, max_out)
    fc["spent_rub"], fc["stop_rub"] = round(spent_rub, 2), stop_rub
    if spent_rub + fc["rub"] > stop_rub:
        raise budget.BudgetExceeded(f"STOP: spent {spent_rub:.2f} ₽ + forecast {fc['rub']:.2f} ₽ > {stop_rub:.1f} ₽")
    return fc


# ── 5. local shot plan (no Director) ────────────────────────────────────────

def shot_count(span: float, lo: float = 1.4, target: float = 2.6) -> int:
    return max(1, min(int(span // lo) or 1, int(round(span / target)) or 1))


def split_span(a: float, b: float, n: int) -> List[float]:
    """n+1 frame-grid boundaries a..b (even shots)."""
    return [a] + [compose.q(a + (b - a) * i / n) for i in range(1, n)] + [b]


def anchors_for(seg: Dict[str, Any], frames: Dict[str, Dict[str, Any]], events: Dict[str, Dict[str, Any]], n: int) -> List[float]:
    """Source times for a segment: its own frames first, then points inside its events (spread), n of them."""
    out = [frames[f]["t"] for f in seg.get("frame_ids", []) if f in frames]
    evs = [events[e] for e in seg["event_ids"] if e in events]
    k = 0
    while len(out) < n and evs and k < n * 4:
        e = evs[k % len(evs)]
        frac = (0.5 + 0.29 * (k // len(evs))) % 1.0
        out.append(round(e["start"] + (e["end"] - e["start"]) * frac, 2))
        k += 1
    return out


def plan_shots(script: Dict[str, Any], vt: Dict[str, Any], frames: Sequence[Dict[str, Any]], events: Sequence[Dict[str, Any]],
               a: float, b: float, cuts_in, min_gap: float = 7.0) -> Dict[str, Any]:
    """Contiguous shot list over [0, total): per segment even shots, source window around the frame/event anchors,
    pushed off hidden scene cuts (cuts_in(lo, hi) → absolute cut times) and off windows already used."""
    fr = {f["id"]: f for f in frames}
    ev = {e["id"]: e for e in events}
    spans = compose.segment_spans(vt)
    shots, warns, used = [], [], []
    for seg in script["segments"]:
        sa, sb = spans[seg["id"]]
        n = shot_count(sb - sa)
        bounds = split_span(sa, sb, n)
        pool = anchors_for(seg, fr, ev, n + 4)
        for i in range(n):
            d = round(bounds[i + 1] - bounds[i], 4)
            pick = None
            for want in (d, d * 0.8, d * 0.6):
                for t in pool:
                    if any(abs(t - u) < min_gap for u in used) and len(pool) > 1:
                        continue
                    lo, hi = max(a, t - 2.0), min(b, t + want + 2.0)
                    w = compose.cut_free_window(cuts_in(lo, hi), lo, hi, t - want / 2, want)
                    if w is not None:
                        pick = (t, w, want)
                        break
                if pick:
                    break
            if pick is None:                       # no clean window near any anchor: take the anchor as is, say so
                t = pool[0] if pool else (a + b) / 2
                st = min(max(a, t - d / 2), b - d)
                pick = (t, (round(st, 3), round(st + d, 3)), d)
                warns.append(f"{seg['id']} shot {i + 1}: no cut-free window near {mmss(t - a)} — taken as is")
            t, (s0, s1), want = pick
            if want < d - 1e-3:                    # a shorter clean window: the slot keeps its length, the clip is held
                warns.append(f"{seg['id']} shot {i + 1}: clean window {want:.1f}s < slot {d:.1f}s")
                s1 = round(s0 + d, 3) if s0 + d <= b else s1
            if t in pool:
                pool.remove(t)
            used.append(t)
            shots.append({"n": len(shots) + 1, "segment_id": seg["id"], "candidate_id": f"{seg['id']}.{i + 1}",
                          "src_start": round(s0, 3), "src_end": round(s0 + d, 3), "t0": bounds[i], "t1": bounds[i + 1],
                          "duration": d, "why": "frame/event anchor", "fallback": False, "scene_shot": "-", "speaking": "n/a"})
    return {"shots": shots, "total": vt["total"], "fps": compose.FPS, "warnings": warns}


# ── 6. the run ──────────────────────────────────────────────────────────────

def known_crime_stems(text: str) -> List[str]:
    return recap.crime_stems_in(text)


def run_lite(input_path: str, window: Tuple[float, float], *, duration: float = 60.0, run_dir: Optional[str] = None,
             n_frames: int = 14, stop_rub: float = LITE_STOP_RUB, tts_voice: str = "aidar", tts_provider: str = "silero",
             language: str = "ru", subtitles: bool = True, censor_profanity: bool = True, forbid_names: Sequence[str] = (),
             day_limit_usd: Optional[float] = None, chat=None, tts=None, script_file: Optional[str] = None) -> Dict[str, Any]:
    from . import recap_runner as rr
    t_all = time.time()
    a, b = window
    out = Path(run_dir or (ROOT.parent / "runtime" / "trailer" / f"lite-{time.strftime('%H%M%S')}")).resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT.parent / "runtime" / "trailer" / "_cache"
    M: Dict[str, Any] = {"mode": "trailer_cinematic", "path": "lite/economy", "version": LITE_VERSION, "run_dir": str(out),
                         "source_path": str(Path(input_path).resolve()), "window_sec": [a, b], "target_duration": duration,
                         "stop_rub": stop_rub, "warnings": [], "timings_sec": {}}
    T = M["timings_sec"]

    def tick(k: str, t0: float) -> None:
        T[k] = round(T.get(k, 0.0) + time.time() - t0, 2)

    def say(m: str) -> None:
        print(f"[lite {time.time() - t_all:6.1f}s] {m}", flush=True)

    def save() -> None:
        T["total"] = round(time.time() - t_all, 2)
        _dump(out / "metrics.json", M)

    M["media_env"] = _desktop_media_env()
    info = media.probe(input_path)
    sha = _sha256(input_path)
    M["source_sha256"] = sha
    asr = run_asr(input_path, cache_dir, sha, language)            # cached → 0 ₽ and no ASR time
    M["asr"] = {k: asr["meta"].get(k) for k in ("backend", "model", "words", "cached")}
    raw_words = asr.get("words") or []

    # 1. digest + frames (local)
    t0 = time.time()
    events = digest(raw_words, a, b)
    dtext = digest_text(events, a)
    frames = choose_frames(input_path, raw_words, a, b, max(LITE_FRAMES[0], min(LITE_FRAMES[1], n_frames)), work)
    tick("digest_frames", t0)
    M["digest"] = {"events": len(events), "chars": len(dtext), "frames": [{k: f[k] for k in ("id", "t", "rel", "faces")} for f in frames]}
    (out / "digest.txt").write_text(dtext, encoding="utf-8")
    say(f"digest {len(events)} events / {len(dtext)} chars; {len(frames)} frames ≤ {FRAME_MAX_PX}px")
    window_text = " ".join(w.get("word", "") for w in raw_words if a <= w.get("start", 0) < b)
    sm: Dict[str, Any] = {"events": events, "forbid_names": recap.forbidden_names({}, forbid_names),
                          "source_crime_stems": known_crime_stems(window_text)}

    # 2. TTS rate → script length window
    if tts is None:
        from scripts.shared.tts import get_provider
        tts = get_provider(tts_provider, tts_voice if tts_provider == "silero" else None)
    if not tts.available():
        raise RuntimeError(f"TTS provider {tts.name} is not available")
    scfg = recap.style_cfg("punchy")
    rate = tts.calibrate(str(cache_dir))
    tempo = float(scfg["tempo"])
    rate = dict(rate, chars_per_sec=round(rate["chars_per_sec"] * tempo, 3), base_sec=round(rate["base_sec"] / tempo, 4), tempo=tempo)
    M["tts"] = dict(tts.info(), rate=rate, tempo=tempo)
    oh = float(scfg["overhead_sec"])
    lo_c, hi_c = (int((duration - oh) * rate["chars_per_sec"] * 0.93), int((duration - oh) * rate["chars_per_sec"] * 1.07))

    # 3. LLM: one Story+Writer call, forecast before it, hard stop
    _load_llm_env()
    from scripts.shared.llm_provider import certifi_transport, chat_json
    kind, tw = rr.select_provider("timeweb") if chat is None else ("fake", None)
    if day_limit_usd is None:
        day_limit_usd = float(os.environ.get("SONYA_TRAILER_LLM_DAY_USD") or budget.DEFAULT_DAY_USD)   # never raised here
    ledgers, chat_kwargs = None, None
    if tw is not None:
        chat_kwargs = dict(providers=[tw], retries=2, max_request_bytes=rr.MAX_REQUEST_BYTES, transport=certifi_transport())
        day = budget.DailyLedger(str(cache_dir / "llm_spend_timeweb.json"), day_limit_usd)
        vled = budget.VideoLedger(str(cache_dir / f"spend_lite-{sha[:16]}-{int(a)}-{int(b)}.json"), stop_rub)
        ledgers = [day, vled]
        M["day_limit_usd"], M["day_spent_before_usd"] = day_limit_usd, round(day.spent(), 5)
    model = tw.model if tw else "fake"
    M["llm_provider"] = f"{tw.name}/{model}" if tw else "fake"
    llm = budget.LLM(chat or chat_json, stop_rub / budget.RUB_PER_USD, model, cache_dir=str(cache_dir / "llm"),
                     ledger=ledgers, chat_kwargs=chat_kwargs, reasoning=tw is None)
    step_v = f"lite-writer-{LITE_VERSION}"

    def spent_rub() -> float:
        vl = [x for x in llm.ledgers if isinstance(x, budget.VideoLedger)]
        return budget.rub(vl[0].spent()) if vl else budget.rub(llm.spent)

    def pre_gate(msgs, label, max_out=MAX_OUT_TOKENS):
        fc = gate(msgs, model, spent_rub(), stop_rub, max_out)
        M.setdefault("forecasts", []).append(dict(fc, step=label))
        say(f"forecast {label}: ≤ {fc['rub']:.2f} ₽ (spent {fc['spent_rub']:.2f} of {stop_rub:.1f} ₽)")

    try:
        t0 = time.time()
        if script_file:
            raw = json.loads(Path(script_file).read_text(encoding="utf-8"))
            script = validate_lite(raw.get("script", raw), sm, frames, lo_c, hi_c, 0.5)
            M["warnings"].append(f"script taken from {script_file}, not from the model")
        else:
            msgs = lite_messages(events, frames, a, b, lo_c, hi_c, rate, duration, sm["forbid_names"])
            pre_gate(msgs, "story_writer")
            holder: Dict[str, Any] = {}

            def validator_fn(d):                              # only the shape: the content checks are ours (one text fix)
                if not isinstance(d, dict) or not isinstance(d.get("segments"), list):
                    raise SchemaError("JSON with segments[] expected")
                return d
            data = llm.ask("lite_story_writer", msgs, step_v, validator_fn, temperature=0.8, max_out=MAX_OUT_TOKENS, effort="medium")
            holder["data"] = data
            _dump(out / "model_answer_1.json", data)
            sm["forbid_names"] = recap.forbidden_names({"characters": data.get("characters") or []}, forbid_names)
            n_rng = (recap.STYLES["punchy"]["segments"][0], recap.STYLES["punchy"]["segments"][1])
            defects: List[str] = quick_defects(data, lo_c, hi_c, n_rng)
            script = None
            try:
                script = validate_lite(data, sm, frames, lo_c, hi_c)
                defects += [d for d in free_checks(script, frames, events) if d not in defects]
            except SchemaError as e:
                if not defects or "segments" not in str(e):
                    defects.append(str(e))
            if defects:
                M["defects_found"] = defects
                say(f"free checks found {len(defects)} defect(s) → ONE text-only fix")
                prev = script if script is not None else {"segments": data["segments"]}
                fmsgs = fix_messages(events, a, prev, defects, lo_c, hi_c, rate)
                pre_gate(fmsgs, "text_fix", FIX_OUT_TOKENS)
                data2 = llm.ask("lite_text_fix", fmsgs, step_v + "+fix", validator_fn, temperature=0.5, max_out=FIX_OUT_TOKENS, effort="medium")
                _dump(out / "model_answer_2.json", data2)
                sm["forbid_names"] = recap.forbidden_names({"characters": (data2.get("characters") or data.get("characters") or [])}, forbid_names)
                script = validate_lite(data2, sm, frames, lo_c, hi_c, 0.5)      # still strict on names / topics; length slack only
                left = free_checks(script, frames, events)
                if left:
                    M["warnings"].append(f"after the fix the free checks still say: {left}")
        tick("llm", t0)
        _dump(out / "script.json", {"version": step_v, "style": "punchy", "script": script,
                                    "chars": sum(len(s["text"]) for s in script["segments"])})
        say(f"script: {sum(len(s['text']) for s in script['segments'])} chars in {len(script['segments'])} segments")

        # 4. TTS + length
        from scripts.shared.tts import wav_duration

        def synth(scr):
            wavs, secs = {}, {}
            for seg in scr["segments"]:
                secs[seg["id"]] = []
                for j, sent in enumerate(recap.sentences_of(seg["text"])):
                    h = hashlib.sha256(f"{tts.name}|{tts.voice}|{sent}".encode()).hexdigest()[:20]
                    p = cache_dir / "tts" / f"{h}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if not p.is_file():
                        tts.synthesize(sent, str(p))
                    if tempo != 1.0:
                        p = Path(rr.speed_up(str(p), str(p.with_suffix(f".x{tempo:.2f}e.wav")), tempo))
                    wavs[(seg["id"], j)] = str(p)
                    secs[seg["id"]].append(wav_duration(str(p)))
            return wavs, compose.voice_timeline(scr, secs, pauses=scfg["pauses"])

        t0 = time.time()
        wavs, vt = synth(script)
        tick("tts", t0)
        if not rr.TARGET_LO <= vt["total"] <= rr.TARGET_HI:
            for sc in (1.25, 1.1, 0.85, 0.7):
                alt = compose.voice_timeline(script, {s["id"]: [r["sec"] for r in s["sentences"]] for s in vt["segments"]},
                                             pause_scale=sc, pauses=scfg["pauses"])
                if abs(alt["total"] - duration) < abs(vt["total"] - duration):
                    vt = alt
                if rr.FIT_LO <= vt["total"] <= rr.FIT_HI:
                    break
            M["warnings"].append(f"voice length needed pause scaling → {vt['total']:.1f}s")

        # 5. shots, render, BEEP, assemble
        t0 = time.time()
        sha_short = sha

        def cuts_in(lo, hi):
            return sorted(set(media.frame_cuts(input_path, lo, hi, str(cache_dir / "clips"), sha_short))
                          | set(validator.hard_cuts_in(input_path, lo, hi)))
        timeline = plan_shots(script, vt, frames, events, a, b, cuts_in)
        problems = compose.check_plan(timeline)
        if problems:
            raise RuntimeError(f"invalid timeline: {problems}")
        tick("shot_plan", t0)
        t0 = time.time()
        (work / "shots").mkdir(exist_ok=True)
        files = media.render_shots(input_path, timeline["shots"], work / "shots", width=info["width"], height=info["height"],
                                   cache_dir=str(cache_dir / "clips"), source_sha256=sha)
        tick("render_shots", t0)
        t0 = time.time()
        voice = str(work / "voice.wav")
        compose.write_voice(vt, wavs, voice)
        stab = compose.stability(vt, voice)
        g = media.voice_gain_db(voice)
        sw, _ = censor.subtitle_words(vt)
        mix_voice = voice
        if censor_profanity:
            try:
                asr_w = censor.asr_words_fn(language, work)(voice)
            except Exception as e:                                     # noqa: BLE001
                asr_w = None
                M["warnings"].append(f"beep: ASR of the voice failed ({str(e)[:100]}), word times are estimated")
            cplan = censor.plan(vt, asr_w)
            if cplan.intervals:
                mix_voice = censor.beep_voice(voice, cplan.intervals, str(work / "voice_beep.wav"), cplan.cfg)
                sw = cplan.words
            M["beep"] = cplan.report()
        ass = None
        if subtitles:
            ass = str(out / "narration.ass")
            Path(ass).write_text(subs.to_ass(sw, vt["total"]), encoding="utf-8")
        final = str(out / "final.mp4")
        media.assemble_recap(files, mix_voice, g["gain_db"], vt["total"], ass, final,
                             shot_frames=[round((s["t1"] - s["t0"]) * 30) for s in timeline["shots"]])
        tick("render_assemble", t0)

        # 6. technical validation (deterministic)
        t0 = time.time()
        text = " ".join(s["text"] for s in script["segments"])
        val = validator.validate(final, tl=timeline, vt=vt, voice_wav=mix_voice, gain_db=g["gain_db"], shot_files=files, source=input_path,
                                 script_text=text, spread_db=stab["spread_db"], asr=rr.asr_text_fn(language, work),
                                 dur_range=(rr.TARGET_LO, rr.TARGET_HI), target=duration, ass_path=ass, pace=rr.PACE_PUNCHY,
                                 forbid_names=sm["forbid_names"], expected_heard=censor.without_profanity(text) if censor_profanity else None)
        tick("validator", t0)
        say(f"validator: passed={val['passed']} failed={val['failed']}")
        _dump(out / "validation.json", val)
        _dump(out / "timeline.json", timeline)
        _dump(out / "narration.json", {"tts": M["tts"], "segments": [
            dict(s, text=next(x["text"] for x in script["segments"] if x["id"] == s["id"])) for s in vt["segments"]]})
        items = [{"n": s["n"], "t": (s["t0"] + s["t1"]) / 2, "label": f"{s['segment_id']} #{s['n']}"} for s in timeline["shots"]]
        sheets = []
        for k in range(0, len(items), 12):
            sp = str(out / f"sheet_{k // 12}.jpg")
            media.contact_sheet(final, items[k:k + 12], sp)
            sheets.append(sp)
        M.update(final_output_path=final, final_duration=val["duration"], technical_passed=val["passed"],
                 technical_failed=val["failed"], shots=len(timeline["shots"]), contact_sheets=sheets,
                 final_size_mb=round(Path(final).stat().st_size / 1e6, 1),
                 invented_gags=[dict(gg, segment_id=s["id"]) for s in script["segments"] for gg in s.get("gags", [])])
    finally:
        M["llm"] = llm.summary()
        M["cost_usd"] = llm.spent
        M["cost_rub"] = budget.rub(llm.spent)
        M["calls"] = llm.calls
        save()
    return M
