"""
storystyle.py — trailer_cinematic style «story» (owner/tester 2026-11-05: TikTok film-recap format, serious tone).

    ONE episode of a feature film (10–25 min, one storyline), told by the hero in the FIRST person, 85–105 s,
    2.9–3.4 words/s. v2 (tester: «скучно, мелко»): the film is CROPPED to 1:1 (or 4:5 / 3:4) around the faces and fills the
    whole frame; big faces and motion are chosen, the cuts get faster towards the climax, static shots get a slow push-in;
    bold white subtitles (1–2 lines) lie ON the picture; two audio variants from one run:
        A = narrator only, B = narrator + the film's sound far below (ducked where the film's heroes talk) + quiet accents
        (whoosh / impact / riser) from the owner's own sound pack.  The picture helpers live in storyfx.py.

Money (v2 rebuild: hard stop 3 ₽): the episode pick is the cached answer of v1 (0 ₽); ONE text-only Story+Writer call by the
ready digest (no stills any more: the shots are chosen by local face/motion features), at most one text-only fix.
No video is ever sent, no Judge. Everything else (checks, TTS tempo fit, shots, render, validation) is local and free.

The energetic style («punchy») and the economy path in lite.py are not touched; this module only reuses their helpers.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import budget, compose, lite, media, recap
from .budget import SchemaError

STORY_VERSION = "trailer-story-v2"
STORY_STOP_RUB = 3.0
EP_MIN, EP_MAX = 10 * 60.0, 25 * 60.0           # episode length, seconds
WIN_LEN = 16 * 60.0                              # candidate window length (snapped to speech pauses afterwards)
WIN_STEP = 150.0
N_CANDIDATES = 5
TARGET_SEC = (85.0, 105.0)
TARGET_AIM = 95.0
PACE = (2.9, 3.4)                                # words per second of the finished video
SHOT_SEC = (1.2, 3.0)                            # whole range; the rhythm by position is storyfx.shot_range
BED_DB = -17.0                                   # film sound under the narrator (brief: −18…−22); speech shots are ducked further
BED_DUCK_DB = -6.0                               # extra duck where the film's heroes talk (nothing intelligible over the narrator)
SUB_TOP = 0.56                                   # top of the subtitle block, fraction of the frame height (samples: ≈ 55–65 %)
SUB_SIZE = 66                                    # bold, medium: the samples' text is ≈ 5.5 % of the frame height
SUB_MAX_W = 930.0                                # one LINE, px at 1080 (outline included in the check)
SUB_LINES = 2
SUB_OUTLINE = 4
SUB_FONT = "Montserrat-ExtraBold.ttf"
SUB_FONT_NAME = "Montserrat ExtraBold"
PAUSES = {"sent": 0.14, "sent_q": 0.22, "after": {"short": 0.16, "normal": 0.30, "long": 0.55}}
FUNCTIONS = ("hook", "origin", "rising", "twist", "cliffhanger")

# stakes / drama stems, general (no film specifics): the local scorer only needs «something is at stake here»
STAKES = re.compile(r"(деньг|миллион|тысяч|долляр|фбр|агент|арест|суд|тюрьм|убь|убит|смерт|умер|предал|предател|обман|лж|"
                    r"врал|правд|секрет|потер|потеря|развод|жен[аыуе]|дочь|отец|брат|друг|семь|компани|акци|сделк|"
                    r"риск|опасн|кризис|катастроф|авари|разбил|утону|шторм|спас|прощ|клян|обещ|никогда|навсегда|"
                    r"последн|конец|всё кончено|выбор|должен|нельзя|поздно|беги|стой|помоги|прошу|умоляю)", re.I)
WORD_RE = re.compile(r"\w+(?:-\w+)*")
MAT_RE = re.compile(r"(?<![\w])(\w*(?:хуй|хуе|хуё|пизд|пезд|ебан|ебат|ебал|ебну|ебис|ебыр|заеб|наеб|поеб|уеб|отъеб|"
                    r"блять|бляд|блядь|сука|суки|суку|сучк|мудак|мудил|гандон|залуп|дроч|шлюх|пидор|пидар)\w*)", re.I)
FIRST_PERSON = re.compile(r"(?<![\w-])(я|меня|мне|мной|мой|моя|моё|мое|мои|моих|моему|моей|мою|мы|нас|нам|нами|наш|наша|наши)(?![\w-])", re.I)


def n_words(text: str) -> int:
    return len(WORD_RE.findall(text))


# ── 1. episode candidates (local, free) ─────────────────────────────────────

def pauses_in(raw_words: Sequence[Dict[str, Any]], min_gap: float = 2.5) -> List[float]:
    """Moments of a speech pause ≥ min_gap (the middle of the gap): natural places to start/end an episode."""
    out, prev = [], None
    for w in raw_words:
        if prev is not None and w.get("start", 0) - prev >= min_gap:
            out.append((prev + w["start"]) / 2)
        prev = w.get("end", w.get("start", 0))
    return out


def snap(t: float, marks: Sequence[float], radius: float = 90.0) -> float:
    near = [m for m in marks if abs(m - t) <= radius]
    return min(near, key=lambda m: abs(m - t)) if near else t


def window_score(raw_words: Sequence[Dict[str, Any]], a: float, b: float) -> Dict[str, float]:
    ws = [w for w in raw_words if a <= w.get("start", 0) < b]
    text = " ".join(w.get("word", "") for w in ws)
    dur = max(1.0, b - a)
    dens = len(ws) / dur                                    # words / s, dialogue-heavy = a story with talk
    stakes = len(STAKES.findall(text)) / max(1, len(ws)) * 100
    # even coverage: share of 60 s bins that contain speech (a long silent / action-only stretch is a bad episode)
    bins = {int((w["start"] - a) // 60) for w in ws}
    cover = len(bins) / max(1, math.ceil(dur / 60))
    first = len(FIRST_PERSON.findall(text)) / max(1, len(ws)) * 100
    mat = len(MAT_RE.findall(text)) / max(1, len(ws)) * 100
    score = min(dens, 3.0) * 1.0 + stakes * 0.9 + cover * 2.0 + min(first, 8) * 0.15 - mat * 0.1
    return {"score": round(score, 3), "words_per_sec": round(dens, 3), "stakes_pct": round(stakes, 2), "cover": round(cover, 2),
            "mat_pct": round(mat, 2)}


def candidates(raw_words: Sequence[Dict[str, Any]], film_sec: float, n: int = N_CANDIDATES, win: float = WIN_LEN,
               step: float = WIN_STEP, skip_head: float = 480.0, skip_tail: float = 600.0,
               max_overlap: float = 0.25, bad: Sequence[float] = (), bad_margin: float = 20.0) -> List[Dict[str, Any]]:
    """Top-n non-overlapping windows (10–25 min) by the local score; edges snapped to speech pauses. Windows that contain a
    time from `bad` (damaged stretch of the file, ± bad_margin) are dropped. Pure."""
    marks = pauses_in(raw_words)
    pool = []
    t = skip_head
    while t + win <= film_sec - skip_tail:
        a, b = snap(t, marks), snap(t + win, marks)
        if EP_MIN <= b - a <= EP_MAX and not any(a - bad_margin <= x <= b + bad_margin for x in bad):
            pool.append(dict(a=round(a, 1), b=round(b, 1), **window_score(raw_words, a, b)))
        t += step
    pool.sort(key=lambda c: -c["score"])
    out: List[Dict[str, Any]] = []
    for c in pool:
        if all(min(c["b"], o["b"]) - max(c["a"], o["a"]) <= max_overlap * (c["b"] - c["a"]) for o in out):
            out.append(c)
        if len(out) >= n:
            break
    out.sort(key=lambda c: c["a"])
    for i, c in enumerate(out):
        c["id"] = f"C{i + 1}"
    return out


def damaged_times(src: str, lo: float, hi: float, step: float = 15.0, cache: Optional[Path] = None) -> List[float]:
    """Probe the file every `step` s (2 s decode each): the times where the decoder reports errors. A long film from the
    internet can hold a broken stretch (the Wolf file: ≈ 21:23–23:04) — no frame or shot may come from there."""
    if cache and cache.is_file():
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            pass
    bad = []
    t = lo
    while t < hi:
        r = subprocess.run([media.ffmpeg_bin(), "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{t:.1f}", "-t", "2", "-i", src, "-an",
                            "-f", "null", "-"], capture_output=True, text=True)
        if r.returncode != 0 or len(r.stderr.splitlines()) > 5:
            bad.append(round(t, 1))
        t += step
    if cache:
        cache.write_text(json.dumps(bad))
    return bad


def candidate_digest(raw_words: Sequence[Dict[str, Any]], c: Dict[str, Any], chars: int = 1500) -> str:
    """~1.5k chars of the strongest lines of a candidate, in order, m:ss from its start (cheap text for the picker)."""
    ev = lite.digest(raw_words, c["a"], c["b"], target_chars=chars, event_sec=(c["b"] - c["a"]) / 14)
    return "\n".join(f"({lite.mmss(e['start'] - c['a'])}) {e['text']}" for e in ev if e["text"])


def pick_messages(raw_words: Sequence[Dict[str, Any]], cands: Sequence[Dict[str, Any]], film_title: str = "") -> List[Dict[str, Any]]:
    body = "\n\n".join(f"{c['id']} — {lite.mmss(c['b'] - c['a'])} мин. Выжимка реплик:\n{candidate_digest(raw_words, c)}" for c in cands)
    task = ("Перед тобой несколько кандидатов-эпизодов ОДНОГО фильма (выжимки реплик по расшифровке; расшифровка может ошибаться). "
            "Нужно выбрать ОДИН эпизод для серьёзного трейлера-рассказа от первого лица героя: одна сильная линия героя, "
            "понятное начало, нарастание ставок, поворот и обрыв на самом интересном (зритель должен захотеть досмотреть фильм). "
            "Выбирай ТОЛЬКО по этим выжимкам, ничего не выдумывай.\n\n" + body +
            "\n\nВерни JSON: {\"choice\": \"C?\", \"why\": \"1–2 предложения, что за линия героя и почему она\", "
            "\"ranking\": [\"C?\", ...]}.")
    return [{"role": "system", "content": "Ты редактор трейлеров. Отвечай только JSON."}, {"role": "user", "content": task}]


def parse_pick(data: Any, cands: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    if not isinstance(data, dict):
        raise SchemaError("JSON object expected")
    cid = str(data.get("choice") or "").strip().upper()
    by = {c["id"]: c for c in cands}
    if cid not in by:
        raise SchemaError(f"choice must be one of {sorted(by)}")
    return by[cid], str(data.get("why") or "")[:400]


# ── 2. Story + Writer call ──────────────────────────────────────────────────

STORY_SYSTEM = (
    "Ты — сценарист серьёзных трейлеров-пересказов для TikTok. Рассказ ведёт ГЕРОЙ фильма от ПЕРВОГО ЛИЦА («Я…», «Мой…», "
    "«Меня…»). Тон серьёзный и драматичный: без шуток, иронии, сленга и мата. Короткие ясные фразы, ровный темп. "
    "ТОЛЬКО факты из выжимки расшифровки: ничего не выдумывай, не придумывай мыслей, мотивов, событий, чисел, причин, которых "
    "там нет; если не уверен — не пиши. Не переворачивай смысл реплик: если в выжимке что-то запрещают, отрицают или отказываются "
    "делать — так и пиши. Имён людей и названий компаний/мест, которые являются именами, нет совсем (ни полных, ни "
    "уменьшительных, ни кличек) — только роли: «мой партнёр», «агент», «жена», «друг», «босс». Цифры только словами. "
    "Структура: 1) ХУК — одно короткое СТРАННОЕ, интригующее утверждение героя, неожиданный факт из эпизода, после которого "
    "хочется понять «как так?» (по духу: «Я сам посадил себя в камеру и выбросил ключ» — это ПРИМЕР СТИЛЯ из другого фильма, "
    "его нельзя использовать; у тебя должен быть свой факт из выжимки), без разгадки; 2) откуда всё началось; 3) нарастание: "
    "ставки растут; 4) поворот; 5) обрыв на САМОМ ОСТРОМ моменте эпизода (угроза, шок, вопрос без ответа), без развязки и без "
    "морали — зритель должен захотеть досмотреть фильм. Отвечай только JSON.")


def story_messages(events: Sequence[Dict[str, Any]], frames: Sequence[Dict[str, Any]], a: float, b: float, lo_w: int, hi_w: int,
                   forbid: Sequence[str] = (), why: str = "") -> List[Dict[str, Any]]:
    """v2: TEXT ONLY by default (frames = [] → no stills; the shots are chosen locally by faces/motion). With frames the v1 call."""
    task = (
        f"ЗАДАЧА: рассказ героя от первого лица, ≈ {TARGET_AIM:.0f} секунд голоса, по эпизоду фильма длиной {lite.mmss(b - a)}. "
        f"ВЕСЬ text (сумма по сегментам) — от {lo_w} до {hi_w} слов (это важно: меньше {lo_w} — брак, пиши полнее, 22–30 слов в "
        "сегменте). Считай слова.\n"
        + (f"Линия эпизода (выбрана заранее): {why}\n" if why else "") +
        "\nВЫЖИМКА РАСШИФРОВКИ (единственный источник фактов; время от начала эпизода; расшифровка может ошибаться и путать "
        "говорящих, мат пересказывай нейтрально):\n" + lite.digest_text(events, a) +
        ("\n\nКАДРЫ: " + ", ".join(f"{f['id']}={f['rel']}" for f in frames) + " (картинки ниже, по порядку)." if frames else "") +
        f"\n\nСценарий — 11–14 сегментов, функции по порядку: hook (1 короткий сегмент, 8–16 слов: странное утверждение героя, "
        "факт из эпизода), origin (2–3), rising (4–6), "
        "twist (1–2), cliffhanger (последний, 1–2 короткие фразы, обрыв на самом остром моменте). Остальные сегменты — 1–2 "
        "предложения, 22–30 слов, предложения ≤ 14 слов. Не меньше 60% сегментов содержат слова «я/мой/меня/мне/мы». "
        "Для каждого сегмента: event_ids (1–3 id E…, ИМЕННО те события, о которых говорит текст; кроме hook идут по "
        "возрастанию времени; у hook — событие, из которого взят факт), "
        + ("frame_ids (1–3 кадра F…, подходящих к ЭТОМУ тексту), " if frames else "") +
        "visual_intent (5–12 слов), pause_after (short|normal|long).\n"
        "Также верни characters — реальные ИМЕНА героев и актёров из эпизода (чтобы запретить их в тексте).\n"
        "Верни JSON: {\"title\": \"…\", \"characters\": [{\"role\": \"…\", \"name\": \"…\"}], \"segments\": [{\"id\": \"S1\", "
        "\"function\": \"" + "|".join(FUNCTIONS) + "\", \"text\": \"…\", \"event_ids\": [\"E…\"], "
        + ("\"frame_ids\": [\"F…\"], " if frames else "") + "\"visual_intent\": \"…\", \"pause_after\": \"short|normal|long\"}]}.")
    if forbid:
        task += "\n\nЗАПРЕЩЁННЫЕ СЛОВА (имена): " + ", ".join(x.rstrip("$") for x in forbid) + "."
    if not frames:
        return [{"role": "system", "content": STORY_SYSTEM}, {"role": "user", "content": task}]
    parts: List[Dict[str, Any]] = [{"type": "text", "text": task}]
    from . import visual
    for f in frames:
        parts.append({"type": "text", "text": f"{f['id']} ({f['rel']}):"})
        parts.append(visual.image_part(f["path"]))
    return [{"role": "system", "content": STORY_SYSTEM}, {"role": "user", "content": parts}]


def fix_messages(events, a: float, script: Dict[str, Any], defects: Sequence[str], lo_w: int, hi_w: int) -> List[Dict[str, Any]]:
    task = (f"Исправь сценарий рассказа от первого лица. Общая длина text: {lo_w}–{hi_w} слов.\n"
            "ДЕФЕКТЫ (исправь все, остальное сохрани):\n- " + "\n- ".join(defects) +
            "\n\nПРЕДЫДУЩИЙ ВАРИАНТ (сегменты можно добавлять и переписывать, JSON тот же):\n" + json.dumps(script, ensure_ascii=False) +
            "\n\nВЫЖИМКА (для сверки фактов):\n" + lite.digest_text(events, a)[:6500] +
            "\n\nВерни тот же JSON целиком.")
    return [{"role": "system", "content": STORY_SYSTEM}, {"role": "user", "content": task}]


# ── 3. free checks of the script ────────────────────────────────────────────

def story_defects(data: Any, events: Sequence[Dict[str, Any]], frames: Sequence[Dict[str, Any]], lo_w: int, hi_w: int,
                  forbid: Sequence[str], crime_ok: Sequence[str], max_sent: int = 16) -> List[str]:
    """All the free, deterministic checks at once → a list of defects (empty = clean). Pure."""
    segs = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segs, list) or not segs:
        return ["нет segments[] в ответе"]
    bad: List[str] = []
    texts = [str((s or {}).get("text", "")) for s in segs if isinstance(s, dict)]
    words = sum(n_words(t) for t in texts)
    if not 11 <= len(segs) <= 14:
        bad.append(f"сегментов {len(segs)}, нужно 11–14")
    if not lo_w <= words <= hi_w:
        bad.append(f"всего {words} слов, нужно {lo_w}–{hi_w} ({'допиши' if words < lo_w else 'сократи'} примерно на "
                   f"{abs(words - (lo_w + hi_w) // 2)} слов)")
    fn = [(s or {}).get("function") for s in segs]
    if fn and fn[0] != "hook":
        bad.append("первый сегмент должен быть hook")
    if fn and fn[-1] != "cliffhanger":
        bad.append("последний сегмент должен быть cliffhanger (обрыв на самом интересном)")
    if texts and fn and fn[0] == "hook":
        if n_words(texts[0]) > 20:
            bad.append(f"hook слишком длинный ({n_words(texts[0])} слов): нужно одно короткое странное утверждение, 8–16 слов")
        if not FIRST_PERSON.search(texts[0]):
            bad.append("hook должен быть от первого лица (я/мой/меня)")
    first = sum(1 for t in texts if FIRST_PERSON.search(t))
    if first < 0.6 * len(texts):
        bad.append(f"от первого лица (я/мой/меня) только {first} из {len(texts)} сегментов — нужно не меньше 60%")
    mat = sorted({m.group(1) for t in texts for m in MAT_RE.finditer(t)})
    if mat:
        bad.append(f"мат в тексте: {', '.join(mat[:4])} — убери, тон серьёзный")
    if any(re.search(r"\d", t) for t in texts):
        bad.append("цифры в тексте — числа только словами")
    names = sorted({w for t in texts for w in recap.find_names(t, forbid)} | {w for t in texts for w in lite.stray_capitals(t)})
    if names:
        bad.append(f"имена/клички в тексте: {', '.join(names[:6])} — замени на роль («мой партнёр», «агент», «жена»)")
    inv = sorted({c for t in texts for c in recap.crime_stems_in(t) if c not in crime_ok})
    if inv:
        bad.append(f"слова о насилии/преступлениях, которых нет в источнике: {inv} — пиши нейтрально")
    long_s = [s for t in texts for s in recap.sentences_of(t) if n_words(s) > max_sent]
    if long_s:
        bad.append(f"{len(long_s)} предложений длиннее {max_sent} слов: «{long_s[0][:60]}…» — разбей на короткие")
    ids = {e["id"] for e in events}
    fids = {f["id"] for f in frames}
    ev_t = {e["id"]: (e["start"] + e["end"]) / 2 for e in events}
    prev = None
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            continue
        eids = [x for x in (s.get("event_ids") or []) if x in ids]
        if not eids:
            bad.append(f"S{i + 1}: нет верных event_ids (только E… из выжимки)")
            continue
        if fids and not [x for x in (s.get("frame_ids") or []) if x in fids]:
            bad.append(f"S{i + 1}: нет верных frame_ids (F… из списка)")
        if s.get("function") == "hook":
            continue
        t = min(ev_t[e] for e in eids)
        if prev is not None and t < prev - 120.0:
            bad.append(f"S{i + 1}: события идут назад во времени — после hook рассказ идёт по порядку эпизода")
        prev = max(prev or t, t)
    return bad


def clean_script(data: Dict[str, Any], events: Sequence[Dict[str, Any]], frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise the model JSON into the script the composer uses (ids S1…, valid ids only, a final dot)."""
    ids = {e["id"] for e in events}
    fids = {f["id"] for f in frames}
    out = []
    for i, s in enumerate(data["segments"]):
        text = re.sub(r"^[\s\-—–«»\"']+", "", str(s["text"])).replace("«", "").replace("»", "").replace('"', "").strip()
        if not re.search(r"[.!?…]$", text):
            text += "."
        pa = s.get("pause_after") if s.get("pause_after") in PAUSES["after"] else "normal"
        out.append({"id": f"S{i + 1}", "function": s.get("function") if s.get("function") in FUNCTIONS else "rising", "text": text,
                    "event_ids": [x for x in s.get("event_ids", []) if x in ids][:4],
                    "frame_ids": [x for x in s.get("frame_ids", []) if x in fids][:3],
                    "visual_intent": str(s.get("visual_intent") or "")[:200], "pause_after": pa})
    out[0]["function"], out[-1]["function"] = "hook", "cliffhanger"
    return {"segments": out, "title": str(data.get("title") or "")[:120],
            "characters": [c for c in (data.get("characters") or []) if isinstance(c, dict)][:12]}


# ── 4. TTS tempo fit (the voice speed is chosen AFTER the text, so the pace lands in 2.9–3.4) ─────────────────

def pause_total(script: Dict[str, Any]) -> float:
    t = 0.0
    for i, s in enumerate(script["segments"]):
        ss = recap.sentences_of(s["text"])
        t += sum((PAUSES["sent_q"] if x.endswith("?") else PAUSES["sent"]) for x in ss[:-1])
        if i < len(script["segments"]) - 1:
            t += PAUSES["after"][s["pause_after"]]
    return t


def fit_tempo(raw_voice_sec: float, pause_sec: float, words: int, aim: float = TARGET_AIM, pace: Tuple[float, float] = PACE,
              lo_t: float = 1.0, hi_t: float = 2.0) -> Dict[str, Any]:
    """Pure: the atempo that makes the whole video ≈ aim seconds (voice/tempo + pauses + lead-in + tail), clipped so that
    the pace words/total stays inside `pace` and the length inside TARGET_SEC. Returns tempo, expected total and pace."""
    fixed = compose.LEAD_IN + compose.TAIL + pause_sec
    total = words / ((pace[0] + pace[1]) / 2)                                    # the middle of the pace range (≈ 3.15 w/s)
    total = min(max(total, TARGET_SEC[0] + 1.0), TARGET_SEC[1] - 1.0)
    tempo = raw_voice_sec / max(1.0, total - fixed)
    tempo_c = min(max(tempo, lo_t), hi_t)
    exp_total = fixed + raw_voice_sec / tempo_c
    return {"tempo": round(tempo_c, 3), "wanted_tempo": round(tempo, 3), "expected_total": round(exp_total, 2),
            "expected_pace": round(words / exp_total, 3),
            "ok": TARGET_SEC[0] <= exp_total <= TARGET_SEC[1] and pace[0] <= words / exp_total <= pace[1]}


# ── 5. shots: storyfx.plan_shots_v2 (faces + motion, rhythm by position) ──────────────────────────────────

def canvas_geometry(aspect: str) -> Dict[str, int]:
    """The output frame IS the cropped picture (1:1 → 1080×1080, 4:5 → 1080×1350, 3:4 → 1080×1440), no bars."""
    from . import storyfx
    w, h = storyfx.canvas_size(aspect)
    return {"w": w, "h": h, "x": 0, "y": 0, "bottom": h}


# ── 6. subtitles ────────────────────────────────────────────────────────────

def _sub_font(size: int):
    from PIL import ImageFont
    from scripts.shared.subtitle_styles import FONTS_DIR
    return ImageFont.truetype(str(Path(FONTS_DIR) / SUB_FONT), size)


def sub_width(text: str, size: int = SUB_SIZE) -> float:
    return float(_sub_font(size).getlength(text))


def wrap_lines(words: Sequence[str], max_w: float = SUB_MAX_W, size: int = SUB_SIZE, max_lines: int = SUB_LINES) -> List[str]:
    """1 line when it fits, else 2 balanced lines (never a break right after a 1–2 letter word); each line ≤ max_w px."""
    full = " ".join(words)
    if sub_width(full, size) <= max_w or len(words) < 2 or max_lines < 2:
        return [full]
    best, best_cost = None, 1e9
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        wa, wb = sub_width(a, size), sub_width(b, size)
        cost = max(wa, wb) + (400 if len(re.sub(r"\W", "", words[k - 1])) <= 2 and not re.search(r"[.!?…,;:]$", words[k - 1]) else 0)
        cost += 0 if max(wa, wb) <= max_w else 5000
        if cost < best_cost:
            best, best_cost = [a, b], cost
    return best or [full]


def cue_fits(words: Sequence[str], max_w: float = SUB_MAX_W, size: int = SUB_SIZE, max_lines: int = SUB_LINES) -> bool:
    lines = wrap_lines(words, max_w, size, max_lines)
    return len(lines) <= max_lines and all(sub_width(x, size) <= max_w for x in lines)


def sub_cues(words: Sequence[Tuple[str, float, float]], max_w: float = SUB_MAX_W, size: int = SUB_SIZE, max_words: int = 6,
             min_words: int = 3, max_sec: float = 3.0, max_lines: int = SUB_LINES) -> List[List[Tuple[str, float, float]]]:
    """Groups of 3–6 words, ≤ max_lines lines of ≤ max_w px (measured with the font), ≤ max_sec, broken at sentence ends / commas;
    a cue never ends on «и / в / с / к / не…»."""
    groups: List[List[Tuple[str, float, float]]] = []
    cur: List[Tuple[str, float, float]] = []
    for w in words:
        trial = cur + [w]
        too_wide = not cue_fits([x[0] for x in trial], max_w, size, max_lines)
        too_long = len(trial) > max_words or (cur and trial[-1][2] - trial[0][1] > max_sec)
        if cur and (too_wide or too_long):
            carry: List[Tuple[str, float, float]] = []
            while len(cur) > 2 and len(re.sub(r"\W", "", cur[-1][0])) <= 2 and not re.search(r"[.!?…,;:]$", cur[-1][0]):
                carry.insert(0, cur.pop())
            groups.append(cur)
            cur = carry + [w]
        else:
            cur = trial
        if re.search(r"[.!?…]$", cur[-1][0]) or (len(cur) >= min_words + 1 and cur[-1][0].endswith(",")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    merged: List[List[Tuple[str, float, float]]] = []
    for g in groups:                                                 # a 1–2 word tail joins the previous cue if it fits
        if merged and len(g) < min_words and len(merged[-1]) + len(g) <= max_words + 1 \
                and not re.search(r"[.!?…]$", merged[-1][-1][0]) \
                and cue_fits([x[0] for x in merged[-1] + g], max_w, size, max_lines):
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    return merged


def story_ass(words: Sequence[Tuple[str, float, float]], total: float, geo: Dict[str, int], size: int = SUB_SIZE) -> str:
    """Bold white subtitles (black outline + soft shadow, 1–2 lines) ON the picture, in the lower-middle where the samples have
    them; a cue lives from its first word to the first word of the next cue."""
    H = geo["h"]
    margin_v = int(round(H * SUB_TOP))
    head = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {geo['w']}", f"PlayResY: {H}", "WrapStyle: 2", "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
            "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Story,{SUB_FONT_NAME},{size},&H00FFFFFF,&H000000FF,&H00000000,&H78000000,0,0,0,0,100,100,0,0,1,{SUB_OUTLINE},2,8,"
            f"{(geo['w'] - int(SUB_MAX_W)) // 2},{(geo['w'] - int(SUB_MAX_W)) // 2},{margin_v},204",
            "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    cl = sub_cues(words, size=size)
    ev = []
    for i, g in enumerate(cl):
        s = g[0][1]
        e = min(total, cl[i + 1][0][1] if i + 1 < len(cl) else g[-1][2] + 0.35)
        e = min(e, g[-1][2] + 0.6)
        if e - s < 0.05:
            continue
        lines = wrap_lines([x[0].replace("\\", "").replace("{", "(").replace("}", ")") for x in g], SUB_MAX_W, size)
        nl = chr(92) + "N"
        ev.append(f"Dialogue: 0,{lite_t(s)},{lite_t(e)},Story,,0,0,0,,{nl.join(lines)}")
    return "\n".join(head + ev) + "\n"


def lite_t(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def check_story_ass(text: str, geo: Dict[str, int], size: int = SUB_SIZE) -> Dict[str, Any]:
    """Deterministic: every cue is 1–2 lines, 1–7 words, each line ≤ SUB_MAX_W px (real font, outline added), the whole block lies
    inside the frame (top ≥ 4 %, bottom ≤ 94 % of the height), nothing leaves the left/right edge."""
    problems, worst, counts, nl = [], 0.0, [], []
    H, W = geo["h"], geo["w"]
    top = H * SUB_TOP
    line_h = size * 1.3
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        body = line.split(",", 9)[9]
        lines = body.split("\\N")
        nl.append(len(lines))
        counts.append(len(body.replace("\\N", " ").split()))
        for ln in lines:
            px = sub_width(ln, size) + 2 * SUB_OUTLINE
            worst = max(worst, px)
            if px > SUB_MAX_W + 2 * SUB_OUTLINE + 0.5 or px > W - 2:
                problems.append(f"{px:.0f}px > {SUB_MAX_W:.0f}: {ln[:40]!r}")
        if len(lines) > SUB_LINES:
            problems.append(f"{len(lines)} lines: {body[:40]!r}")
        if counts[-1] > 7:
            problems.append(f"{counts[-1]} words: {body[:40]!r}")
        if top + len(lines) * line_h > 0.94 * H:
            problems.append(f"the block leaves the frame: {body[:40]!r}")
    if top < 0.04 * H:
        problems.append("subtitle block starts above the frame")
    return {"name": "subtitles_in_frame", "ok": not problems, "value": {"worst_px": round(worst, 1), "max_px": SUB_MAX_W, "cues": len(counts),
            "avg_words": round(sum(counts) / max(1, len(counts)), 2), "two_line_cues": sum(1 for n in nl if n == 2), "size": size,
            "block_top_pct": round(100 * SUB_TOP), "block_bottom_px_max": round(top + max(nl or [1]) * line_h), "frame_h": H},
            "detail": "; ".join(problems[:5])}


def check_timeline_story(final: str, tl: Dict[str, Any], dur: float, geo: Dict[str, int], tol: float = 0.12,
                         threshold: float = 0.12) -> Dict[str, Any]:
    """validator.check_timeline_match, but the cut detector looks only at the picture rectangle: on a letterboxed frame
    the whole-frame scene score is diluted by the black bars and finds nothing."""
    from . import validator as V
    r = subprocess.run([media.ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", final, "-an", "-vf",
                        f"crop={geo['w']}:{geo['h']}:{geo['x']}:{geo['y']},scale=360:-2,select='gt(scene,{threshold})',metadata=print:file=-",
                        "-f", "null", "-"], capture_output=True, text=True)
    scored = V.parse_scene_scores(r.stdout)
    found = [t for t, _ in scored]
    shots = tl["shots"]
    planned = [x["t0"] for x in shots[1:]]
    hit = [p for p in planned if any(abs(p - f) <= tol for f in found)]
    unplanned = [t for t, sc in scored if sc >= V.HARD_CUT and not any(abs(p - t) <= tol for p in planned)]
    problems = []
    if abs(shots[-1]["t1"] - dur) > 2 / 30 + 0.05:
        problems.append(f"timeline {shots[-1]['t1']:.2f}s vs video {dur:.2f}s")
    if planned and len(hit) / len(planned) < 0.85:
        problems.append(f"only {len(hit)}/{len(planned)} planned cuts found in the picture")
    if len(unplanned) > 2:
        problems.append(f"{len(unplanned)} unplanned hard cut(s) at {[round(u, 2) for u in unplanned[:5]]}")
    return {"name": "timeline_match", "ok": not problems, "value": {"planned_cuts": len(planned), "matched": len(hit),
            "missing": [round(p, 2) for p in planned if p not in hit][:12], "unplanned": [round(u, 2) for u in unplanned[:8]],
            "detector": "picture rectangle only"}, "detail": "; ".join(problems)}


# ── 7. the run ──────────────────────────────────────────────────────────────

ACCEPT_WORDS = (250, 345)
MAX_OUT_STORY = 4200
MAX_OUT_PICK = 300
MAX_OUT_FIX = 3600
DIGEST_CHARS_STORY = 9000
EVENT_SEC_STORY = 20.0


def run_story(input_path: str, *, run_dir: Optional[str] = None, n_frames: int = 0, stop_rub: float = STORY_STOP_RUB,
              tts_voice: str = "aidar", tts_provider: str = "silero", language: str = "ru", forbid_names: Sequence[str] = (),
              window: Optional[Tuple[float, float]] = None, bed_db: float = BED_DB, day_limit_usd: Optional[float] = None,
              chat=None, tts=None, script_file: Optional[str] = None, pick: Optional[str] = None, crop: str = "1:1",
              sfx_dir: Optional[str] = None, fix_notes: Optional[str] = None) -> Dict[str, Any]:
    """window = a fixed episode (skips the picker call); pick = 'C3' (skips the picker call, uses the local candidate)."""
    import hashlib
    import os
    from . import recap_runner as rr, storyfx, validator
    from .runner import ROOT, _desktop_media_env, _dump, _load_llm_env, _sha256, run_asr
    t_all = time.time()
    out = Path(run_dir or (ROOT.parent / "runtime" / "trailer" / f"story-{time.strftime('%H%M%S')}")).resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT.parent / "runtime" / "trailer" / "_cache"
    M: Dict[str, Any] = {"mode": "trailer_cinematic", "style": "story", "version": STORY_VERSION, "run_dir": str(out),
                         "source_path": str(Path(input_path).resolve()), "stop_rub": stop_rub, "crop": crop, "warnings": [], "timings_sec": {}}
    T = M["timings_sec"]

    def say(m: str) -> None:
        print(f"[story {time.time() - t_all:6.1f}s] {m}", flush=True)

    def save() -> None:
        T["total"] = round(time.time() - t_all, 2)
        _dump(out / "metrics.json", M)

    M["media_env"] = _desktop_media_env()
    info = media.probe(input_path)
    sha = _sha256(input_path)
    M["source_sha256"] = sha
    asr = run_asr(input_path, cache_dir, sha, language)                       # cached → 0 ₽
    raw_words = asr.get("words") or []
    M["asr"] = {k: asr["meta"].get(k) for k in ("backend", "model", "words", "cached")}
    film_sec = min(float(info.get("duration") or 1e9), (raw_words[-1]["end"] + 5.0) if raw_words else 0.0)
    t0 = time.time()
    bad = damaged_times(input_path, 0.0, film_sec, cache=cache_dir / f"damage-{sha[:16]}.json")
    T["damage_scan"] = round(time.time() - t0, 2)
    M["damaged_probe_times"] = bad
    cands = candidates(raw_words, film_sec, skip_tail=min(600.0, film_sec * 0.03), bad=bad)
    M["candidates"] = cands
    say("candidates: " + "; ".join(f"{c['id']} {lite.mmss(c['a'])}–{lite.mmss(c['b'])} score {c['score']}" for c in cands))

    _load_llm_env()
    from scripts.shared.llm_provider import certifi_transport, chat_json
    kind, tw = rr.select_provider("timeweb") if chat is None else ("fake", None)
    if day_limit_usd is None:
        day_limit_usd = float(os.environ.get("SONYA_TRAILER_LLM_DAY_USD") or budget.DEFAULT_DAY_USD)       # never raised here
    ledgers, chat_kwargs = None, None
    if tw is not None:
        chat_kwargs = dict(providers=[tw], retries=2, max_request_bytes=rr.MAX_REQUEST_BYTES, transport=certifi_transport())
        day = budget.DailyLedger(str(cache_dir / "llm_spend_timeweb.json"), day_limit_usd)
        vled = budget.VideoLedger(str(cache_dir / f"spend_story2-{sha[:16]}.json"), stop_rub)
        ledgers = [day, vled]
        M["day_limit_usd"], M["day_spent_before_usd"] = day_limit_usd, round(day.spent(), 5)
    model = tw.model if tw else "fake"
    M["llm_provider"] = f"{tw.name}/{model}" if tw else "fake"
    llm = budget.LLM(chat or chat_json, stop_rub / budget.RUB_PER_USD, model, cache_dir=str(cache_dir / "llm"),
                     ledger=ledgers, chat_kwargs=chat_kwargs, reasoning=tw is None)

    def spent_rub() -> float:
        vl = [x for x in llm.ledgers if isinstance(x, budget.VideoLedger)]
        return budget.rub(vl[0].spent()) if vl else budget.rub(llm.spent)

    def pre_gate(msgs, label, max_out):
        fc = lite.gate(msgs, model, spent_rub(), stop_rub, max_out)
        M.setdefault("forecasts", []).append(dict(fc, step=label))
        say(f"forecast {label}: ≤ {fc['rub']:.2f} ₽ (spent {fc['spent_rub']:.2f} of {stop_rub:.1f} ₽)")

    try:
        # 1. which episode (one small text-only call, or fixed by the caller)
        t0 = time.time()
        why = ""
        if window:
            ep = {"id": "W", "a": window[0], "b": window[1]}
        elif pick:
            ep = next(c for c in cands if c["id"] == pick)
        else:
            msgs = pick_messages(raw_words, cands)
            pre_gate(msgs, "pick_episode", MAX_OUT_PICK + 600)
            data = llm.ask("story_pick", msgs, f"story-pick-{STORY_VERSION}", lambda d: (parse_pick(d, cands), d)[1],
                           temperature=0.3, max_out=MAX_OUT_PICK + 600, effort="low")
            _dump(out / "model_pick.json", data)
            ep, why = parse_pick(data, cands)
        a, b = float(ep["a"]), float(ep["b"])
        M["episode"] = {"id": ep["id"], "start": a, "end": b, "start_mmss": lite.mmss(a), "end_mmss": lite.mmss(b),
                        "minutes": round((b - a) / 60, 1), "why": why}
        say(f"episode {ep['id']}: {lite.mmss(a)}–{lite.mmss(b)} ({(b - a) / 60:.1f} min)")

        # 2. digest + frames (local)
        events = lite.digest(raw_words, a, b, target_chars=DIGEST_CHARS_STORY, event_sec=EVENT_SEC_STORY)
        dtext = lite.digest_text(events, a)
        frames: List[Dict[str, Any]] = []                                   # v2: no stills to the model (shots are chosen locally)
        (out / "digest.txt").write_text(dtext, encoding="utf-8")
        M["digest"] = {"events": len(events), "chars": len(dtext), "frames": 0}
        say(f"digest {len(events)} events / {len(dtext)} chars; text-only writer call")
        window_text = " ".join(w.get("word", "") for w in raw_words if a <= w.get("start", 0) < b)
        crime_ok = recap.crime_stems_in(window_text)
        forbid = recap.forbidden_names({}, forbid_names)
        lo_w, hi_w = 300, 340
        tick_llm = time.time()

        # 3. ONE Story+Writer call (+ at most one text-only fix)
        if script_file:
            raw = json.loads(Path(script_file).read_text(encoding="utf-8"))
            data = raw.get("script", raw)
            M["warnings"].append(f"script taken from {script_file}, not from the model")
            if fix_notes:                                                  # a reviewer's fact-check of that script → ONE text-only fix
                notes = [x.strip() for x in Path(fix_notes).read_text(encoding="utf-8").splitlines() if x.strip()]
                forbid = recap.forbidden_names({"characters": data.get("characters") or []}, forbid_names)
                defects = notes + story_defects(data, events, frames, lo_w, hi_w, forbid, crime_ok)
                M["defects_found"] = defects
                say(f"fix with {len(notes)} reviewer note(s) → ONE more text-only call")

                def shape(d):
                    if not isinstance(d, dict) or not isinstance(d.get("segments"), list):
                        raise SchemaError("JSON with segments[] expected")
                    return d
                fm = fix_messages(events, a, {"segments": data["segments"]}, defects, lo_w, hi_w)
                pre_gate(fm, "text_fix_notes", MAX_OUT_FIX)
                data = llm.ask("story_fix_notes", fm, f"story-writer-{STORY_VERSION}+fixnotes", shape, temperature=0.4, max_out=MAX_OUT_FIX, effort="medium")
                _dump(out / "model_answer_3.json", data)
                forbid = recap.forbidden_names({"characters": data.get("characters") or []}, forbid_names)
        else:
            msgs = story_messages(events, frames, a, b, lo_w, hi_w, forbid, why)
            pre_gate(msgs, "story_writer", MAX_OUT_STORY)

            def shape(d):
                if not isinstance(d, dict) or not isinstance(d.get("segments"), list):
                    raise SchemaError("JSON with segments[] expected")
                return d
            data = llm.ask("story_writer", msgs, f"story-writer-{STORY_VERSION}", shape, temperature=0.7, max_out=MAX_OUT_STORY, effort="medium")
            _dump(out / "model_answer_1.json", data)
            forbid = recap.forbidden_names({"characters": data.get("characters") or []}, forbid_names)
            defects = story_defects(data, events, frames, lo_w, hi_w, forbid, crime_ok)
            if defects:
                M["defects_found"] = defects
                say(f"free checks found {len(defects)} defect(s) → ONE text-only fix")
                fm = fix_messages(events, a, {"segments": data["segments"]}, defects, lo_w, hi_w)
                pre_gate(fm, "text_fix", MAX_OUT_FIX)
                data = llm.ask("story_fix", fm, f"story-writer-{STORY_VERSION}+fix", shape, temperature=0.4, max_out=MAX_OUT_FIX, effort="medium")
                _dump(out / "model_answer_2.json", data)
                forbid = recap.forbidden_names({"characters": data.get("characters") or []}, forbid_names)
        # what stops the run is the SPEC (85–105 s at 2.9–3.4 w/s ⇒ ≈ 250–345 words, the tempo is fitted afterwards), not the stricter
        # target that was asked of the model
        left = story_defects(data, events, frames, ACCEPT_WORDS[0], ACCEPT_WORDS[1], forbid, crime_ok, max_sent=22)
        M["defects_left"] = left
        if left:
            M["warnings"].append(f"after the model's last answer the free checks still say: {left}")
        script = clean_script(data, events, frames)
        T["llm"] = round(time.time() - tick_llm, 2)
        words = sum(n_words(s["text"]) for s in script["segments"])
        _dump(out / "script.json", {"version": STORY_VERSION, "style": "story", "script": script, "words": words})
        say(f"script: {words} words in {len(script['segments'])} segments; defects left: {len(left)}")
        if left and (not script_file or fix_notes):
            raise RuntimeError("script still has defects after the one allowed fix — stopped before TTS/render: " + "; ".join(left[:4]))

        # 4. TTS: raw at tempo 1.0 → tempo fitted to the text → sentence wavs
        t0 = time.time()
        if tts is None:
            from scripts.shared.tts import get_provider
            tts = get_provider(tts_provider, tts_voice if tts_provider == "silero" else None)
        if not tts.available():
            raise RuntimeError(f"TTS provider {tts.name} is not available")
        from scripts.shared.tts import wav_duration
        raw_wavs: Dict[Tuple[str, int], str] = {}
        raw_sec: Dict[str, List[float]] = {}
        for seg in script["segments"]:
            raw_sec[seg["id"]] = []
            for j, sent in enumerate(recap.sentences_of(seg["text"])):
                h = hashlib.sha256(f"{tts.name}|{tts.voice}|{sent}".encode()).hexdigest()[:20]
                p = cache_dir / "tts" / f"{h}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.is_file():
                    tts.synthesize(sent, str(p))
                raw_wavs[(seg["id"], j)] = str(p)
                raw_sec[seg["id"]].append(wav_duration(str(p)))
        raw_total = sum(sum(v) for v in raw_sec.values())
        ft = fit_tempo(raw_total * 0.97, pause_total(script), words)          # 0.97: atempo edge trims shave a little
        tempo = ft["tempo"]
        M["tts"] = dict(tts.info(), tempo=tempo, fit=ft, raw_voice_sec=round(raw_total, 2))
        say(f"voice raw {raw_total:.1f}s → tempo {tempo} (expected total {ft['expected_total']}s, pace {ft['expected_pace']})")
        def speak(tp: float):
            wv, sc = {}, {}
            for seg in script["segments"]:
                sc[seg["id"]] = []
                for j, _ in enumerate(recap.sentences_of(seg["text"])):
                    p = Path(raw_wavs[(seg["id"], j)])
                    q = Path(rr.speed_up(str(p), str(p.with_suffix(f".x{tp:.3f}e.wav")), tp)) if abs(tp - 1.0) > 1e-3 else p
                    wv[(seg["id"], j)] = str(q)
                    sc[seg["id"]].append(wav_duration(str(q)))
            return wv, sc, compose.voice_timeline(script, sc, pauses=PAUSES)
        wavs, secs, vt = speak(tempo)
        if abs(vt["total"] - ft["expected_total"]) > 1.5 and abs(tempo - 1.0) > 1e-3:       # atempo edge trims: one correction
            real_voice = sum(sum(v) for v in secs.values())
            want_voice = max(1.0, real_voice - (vt["total"] - ft["expected_total"]))
            tempo2 = round(min(max(tempo * real_voice / want_voice, 1.0), 2.0), 3)
            if abs(tempo2 - tempo) > 0.01:
                tempo = tempo2
                M["tts"]["tempo"] = tempo
                wavs, secs, vt = speak(tempo)
                say(f"tempo corrected to {tempo}")
        T["tts"] = round(time.time() - t0, 2)
        say(f"voice timeline total {vt['total']:.1f}s, pace {words / vt['total']:.2f} w/s")

        # 5. shots: cropped to the chosen aspect around the faces; big faces / motion; rhythm by position
        t0 = time.time()
        sha_short = sha

        def cuts_in(lo, hi):
            return sorted(set(media.frame_cuts(input_path, lo, hi, str(cache_dir / "clips"), sha_short))
                          | set(validator.hard_cuts_in(input_path, lo, hi)))
        scan_data = storyfx.scan_episode(input_path, max(0.0, a - 2.0), b + 2.0,
                                         cache_dir / f"story-scan-{sha[:16]}-{int(a)}-{int(b)}.json")
        scan = storyfx.Scan(scan_data)
        T["scan"] = round(time.time() - t0, 2)
        cw, ch = storyfx.crop_window(info["width"], info["height"], crop)
        geo = canvas_geometry(crop)
        M["crop_geometry"] = {"aspect": crop, "source": f"{info['width']}x{info['height']}", "window": f"{cw}x{ch}",
                              "canvas": f"{geo['w']}x{geo['h']}", "face_detector": scan_data["meta"]["detector"],
                              "scan_samples": scan_data["meta"]["samples"], "motion_ref": round(scan.ref, 3), "motion_low": round(scan.low, 3)}
        starts = [w.get("start", 0.0) for w in raw_words]

        def speech_in(x0: float, x1: float) -> bool:
            import bisect
            k = bisect.bisect_left(starts, x0 - 0.2)
            return k < len(starts) and starts[k] < x1
        t0 = time.time()
        timeline = storyfx.plan_shots_v2(script, vt, events, a, b, cuts_in, scan, cw, speech_in)
        problems = compose.check_plan(timeline)
        if problems:
            raise RuntimeError(f"invalid timeline: {problems}")
        T["shot_plan"] = round(time.time() - t0, 2)
        t0 = time.time()
        (work / "shots").mkdir(exist_ok=True)
        files, beds = [], []
        for sh in timeline["shots"]:
            fv, fa = str(work / "shots" / f"shot_{sh['n']:02d}.mp4"), str(work / "shots" / f"bed_{sh['n']:02d}.wav")
            storyfx.render_shot_v2(input_path, sh, fv, fa, info["width"], cw, ch, geo["w"], geo["h"])
            files.append(fv)
            beds.append(fa)
        T["render_shots"] = round(time.time() - t0, 2)

        # 6. voice, subtitles, variant A (voice only), variant B (+ film sound + accents)
        t0 = time.time()
        voice = str(work / "voice.wav")
        compose.write_voice(vt, wavs, voice)
        stab = compose.stability(vt, voice)
        g = media.voice_gain_db(voice)
        sw, _ = lite.censor.subtitle_words(vt)
        ass = str(out / "narration.ass")
        Path(ass).write_text(story_ass(sw, vt["total"], geo), encoding="utf-8")
        final_a = str(out / "final_A.mp4")
        media.assemble_recap(files, voice, g["gain_db"], vt["total"], ass, final_a,
                             shot_frames=[round((s["t1"] - s["t0"]) * 30) for s in timeline["shots"]])
        final_b = str(out / "final_B.mp4")
        accents = [] if sfx_dir == "" else storyfx.plan_accents(timeline, script, vt, sfx_dir or storyfx.SFX_DEFAULT_DIR)
        if not accents:
            M["warnings"].append("no sound accents (sound pack folder missing or --no-sfx)")
        M["bed"] = mix_bed(final_a, voice, g["gain_db"], beds, [bool(s["film_speech"]) for s in timeline["shots"]], vt["total"],
                           final_b, work, bed_db, accents)
        M["accents"] = [{k: x[k] for k in ("kind", "tag", "file", "at", "start_s", "peak_s", "gain_db")} for x in accents]
        T["render_assemble"] = round(time.time() - t0, 2)

        # 7. validation
        t0 = time.time()
        text = " ".join(s["text"] for s in script["segments"])
        val = validator.validate(final_a, tl=timeline, vt=vt, voice_wav=voice, gain_db=g["gain_db"], shot_files=files, source=input_path,
                                 script_text=text, spread_db=stab["spread_db"], asr=rr.asr_text_fn(language, work),
                                 dur_range=TARGET_SEC, target=TARGET_AIM, ass_path=None, pace=PACE, forbid_names=forbid)
        val["checks"] = [c for c in val["checks"] if c["name"] not in ("timeline_match", "params")]
        pr = validator.check_params(final_a)                     # the generic check wants 1080×1920: re-judge it for this canvas
        size_ok = pr["value"].get("size") == f"{geo['w']}x{geo['h']}"
        rest = [x for x in (pr.get("detail") or "").split(";") if x and not x.strip().startswith("size")]
        val["checks"].append({"name": "params", "ok": size_ok and not rest, "value": pr["value"],
                              "detail": "" if size_ok and not rest else f"want {geo['w']}x{geo['h']}; {pr.get('detail')}"})
        val["checks"].append(check_timeline_story(final_a, timeline, val["duration"], geo))
        val["checks"].append(check_story_ass(Path(ass).read_text(encoding="utf-8"), geo))
        cs = storyfx.cut_stats_v2(timeline)
        val["checks"].append({"name": "cuts_rhythm", "ok": not cs["out_of_range"] and cs["avg_last_third"] < cs["avg_first_third"] and
                              cs["no_face_shots"] <= max(4, len(timeline["shots"]) // 5), "value": cs,
                              "detail": "shots outside 1.15–3.45 s, or the climax is not faster, or too many shots without a face"})
        # variant B: the same picture; loudness, narrator intact, bed level
        b_checks = [validator.check_duration(float(validator.probe_all(final_b)["format"]["duration"]), *TARGET_SEC, target=TARGET_AIM),
                    validator.check_loudness(final_b, stab["spread_db"]),
                    validator.check_words_intact(final_b, vt, text, rr.asr_text_fn(language, work), edge_db=0.0),   # edges carry the film's bed: ASR only
                    {"name": "bed_level", "ok": bed_db + BED_DUCK_DB - 4.0 <= M["bed"]["measured_rel_db"] <= bed_db + 2.0, "value": M["bed"], "detail": ""}]
        val["variant_B_checks"] = b_checks
        val["failed"] = [c["name"] for c in val["checks"] if not c["ok"]]
        val["passed"] = not val["failed"]
        val["variant_B_failed"] = [c["name"] for c in b_checks if not c["ok"]]
        T["validator"] = round(time.time() - t0, 2)
        say(f"validator A: passed={val['passed']} failed={val['failed']}; B failed={val['variant_B_failed']}")
        _dump(out / "validation.json", val)
        _dump(out / "timeline.json", timeline)
        _dump(out / "narration.json", {"tts": M["tts"], "segments": [
            dict(s, text=next(x["text"] for x in script["segments"] if x["id"] == s["id"])) for s in vt["segments"]]})
        items = [{"n": s["n"], "t": (s["t0"] + s["t1"]) / 2, "label": f"{s['segment_id']} #{s['n']}"} for s in timeline["shots"]]
        sheets = []
        for k in range(0, len(items), 12):
            sp = str(out / f"sheet_{k // 12}.jpg")
            media.contact_sheet(final_a, items[k:k + 12], sp)
            sheets.append(sp)
        M.update(final_A=final_a, final_B=final_b, final_duration=val["duration"], technical_passed=val["passed"],
                 technical_failed=val["failed"], variant_B_failed=val["variant_B_failed"], shots=len(timeline["shots"]),
                 cuts=cs, words=words, pace=round(words / val["duration"], 2), contact_sheets=sheets,
                 sizes_mb={"A": round(Path(final_a).stat().st_size / 1e6, 1), "B": round(Path(final_b).stat().st_size / 1e6, 1)})
    finally:
        M["llm"] = llm.summary()
        M["cost_usd"] = llm.spent
        M["cost_rub"] = budget.rub(llm.spent)
        M["calls"] = llm.calls
        save()
    return M


def mix_bed(final_a: str, voice_wav: str, gain_db: float, bed_wavs: Sequence[str], speech: Sequence[bool], total: float,
            out_path: str, work: Path, bed_db: float = BED_DB, accents: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Variant B: the picture of A (stream copy) + narrator (same chain as A) + the film's own sound `bed_db` dB below the narrator
    (integrated loudness; shots where the film's heroes talk are ducked BED_DUCK_DB further) + the quiet accents."""
    from . import storyfx
    bed_plain, bed_raw = str(work / "bed_plain.wav"), str(work / "bed_raw.wav")
    storyfx.build_bed(bed_wavs, [False] * len(bed_wavs), 0.0, bed_plain)
    storyfx.build_bed(bed_wavs, speech, BED_DUCK_DB, bed_raw)
    vm = media.loudness(voice_wav)
    voice_lufs = (vm["integrated_lufs"] if vm["integrated_lufs"] is not None else -16.0) + gain_db
    bm = media.loudness(bed_plain)                                  # bed_db is the level of the shots WITHOUT film speech;
    bed_lufs = bm["integrated_lufs"] if bm["integrated_lufs"] is not None else -30.0     # where the heroes talk it is BED_DUCK_DB lower
    bed_gain = (voice_lufs + bed_db) - bed_lufs
    acc_parts, acc_labels = storyfx.accent_filters(accents, 3, total)
    fc = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,{media.voice_chain(gain_db)},atrim=duration={total:.3f},asetpts=PTS-STARTPTS[v];"
          f"[2:a]aresample=48000,aformat=channel_layouts=stereo,volume={bed_gain:.2f}dB,atrim=duration={total:.3f},apad,atrim=duration={total:.3f},"
          f"asetpts=PTS-STARTPTS,afade=t=out:st={max(0.0, total - 0.6):.3f}:d=0.6[b];")
    fc += "".join(x + ";" for x in acc_parts)
    fc += f"[v][b]{''.join(acc_labels)}amix=inputs={2 + len(acc_labels)}:normalize=0:duration=first,alimiter=limit=0.89:level=disabled[a]"
    extra: List[str] = []
    for x in accents:
        extra += ["-i", x["path"]]
    r = subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(Path(final_a).resolve()),
                        "-i", str(Path(voice_wav).resolve()), "-i", bed_raw, *extra, "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(Path(out_path).resolve())],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"variant B mix failed: {r.stderr[-500:]}")
    scaled = str(work / "bed_scaled.wav")                          # measure what B really carries (not the plan)
    subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", bed_raw, "-af", f"volume={bed_gain:.2f}dB",
                    "-c:a", "pcm_s16le", scaled], capture_output=True, text=True)
    sm = media.loudness(scaled)["integrated_lufs"]
    return {"bed_db_wanted": bed_db, "duck_db_where_film_talks": BED_DUCK_DB, "ducked_shots": sum(1 for x in speech if x),
            "voice_lufs": round(voice_lufs, 2), "bed_lufs_raw": round(bed_lufs, 2), "bed_gain_db": round(bed_gain, 2),
            "bed_lufs_in_mix": sm, "measured_rel_db": round((sm if sm is not None else -99.0) - voice_lufs, 2), "accents": len(accents)}
