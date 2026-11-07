"""
story.py — the LLM half of trailer_cinematic v1 (docs/TRAILER_V1_SPEC.md). Prompts + strict validators.

    Story Analyst  map    (per 15–20 min chunk: transcript with timestamps + scene boundaries,
                           plus a SMALL bounded memory of the story so far)  → new factual events
                   refine (reduce: chunk events → one global story map)
    Director       story map + target + timestamp candidates → beats[]
    Writer         story map + selected beats + target + event ids (no transcript) → phrases[]
    Judge          separate pass, 0–10 scores + rewrite_instructions (one rewrite max — runner)

Every `validate_*` returns a clean structure or raises budget.SchemaError (the LLM wrapper then sends
ONE repair request with the error text). The model sees ONLY the transcript and scene boundaries —
no frames (v1 limitation, reported).
Prompts are SONYA's own text.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .budget import SchemaError

MAP_VERSION = "trailer-analyst-map-v2"
REFINE_VERSION = "trailer-analyst-refine-v2"
DIRECTOR_VERSION = "trailer-director-v1"
WRITER_VERSION = "trailer-writer-v2"
JUDGE_VERSION = "trailer-judge-v2"

EVENT_EMOTIONS = ("calm", "warm", "humor", "tension", "suspense", "dread", "intense", "conflict", "twist", "sad")
STORY_FUNCTIONS = ("hook", "setup", "escalation", "twist", "climax", "tease")
AUDIO_OWNERS = ("narration", "dialogue", "both")
BEAT_EMOTIONS = ("suspense", "intense", "twist", "calm", "dread")
JUDGE_KEYS = ("hook", "coherence", "suspense", "factuality", "emotional_progression", "repetition",
              "spoiler_control", "duration_fit")
REWRITE_THRESHOLDS = {"factuality": 9, "coherence": 8, "hook": 8, "suspense": 7}
MEMORY_CHARS = 3000          # bounded story-so-far memory passed between chunks


def fmt_t(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _num(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise SchemaError(f"{what}: number expected, got {x!r}") from None
    if v != v:
        raise SchemaError(f"{what}: NaN")
    return v


def _str(x: Any, what: str, max_len: int = 400, required: bool = True) -> str:
    if not isinstance(x, str) or (required and not x.strip()):
        raise SchemaError(f"{what}: non-empty string expected")
    return re.sub(r"\s+", " ", x).strip()[:max_len]


def _str_list(x: Any, what: str, max_items: int = 20, max_len: int = 200) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise SchemaError(f"{what}: list expected")
    return [re.sub(r"\s+", " ", str(v)).strip()[:max_len] for v in x if str(v).strip()][:max_items]


def _need(d: Any, keys: Sequence[str], what: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        raise SchemaError(f"{what}: JSON object expected")
    miss = [k for k in keys if k not in d]
    if miss:
        raise SchemaError(f"{what}: missing keys {miss}")
    return d


# ── 1. Story Analyst: map ───────────────────────────────────────────────────

ANALYST_SYSTEM = (
    "Ты — аналитик сюжета для монтажёра трейлеров. Длинное видео приходит частями: расшифровка речи с "
    "таймкодами (секунды от начала исходника) и границы монтажных планов. Кадров ты НЕ видишь — только "
    "текст и границы сцен, поэтому не описывай внешность, место и действия, которых нет в речи. "
    "Твоя работа — факты: кто участвует, что происходит, какой конфликт, что меняется. Ничего не "
    "выдумывай: у каждого события есть evidence — дословная короткая цитата из расшифровки. "
    "Отвечай только JSON.")

EVENT_SCHEMA = ('{"id": "E<часть>.<n>", "start": сек, "end": сек, "summary": "что происходит, 1–2 предложения, '
                'только факты", "participants": ["имя или роль"], "importance": 1..5, '
                '"evidence": ["дословная цитата из расшифровки (≤ 15 слов)"], '
                '"emotion": "' + "|".join(EVENT_EMOTIONS) + '"}')


def transcript_lines(sentences: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in sentences) or "(речи нет)"


def scene_line(shots: Sequence[Dict[str, Any]]) -> str:
    return " ".join(f"{c['start']:.1f}" for c in shots[1:]) or "(нет склеек)"


def compact_memory(memory: Dict[str, Any], max_chars: int = MEMORY_CHARS) -> Dict[str, Any]:
    """Bounded story-so-far: premise, conflict, characters, last event summaries — never the transcript."""
    m = {"premise": str(memory.get("premise") or "")[:400], "conflict": str(memory.get("conflict") or "")[:400],
         "characters": [str(c)[:120] for c in (memory.get("characters") or [])][:12],
         "open_questions": [str(q)[:160] for q in (memory.get("open_questions") or [])][:6],
         "recent_events": [str(e)[:200] for e in (memory.get("recent_events") or [])][-12:]}
    while len(json.dumps(m, ensure_ascii=False)) > max_chars and m["recent_events"]:
        m["recent_events"].pop(0)
    return m


def map_messages(chunk: Dict[str, Any], memory: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    task = (
        f"ЧАСТЬ {chunk['index'] + 1} из {chunk['total']}: {fmt_t(chunk['start'])}–{fmt_t(chunk['end'])} "
        f"(секунды {chunk['start']:.0f}–{chunk['end']:.0f}). Язык речи: {language}.\n\n"
        "ЧТО УЖЕ ИЗВЕСТНО ИЗ ПРЕДЫДУЩИХ ЧАСТЕЙ (краткая память; не повторяй эти события):\n"
        + json.dumps(compact_memory(memory), ensure_ascii=False) +
        "\n\nРАСШИФРОВКА [начало-конец, сек]:\n" + transcript_lines(chunk["sentences"]) +
        "\n\nСКЛЕЙКИ ПЛАНОВ (сек): " + scene_line(chunk["shots"]) +
        "\n\nВерни JSON:\n{\"characters\": [{\"name\": \"имя или роль\", \"description\": \"кто это, чего хочет\"}], "
        "\"premise\": \"о чём история (с учётом памяти)\", \"conflict\": \"главный конфликт на сейчас\", "
        "\"events\": [" + EVENT_SCHEMA + "], \"open_questions\": [\"что пока не решено\"]}\n"
        "Только НОВЫЕ события этой части, 6–14 штук, значимые для истории (не каждую реплику). "
        "start/end — из таймкодов расшифровки, внутри этой части. Заставку, титры и рекламу событиями не считай.")
    return [{"role": "system", "content": ANALYST_SYSTEM}, {"role": "user", "content": task}]


def _event(e: Any, lo: float, hi: float, what: str) -> Dict[str, Any]:
    _need(e, ("id", "start", "end", "summary", "participants", "importance", "evidence", "emotion"), what)
    s, en = _num(e["start"], what + ".start"), _num(e["end"], what + ".end")
    if en < s:
        s, en = en, s
    if en <= lo - 5 or s >= hi + 5:
        raise SchemaError(f"{what}: {s:.0f}-{en:.0f} outside the chunk {lo:.0f}-{hi:.0f}")
    s, en = max(lo, s), min(hi, en)
    if en - s < 0.5:
        raise SchemaError(f"{what}: shorter than 0.5 s")
    try:
        imp = int(round(float(e["importance"])))
    except (TypeError, ValueError):
        raise SchemaError(f"{what}.importance: 1..5 expected") from None
    emo = e["emotion"] if e["emotion"] in EVENT_EMOTIONS else None
    if emo is None:
        raise SchemaError(f"{what}.emotion: one of {EVENT_EMOTIONS}")
    ev = _str_list(e["evidence"], what + ".evidence", 3, 200)
    if not ev:
        raise SchemaError(f"{what}.evidence: at least one quote")
    return {"id": _str(e["id"], what + ".id", 16), "start": round(s, 2), "end": round(en, 2),
            "summary": _str(e["summary"], what + ".summary", 400),
            "participants": _str_list(e["participants"], what + ".participants", 6, 60),
            "importance": max(1, min(5, imp)), "evidence": ev, "emotion": emo}


def validate_map(data: Any, chunk: Dict[str, Any]) -> Dict[str, Any]:
    _need(data, ("characters", "premise", "conflict", "events", "open_questions"), "map")
    if not isinstance(data["events"], list) or not data["events"]:
        raise SchemaError("map.events: non-empty list expected")
    evs = [_event(e, chunk["start"], chunk["end"], f"events[{i}]") for i, e in enumerate(data["events"])]
    for i, e in enumerate(evs):                       # chunk-local ids, unique
        e["id"] = f"E{chunk['index'] + 1}.{i + 1}"
    chars = []
    for c in data["characters"] or []:
        if isinstance(c, dict) and str(c.get("name") or "").strip():
            chars.append({"name": str(c["name"]).strip()[:60], "description": str(c.get("description") or "")[:200]})
    return {"characters": chars[:15], "premise": _str(data["premise"], "map.premise", 600, False),
            "conflict": _str(data["conflict"], "map.conflict", 600, False), "events": sorted(evs, key=lambda e: e["start"]),
            "open_questions": _str_list(data["open_questions"], "map.open_questions", 8, 200)}


def update_memory(memory: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    names = {c.split(" — ")[0] for c in memory.get("characters") or []}
    chars = list(memory.get("characters") or [])
    for c in part["characters"]:
        if c["name"] not in names:
            chars.append(f"{c['name']} — {c['description'][:80]}")
    return compact_memory({"premise": part["premise"] or memory.get("premise"),
                           "conflict": part["conflict"] or memory.get("conflict"), "characters": chars,
                           "open_questions": part["open_questions"],
                           "recent_events": list(memory.get("recent_events") or [])
                           + [f"{fmt_t(e['start'])} {e['summary'][:140]}" for e in part["events"]]})


# ── 1b. Story Analyst: refine (reduce) ──────────────────────────────────────

REFINE_SYSTEM = (
    "Ты — аналитик сюжета. Перед тобой факты обо всём видео, собранные по частям. Собери из них одну "
    "глобальную карту истории. Новых событий не придумывай: используй только переданные id событий. "
    "Отвечай только JSON.")


def refine_messages(parts: List[Dict[str, Any]], duration: float, language: str) -> List[Dict[str, Any]]:
    events = [{k: e[k] for k in ("id", "start", "end", "summary", "participants", "importance", "emotion")}
              for p in parts for e in p["events"]]
    chars = {}
    for p in parts:
        for c in p["characters"]:
            chars.setdefault(c["name"], c["description"])
    task = (
        f"Длина исходника: {fmt_t(duration)}. Частей: {len(parts)}. Язык: {language}.\n\n"
        "ПЕРСОНАЖИ ИЗ ЧАСТЕЙ:\n" + json.dumps(chars, ensure_ascii=False) +
        "\n\nПОСЫЛКИ И КОНФЛИКТЫ ПО ЧАСТЯМ:\n" + json.dumps([{"premise": p["premise"], "conflict": p["conflict"]}
                                                          for p in parts], ensure_ascii=False) +
        "\n\nВСЕ СОБЫТИЯ ПО ПОРЯДКУ:\n" + json.dumps(events, ensure_ascii=False) +
        "\n\nВОПРОСЫ ИЗ ЧАСТЕЙ:\n" + json.dumps([q for p in parts for q in p["open_questions"]], ensure_ascii=False) +
        "\n\nВерни JSON:\n{\"characters\": [{\"name\": str, \"description\": \"кто это и чего хочет\"}], "
        "\"premise\": \"о чём вся история, 1–2 предложения\", \"conflict\": \"главный конфликт\", "
        "\"event_ids\": [id значимых для истории событий, по порядку; лишнее и повторы убери], "
        "\"open_questions\": [\"что остаётся нерешённым на конец материала\"], "
        "\"ending\": \"чем заканчивается материал (только для проверки фактов, в трейлер не пойдёт)\", "
        "\"spoiler_sensitive_events\": [id событий, которые раскрывают развязку или главный поворот]}")
    return [{"role": "system", "content": REFINE_SYSTEM}, {"role": "user", "content": task}]


def validate_story_map(data: Any, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    _need(data, ("characters", "premise", "conflict", "event_ids", "open_questions", "ending",
                 "spoiler_sensitive_events"), "story_map")
    by_id = {e["id"]: e for p in parts for e in p["events"]}
    ids = [i for i in _str_list(data["event_ids"], "event_ids", 200, 16) if i in by_id]
    if len(ids) < 3:
        raise SchemaError("story_map.event_ids: at least 3 known event ids expected")
    ids = sorted(dict.fromkeys(ids), key=lambda i: by_id[i]["start"])
    chars = [{"name": str(c.get("name"))[:60], "description": str(c.get("description") or "")[:200]}
             for c in data["characters"] or [] if isinstance(c, dict) and c.get("name")]
    return {"characters": chars[:15], "premise": _str(data["premise"], "premise", 600),
            "conflict": _str(data["conflict"], "conflict", 600),
            "events": [by_id[i] for i in ids],
            "open_questions": _str_list(data["open_questions"], "open_questions", 8, 200),
            "ending": _str(data["ending"], "ending", 600, False),
            "spoiler_sensitive_events": [i for i in _str_list(data["spoiler_sensitive_events"],
                                                              "spoiler_sensitive_events", 30, 16) if i in by_id]}


# ── 2. Director ─────────────────────────────────────────────────────────────

DIRECTOR_SYSTEM = (
    "Ты — режиссёр монтажа вертикальных кинотрейлеров. Перед тобой карта истории и кандидаты по таймкодам. "
    "Собери последовательность beats — одну связную линию: начало цепляет, середина усиливает конфликт, "
    "конец оставляет интригу. Финальную развязку не раскрывай. Хронологию можно слегка переставить ради "
    "драматургии, факты менять нельзя. Одну и ту же сцену дважды не бери. Отвечай только JSON.")


def candidates(story_map: Dict[str, Any], sentences: Sequence[Dict[str, Any]], shots: Sequence[Dict[str, Any]],
               per_event: int = 8) -> List[Dict[str, Any]]:
    """Timestamp candidates per event: its sentences (short) and its shot cuts — not the whole transcript."""
    out = []
    for e in story_map["events"]:
        ss = [s for s in sentences if s["end"] > e["start"] and s["start"] < e["end"]]
        if len(ss) > per_event:                       # keep the longest (most complete) lines, in time order
            keep = sorted(sorted(ss, key=lambda s: -(s["end"] - s["start"]))[:per_event], key=lambda s: s["start"])
            ss = keep
        cuts = [c["start"] for c in shots if e["start"] < c["start"] < e["end"]]
        out.append({"event_id": e["id"], "lines": [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text'][:140]}" for s in ss],
                    "cuts": [round(c, 1) for c in cuts[:12]]})
    return out


def director_messages(story_map: Dict[str, Any], cands: List[Dict[str, Any]], target: float) -> List[Dict[str, Any]]:
    n_lo, n_hi = max(5, int(target / 9)), max(7, int(target / 5))
    task = (
        f"ЦЕЛЕВАЯ ДЛИНА ТРЕЙЛЕРА: {target:.0f} с (сумма end_candidate-start_candidate ≈ {target:.0f} с ± 10%).\n\n"
        "КАРТА ИСТОРИИ:\n" + json.dumps(story_map, ensure_ascii=False) +
        "\n\nКАНДИДАТЫ ПО СОБЫТИЯМ (реплики с таймкодами и склейки внутри события):\n"
        + json.dumps(cands, ensure_ascii=False) +
        f"\n\nВыбери {n_lo}–{n_hi} beats в ПОРЯДКЕ ПОКАЗА. Для каждого:\n"
        "audio_owner: \"dialogue\" — звучит оригинальная реплика без закадра (start/end — по таймкодам 1–2 "
        "соседних реплик, сильных и понятных без контекста, до 10 с); \"narration\" — закадровый голос поверх "
        "кадра (3–7 с; лучше там, где важной речи нет); \"both\" — реплика и голос в паузе между репликами "
        "(только если между репликами есть пауза ≥ 2 с — в плотной речи не используй). Чередуй: narration → "
        "dialogue → narration …; не меньше 40% beats — narration, не больше двух реплик подряд; примерно "
        "половина длины — оригинальные реплики. Первый beat — hook, последний — tease без развязки. События из "
        "spoiler_sensitive_events не бери (или spoiler_level ≥ 0.7 и только как намёк).\n"
        "Верни JSON: {\"beats\": [{\"beat_id\": \"B1\", \"event_ids\": [\"E…\"], \"story_function\": \""
        + "|".join(STORY_FUNCTIONS) + "\", \"start_candidate\": сек, \"end_candidate\": сек, \"audio_owner\": \""
        + "|".join(AUDIO_OWNERS) + "\", \"emotion\": \"" + "|".join(BEAT_EMOTIONS) +
        "\", \"spoiler_level\": 0..1, \"priority\": 1..5, \"why\": \"≤ 12 слов\"}]}")
    return [{"role": "system", "content": DIRECTOR_SYSTEM}, {"role": "user", "content": task}]


def validate_beats(data: Any, story_map: Dict[str, Any], duration: float) -> List[Dict[str, Any]]:
    _need(data, ("beats",), "director")
    if not isinstance(data["beats"], list) or len(data["beats"]) < 3:
        raise SchemaError("beats: at least 3 beats expected")
    evs = {e["id"]: e for e in story_map["events"]}
    out, seen = [], set()
    for i, b in enumerate(data["beats"]):
        w = f"beats[{i}]"
        _need(b, ("beat_id", "event_ids", "story_function", "start_candidate", "end_candidate", "audio_owner",
                  "emotion", "spoiler_level", "priority"), w)
        ids = [x for x in _str_list(b["event_ids"], w + ".event_ids", 4, 16)]
        unknown = [x for x in ids if x not in evs]
        if not ids or unknown:
            raise SchemaError(f"{w}.event_ids: unknown ids {unknown or '[]'} (use ids from the story map)")
        for k, allowed in (("story_function", STORY_FUNCTIONS), ("audio_owner", AUDIO_OWNERS),
                           ("emotion", BEAT_EMOTIONS)):
            if b[k] not in allowed:
                raise SchemaError(f"{w}.{k}: one of {allowed}")
        s, e = _num(b["start_candidate"], w + ".start_candidate"), _num(b["end_candidate"], w + ".end_candidate")
        if not (0 <= s < e <= duration + 0.5):
            raise SchemaError(f"{w}: start/end {s}-{e} outside the source 0-{duration:.0f}")
        lo = min(evs[x]["start"] for x in ids) - 10
        hi = max(evs[x]["end"] for x in ids) + 10
        if s < lo or e > hi:
            raise SchemaError(f"{w}: {s:.0f}-{e:.0f} is not inside its events ({lo + 10:.0f}-{hi - 10:.0f})")
        bid = _str(b["beat_id"], w + ".beat_id", 8)
        if bid in seen:
            bid = f"B{i + 1}"
        seen.add(bid)
        try:
            prio = max(1, min(5, int(round(float(b["priority"])))))
        except (TypeError, ValueError):
            raise SchemaError(f"{w}.priority: 1..5") from None
        sp = max(0.0, min(1.0, _num(b["spoiler_level"], w + ".spoiler_level")))
        out.append({"beat_id": bid, "event_ids": ids, "story_function": b["story_function"],
                    "start_candidate": round(s, 2), "end_candidate": round(min(e, duration), 2),
                    "audio_owner": b["audio_owner"], "emotion": b["emotion"], "spoiler_level": sp,
                    "priority": prio, "why": str(b.get("why") or "")[:160]})
    check_alternation(out)
    return out


def check_alternation(beats: Sequence[Dict[str, Any]]) -> None:
    """Spec: voice-over BETWEEN the original lines — at least ~40% voiced beats (narration),
    never more than two dialogue beats in a row."""
    voiced = sum(1 for b in beats if b["audio_owner"] == "narration")
    need = max(2, int(round(len(beats) * 0.4)))
    if voiced < need:
        raise SchemaError(f"only {voiced} narration beats of {len(beats)}; need at least {need} — the voice-over "
                          "goes between the original lines (narration → dialogue → narration …)")
    run = 0
    for b in beats:
        run = run + 1 if b["audio_owner"] != "narration" else 0
        if run > 2:
            raise SchemaError(f"{b['beat_id']}: three non-narration beats in a row — put a narration beat between them")


# ── 3. Writer ───────────────────────────────────────────────────────────────

WRITER_SYSTEM = (
    "Ты — автор закадрового текста кинотрейлеров на русском. Голос — вовлечённый всезнающий рассказчик. "
    "Короткие сильные фразы, напряжение, интрига, причинно-следственная связь между фразами. Нельзя: сухое "
    "перечисление сцен; канцелярит; «в этом видео»; постоянные клише вроде «но он ещё не знает»; одинаковые "
    "зачины. Текст должен звучать естественно вслух. Пиши ТОЛЬКО о событиях из карты истории (по их id) — "
    "никаких новых событий, имён и фактов. Числа — словами. Без кавычек, тире в начале и эмодзи. "
    "Отвечай только JSON.")


def writer_rows(beats: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for b in beats:
        r = {"beat_id": b["beat_id"], "event_ids": b["event_ids"], "story_function": b["story_function"],
             "emotion": b["emotion"], "audio_owner": b["audio_owner"]}
        if b.get("voice_room_sec"):
            r["max_seconds"] = round(b["voice_room_sec"], 1)
            r["max_chars"] = b.get("max_chars")
        if b.get("original_line"):
            r["original_line"] = b["original_line"][:200]
        rows.append(r)
    return rows


def writer_messages(story_map: Dict[str, Any], beats: Sequence[Dict[str, Any]], target: float,
                    previous: Optional[List[Dict[str, Any]]] = None,
                    rewrite_instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    sm = {k: story_map[k] for k in ("characters", "premise", "conflict", "open_questions", "spoiler_sensitive_events")}
    sm["events"] = [{k: e[k] for k in ("id", "summary", "participants", "emotion")} for e in story_map["events"]]
    task = (
        f"Длина трейлера ≈ {target:.0f} с.\n\nКАРТА ИСТОРИИ (единственный источник фактов; ending скрыт "
        "намеренно):\n" + json.dumps(sm, ensure_ascii=False) +
        "\n\nBEATS В ПОРЯДКЕ ПОКАЗА. audio_owner=dialogue — звучит оригинальная реплика (original_line), ты "
        "молчишь; narration и both — твоя фраза, не длиннее max_seconds / max_chars (голос уже измерен):\n"
        + json.dumps(writer_rows(beats), ensure_ascii=False) +
        "\n\nНапиши по одной фразе на каждый beat с max_seconds (можно пропустить один, если кадру лучше "
        "тишина). Фраза подхватывает предыдущую реплику и подводит к следующей. Первая — крючок, последняя — "
        "вопрос или обрыв без ответа.\nВерни JSON: {\"phrases\": [{\"text\": \"…\", \"beat_id\": \"B…\", "
        "\"event_ids\": [\"E…\"], \"target_seconds\": сек}]}")
    if previous is not None and rewrite_instructions:
        task += ("\n\nПРЕДЫДУЩИЙ ВАРИАНТ:\n" + json.dumps(previous, ensure_ascii=False) +
                 "\n\nПРАВКИ РЕДАКТОРА (выполни все):\n" + rewrite_instructions)
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": task}]


def validate_phrases(data: Any, beats: Sequence[Dict[str, Any]], story_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    _need(data, ("phrases",), "writer")
    if not isinstance(data["phrases"], list) or not data["phrases"]:
        raise SchemaError("phrases: non-empty list expected")
    voiced = {b["beat_id"]: b for b in beats if b.get("voice_room_sec")}
    known = {e["id"] for e in story_map["events"]}
    out, used = [], set()
    for i, p in enumerate(data["phrases"]):
        w = f"phrases[{i}]"
        _need(p, ("text", "beat_id", "event_ids", "target_seconds"), w)
        bid = p["beat_id"]
        if bid not in voiced:
            raise SchemaError(f"{w}.beat_id {bid!r}: only beats with max_seconds take a phrase ({sorted(voiced)})")
        if bid in used:
            raise SchemaError(f"{w}: second phrase for {bid}")
        ids = _str_list(p["event_ids"], w + ".event_ids", 4, 16)
        bad = [x for x in ids if x not in known]
        if not ids or bad:
            raise SchemaError(f"{w}.event_ids: {bad or 'empty'} — only event ids from the story map")
        text = _str(p["text"], w + ".text", 300).lstrip("—– ").strip("«»\"").strip()
        used.add(bid)
        out.append({"text": text, "beat_id": bid, "event_ids": ids,
                    "target_seconds": round(_num(p["target_seconds"], w + ".target_seconds"), 2)})
    order = {b["beat_id"]: i for i, b in enumerate(beats)}
    return sorted(out, key=lambda x: order[x["beat_id"]])


# ── 4. Judge ────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "Ты — строгий редактор трейлеров, отдельный от автора. Оцени трейлер (порядок beats + закадр) по шкале "
    "0–10. hook — цепляет ли начало; coherence — одна связная история; suspense — хочется узнать дальше; "
    "factuality — каждая фраза подтверждается картой истории (10 = ничего не выдумано); "
    "emotional_progression — эмоция нарастает; repetition — 10 = нет повторов мыслей и зачинов; "
    "spoiler_control — 10 = развязка не раскрыта; duration_fit — фразы укладываются в max_seconds. "
    "Замечания — конкретные: какая фраза, что не так, как исправить. Отвечай только JSON.")


def judge_messages(story_map: Dict[str, Any], beats: Sequence[Dict[str, Any]], phrases: Sequence[Dict[str, Any]],
                   est_sec: Dict[str, float]) -> List[Dict[str, Any]]:
    by = {p["beat_id"]: p for p in phrases}
    seq = []
    for b in beats:
        r = {"beat_id": b["beat_id"], "story_function": b["story_function"], "audio_owner": b["audio_owner"],
             "event_ids": b["event_ids"]}
        if b.get("original_line"):
            r["original_line"] = b["original_line"][:200]
        if b["beat_id"] in by:
            r["voice"] = by[b["beat_id"]]["text"]
            r["voice_event_ids"] = by[b["beat_id"]]["event_ids"]
            r["max_seconds"] = round(b.get("voice_room_sec") or 0, 1)
            r["voice_seconds_estimate"] = round(est_sec.get(b["beat_id"], 0), 1)
        seq.append(r)
    task = ("КАРТА ИСТОРИИ (с развязкой — для проверки спойлеров):\n" + json.dumps(story_map, ensure_ascii=False) +
            "\n\nТРЕЙЛЕР ПО ПОРЯДКУ:\n" + json.dumps(seq, ensure_ascii=False) +
            "\n\nВерни JSON: {\"scores\": {" + ", ".join(f"\"{k}\": 0-10" for k in JUDGE_KEYS) +
            "}, \"factual_errors\": [str], \"clichés\": [str], \"weak_beats\": [\"B…: почему\"], "
            "\"rewrite_instructions\": \"конкретные правки для автора, одним текстом\"}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": task}]


def validate_judge(data: Any) -> Dict[str, Any]:
    _need(data, ("scores", "factual_errors", "rewrite_instructions"), "judge")
    sc = _need(data["scores"], JUDGE_KEYS, "judge.scores")
    scores = {k: max(0.0, min(10.0, _num(sc[k], f"scores.{k}"))) for k in JUDGE_KEYS}
    cl = data.get("clichés", data.get("cliches"))
    return {"scores": scores, "factual_errors": _str_list(data["factual_errors"], "factual_errors", 10, 300),
            "clichés": _str_list(cl, "clichés", 10, 200),
            "weak_beats": _str_list(data.get("weak_beats"), "weak_beats", 10, 200),
            "rewrite_instructions": str(data.get("rewrite_instructions") or "")[:2000]}


def needs_rewrite(verdict: Dict[str, Any]) -> List[str]:
    """Spec thresholds: factuality < 9, coherence < 8, hook < 8, suspense < 7 → one rewrite."""
    return [f"{k} {verdict['scores'][k]:g} < {v}" for k, v in REWRITE_THRESHOLDS.items() if verdict["scores"][k] < v]
