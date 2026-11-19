"""
stream_viral.py
===============
Streamer moment selection on the Viral peak engine (owner 2026-11-19: «взять из режима Вирал уровень обнаружения
пика того, что происходит»), stream-moments-v3.

Viral (cut_selection.select_story_clips + llm_editor, the Mac mini recipe) finds peaks well because:
  • candidates are whole stories — true sentence start, a statement end, never a question / a cut-off word;
  • the local rank adds the audio energy peak to the story rules (fragment openers, closure, choppiness);
  • the editor judges ONLY the clip's own words on an ABSOLUTE scale (85+ shared / 70+ solid / < 50 never) and
    says whether it is self-contained; strong + self-contained first, weak ones only to reach a floor.

v3 = that engine + the stream on top:
  • candidates: Viral story candidates ∪ the stream candidates (GigaAM run-on speech: pause-ended sentences);
  • local score: stream excitement (voice, reactions, game cuts / flashes, chat) + the Viral story score
    − routine chat talk (donation / greeting reads, «сейчас продолжим», sound checks) − ad inserts (stream_ads: out);
  • editor (one Gemini call, analysis only): absolute rubric + stream rules; sees the setup phrases BEFORE the
    moment and may start the clip there («завязка до панчлайна»), flags chat_talk / ad / self_contained,
    writes a context title;
  • final check (second small call): the exact final span of every pick — is it understandable without the
    stream, routine chat, an ad? Failures go, the floor (3 for a 30+ min stream) is kept with the best of the rest;
  • «лучше 5–6 сильных, чем 8 со слабыми»: no filling to 8 any more; NO_CLIPS only without speech.
Output format = stream_moments.select_moments (the runner does not care which engine ran).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import stream_moments as sm

logger = logging.getLogger(__name__)

VERSION = "stream-moments-v3-viral"
PROMPT_VERSION = "stream-editor-v4-viral"
CHECK_PROMPT_VERSION = "stream-final-check-v1"

# routine talk with the chat: thanks for donations / subs, greetings, nick reads, «сейчас продолжим», sound checks
_CHAT = re.compile(
    r"(?i)(спасибо\s+(тебе\s+|вам\s+|большое\s+|огромное\s+)?(за\s+)?(донат|подписк|поддержк|сабк|фоллоу|"
    r"стрим|рейд|буст|платн|тёпл|тепл|слова)\w*|задонатил\w*|донат\w*|подписал\w*|подписчик\w*|сабк\w*|"
    r"\bчат\w*|в\s+чате|всем\s+привет|привет\s+всем|добро\s+пожаловать|хорошего\s+(просмотра|вечера|дня)|"
    r"удачного\s+стрима|приятного\s+просмотра|вопрос\w*\s+(от|из\s+чата)|пишут|спрашива\w+|отвечаю|"
    r"сейчас\s+продолжим|щас\s+продолжим|отойду|перерыв\w*|звук\w*|микрофон\w*|слышно|не\s+слышно|лага\w*|"
    r"фпс|пинг\w*|настройк\w*|рублей\s+от|до\s+завтра|всем\s+пока|ребят\w*|братух\w*)")
_THANKS = re.compile(r"(?i)спасибо")


def chat_talk(text: str, dur: float) -> Tuple[float, int]:
    """(penalty, cue count): routine chat talk density. 3+ cues in ~30 s (or 3+ «спасибо») is a chat segment."""
    n = len(_CHAT.findall(text)) + max(0, len(_THANKS.findall(text)) - 1)
    rate = n * 30.0 / max(dur, 10.0)
    if rate < 2.5:
        return 0.0, n
    return round(min(2.5, 0.45 * (rate - 1.5)), 3), n


def floor_for(n_sec: float, max_clips: int, cfg: Optional[sm.MomentConfig] = None) -> int:
    """«лучше 5–6 сильных, чем 8 со слабыми»: 3 for a 30+ min stream, 2 for 7+ min, 1 below (STREAMER_MIN_CLIPS_V3)."""
    import os
    if cfg is not None and cfg.min_clips is not None:
        return max(0, min(max_clips, int(cfg.min_clips)))
    try:
        long_floor = int(os.environ.get("STREAMER_MIN_CLIPS_V3") or 3)
    except ValueError:
        long_floor = 3
    f = long_floor if n_sec >= 1800 else min(long_floor, 2) if n_sec >= 420 else 1
    return max(0, min(max_clips, f))


# ── candidates ──────────────────────────────────────────────────────────────

def story_candidates(words, cfg: sm.MomentConfig) -> List[Dict[str, Any]]:
    """Viral whole-story candidates in the stream's length window (after the jump cuts)."""
    from .cut_selection import StoryConfig, build_story_candidates
    scfg = StoryConfig(min_duration=cfg.min_duration, max_duration=cfg.max_duration + 15.0,
                       skip_intro_sec=cfg.skip_intro_sec, skip_intro_min_source=0.0)
    out = []
    for c in build_story_candidates(words, scfg):
        t = sm._tight(words, c["first"], c["last"], cfg.jump_gap)
        if not (cfg.min_duration - 1e-6 <= t <= cfg.max_duration + 1e-6):
            continue
        out.append(dict(c, tight=t, origin="story"))
    return out


def _story_score(c: Dict[str, Any], loud: Sequence[float]) -> Tuple[float, List[str]]:
    from .cut_selection import StoryConfig, story_score
    need = ("questions", "q_end_times", "next_is_question", "short_ratio", "longest_sentence", "words")
    if not all(k in c for k in need):
        return 0.0, []
    sc, why = story_score(c, StoryConfig(strategy="viral", energy=list(loud) if loud is not None else None))
    return sc, ["viral " + f"{sc:.2f}"] + [w for w in why if not w.endswith(" s")]


# ── editor ──────────────────────────────────────────────────────────────────

def _system() -> str:
    from .llm_editor import SYSTEM as VIRAL_SYSTEM
    return (VIRAL_SYSTEM.replace("из длинных русскоязычных интервью и шоу", "из записей русскоязычных стримов "
                                 "(игры, IRL)") +
            " СТРИМ: зритель видит вебку стримера и игру, паузы вырезаны. Сильный клип со стрима: игровое событие "
            "(погоня, смерть, провал, клатч, абсурд) + яркая реакция стримера (крик, смех, мат от шока, перепалка), "
            "смешное, неожиданное, эпичное, приход гостя с подводкой. Клип должен быть ПОНЯТЕН БЕЗ СТРИМА: в нём "
            "есть завязка (что происходит / о чём речь) до панчлайна — если завязка стоит в фразах ДО фрагмента, "
            "начни клип с неё. НЕ клип (score < 40): рутинное общение с чатом — благодарности за донаты и подписки, "
            "приветы по никам, ответы на вопросы чата, «сейчас продолжим», настройки звука, технические паузы — "
            "если в нём нет яркой реакции, шутки или конфликта; реклама и спонсорские вставки (промокоды, ссылки, "
            "сервисы, казино, скины) — никогда; спокойное комментирование без события; обрывок без начала.")


def editor_messages(pool: List[Dict[str, Any]], want: int, context: str = "", types: Sequence[str] = (),
                    max_sec: float = 45.0) -> List[Dict[str, str]]:
    items = []
    for c in pool:
        items.append({"id": c["id"], "signals": c.get("tags", ""), "clip_starts_at": c["core_idx"],
                      **({"lead_in": c["lead_in_txt"]} if c.get("lead_in_txt") else {}),
                      "phrases": [f"{i}) {t:+.0f}s {txt[:240]}" for i, (t, txt) in enumerate(c["sentences_txt"])]})
    from .stream_types import editor_types_line
    head = (f"Стрим: {context[:200]}\n" if context else "") + editor_types_line(types, max_sec)
    user = (head + f"Нужно найти до {want} СИЛЬНЫХ клипов (лучше меньше, но сильных; бывает, что сильных 3–6). "
            "Для каждого фрагмента: phrases — фразы с номером и временем от начала момента (отрицательное время — "
            "то, что было ДО момента: там может быть завязка); clip_starts_at — номер фразы, с которой начинается "
            "сам момент. Выбери start (номер фразы начала клипа: с завязки, если без неё непонятно, иначе сразу с "
            "события) и end (номер фразы конца — после развязки/реакции), чтобы вышло 20–" + str(int(max_sec + 5)) +
            " с.\nФрагменты:\n" + json.dumps(items, ensure_ascii=False) +
            "\n\nВерни JSON: {\"clips\": [{\"id\": str, \"score\": 0-100 (абсолютная шкала), "
            "\"type\": \"эмоция|конфликт|смешное|неожиданное|фейл|клатч|история|гость|выход|шутка|философия|завоз|хаос|чат|реклама|скучно\", "
            "\"self_contained\": bool (понятно без стрима: есть завязка и развязка), \"chat_talk\": bool (рутинное "
            "общение с чатом без яркой реакции), \"ad\": bool (реклама/спонсор), "
            "\"use_lead_in\": bool (только если есть lead_in — подводка к гостю), \"start\": int, \"end\": int, "
            "\"payoff\": int (номер фразы развязки), \"punchline\": int (номер ударной фразы), "
            "\"title\": \"заголовок 3–7 слов по-русски: ЧТО происходит (контекст), только то, что правда есть в клипе\", "
            "\"why\": \"до 10 слов\", \"keywords\": [до 3 слов ИЗ ТЕКСТА]}]} — для КАЖДОГО фрагмента (всего "
            + str(len(items)) + ", верни все " + str(len(items)) + "), по убыванию score.")
    return [{"role": "system", "content": _system()}, {"role": "user", "content": user}]


def parse_editor(text: str, pool: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Editor JSON → verdicts (fail-closed: unknown ids dropped, bad phrase numbers clamped, flags must be true)."""
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M))
    except ValueError:
        return []
    out = []
    for x in (data.get("clips") if isinstance(data, dict) else data) or []:
        if not isinstance(x, dict) or x.get("id") not in pool:
            continue
        c = pool[x["id"]]
        n = len(c.get("sentences") or [None])
        try:
            score = max(0.0, min(100.0, float(x.get("score", 0))))
            st = int(x.get("start", c["core_idx"]))
            en = int(x.get("end", n - 1))
        except (TypeError, ValueError):
            continue
        st, en = max(0, min(n - 1, st)), max(0, min(n - 1, en))
        if en < st:
            st, en = c["core_idx"], n - 1
        said = {re.sub(r"[^\w-]", "", w).lower() for w in c["view_text"].split()}
        kws = [k for k in (x.get("keywords") or [])[:3]
               if isinstance(k, str) and re.sub(r"[^\w-]", "", k).lower() in said]
        typ = str(x.get("type") or "").strip().lower()[:20]

        def _idx(k):
            try:
                i = int(x.get(k))
            except (TypeError, ValueError):
                return None
            return i if st <= i <= en else None
        sc = x.get("self_contained") is True
        out.append({"id": x["id"], "score": score, "type": typ, "self_contained": sc, "clear_3s": sc,
                    "chat_talk": x.get("chat_talk") is True or typ == "чат",
                    "ad": x.get("ad") is True or typ == "реклама",
                    "start": st, "end": en, "hook": str(x.get("title") or "").strip()[:60],
                    "reason": str(x.get("why") or "").strip()[:160], "keywords": kws,
                    "payoff": _idx("payoff"), "punchline": _idx("punchline"),
                    "use_lead_in": x.get("use_lead_in") is True})
    return out


def _editor_view(c: Dict[str, Any], words, all_sents, cfg: sm.MomentConfig, ads) -> None:
    """Phrases the editor may choose from: ≤ 18 s of setup BEFORE the moment + the moment grown to ~55 s.
    Never into an ad insert."""
    from . import stream_ads
    last = c["last"]
    for s0, e0 in all_sents:
        if s0 <= last:
            continue
        if sm._tight(words, c["first"], e0, cfg.jump_gap) > cfg.max_duration + 10.0 \
                or stream_ads.overlaps(words[s0].start, words[e0].end, ads, 0.3):
            break
        last = e0
    first = c["first"]
    for s0, e0 in reversed(all_sents):
        if e0 >= first:
            continue
        if c["start"] - words[s0].start > 18.0 or stream_ads.overlaps(words[s0].start, words[e0].end, ads, 0.3):
            break
        first = s0
    view = dict(c, first=first, last=last)
    sents = sm._sentence_list(words, view)
    c["view_first"], c["view_last"] = first, last
    c["sentences"] = sents
    c["core_idx"] = next((i for i, (a, _b) in enumerate(sents) if a >= c["first"]), 0)
    c["sentences_txt"] = [(words[a].start - c["start"], " ".join(w.text.strip() for w in words[a:b + 1]))
                          for a, b in sents]
    c["view_text"] = " ".join(t for _, t in c["sentences_txt"])


def _apply_trim(c: Dict[str, Any], v: Dict[str, Any], words, cfg: sm.MomentConfig) -> Dict[str, Any]:
    """Editor's start / end phrases → word span; the setup may add ≤ 5 s over max_duration."""
    sents = c.get("sentences") or []
    if not sents:
        return c
    hi = cfg.max_duration + (5.0 if v["start"] < c["core_idx"] else 3.0)
    a, j = sents[v["start"]][0], v["end"]
    b = sents[j][1]
    while sm._tight(words, a, b, cfg.jump_gap) < cfg.min_duration and j + 1 < len(sents):
        j += 1
        b = sents[j][1]
    while sm._tight(words, a, b, cfg.jump_gap) > hi and j > max(v["start"], c["core_idx"]):
        j -= 1
        b = sents[j][1]
    t = sm._tight(words, a, b, cfg.jump_gap)
    if not (cfg.min_duration - 2.0 <= t <= hi):
        return dict(c, trim_note="editor span rejected (length)")
    if a == c["first"] and b == c["last"]:
        return c
    from .cut_selection import _first_word
    return dict(c, first=a, last=b, start=words[a].start, end=words[b].end, tight=t, opener=_first_word(words[a].text),
                text=" ".join(w.text.strip() for w in words[a:b + 1]),
                trim_note=f"editor trim: phrases {v['start']}-{j} (moment at {c['core_idx']})")


def _call(chat, msgs, prompt_version, cfg, timeout=180.0) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"prompt_version": prompt_version}
    try:
        if chat is None:
            from . import llm_provider
            chat = llm_provider.chat_json
        res = chat(msgs, prompt_version=prompt_version, opt_in=True, timeout=timeout,
                   extra={"reasoning": {"effort": "low"}})
        tel = res.telemetry() if hasattr(res, "telemetry") else {}
        meta.update(tel)
        cost = tel.get("cost_usd")
        meta["cost_rub"] = round(float(cost) * cfg.rub_per_usd, 3) if cost is not None else None
        if cost is None and tel.get("input_tokens") is not None:
            est = (float(tel["input_tokens"]) * sm._PRICE_IN + float(tel.get("output_tokens") or 0) * sm._PRICE_OUT) / 1e6
            meta["cost_rub_estimate"] = round(est * cfg.rub_per_usd, 3)
        meta["status"] = "ok"
        return res.text, meta
    except Exception as exc:                                  # never let the editor kill the run
        meta.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return None, meta


# ── final check ─────────────────────────────────────────────────────────────

CHECK_SYSTEM = ("Ты — выпускающий редактор коротких вертикальных клипов со стримов. Зритель листает ленту и видит "
                "ТОЛЬКО этот клип (вебка + игра, паузы вырезаны) и заголовок. Проверь каждый клип строго. "
                "Отвечай только JSON.")


def check_messages(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    user = ("Клипы:\n" + json.dumps(items, ensure_ascii=False) +
            "\n\nДля КАЖДОГО верни JSON: {\"clips\": [{\"id\": str, \"clear\": bool (понятно без стрима, что "
            "происходит и в чём соль: есть завязка и развязка), \"chat_talk\": bool (это рутинное общение с чатом: "
            "благодарности, приветы, ответы на вопросы, тех. паузы — без яркой реакции/шутки/конфликта), "
            "\"ad\": bool (есть реклама/спонсорская вставка/промокод/ссылка), \"score\": 0-100 (абсолютная: 85+ "
            "поделятся, 70+ крепкий, <50 не публиковать), \"title\": \"заголовок 3–7 слов: что происходит, только "
            "правда из клипа\"}]}")
    return [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}]


def _parse_check(text: Optional[str], ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M))
    except ValueError:
        return {}
    out = {}
    for x in (data.get("clips") if isinstance(data, dict) else data) or []:
        if not isinstance(x, dict) or x.get("id") not in ids:
            continue
        try:
            score = max(0.0, min(100.0, float(x.get("score", 0))))
        except (TypeError, ValueError):
            continue
        out[x["id"]] = {"clear": x.get("clear") is True, "chat_talk": x.get("chat_talk") is True,
                        "ad": x.get("ad") is True, "score": score, "title": str(x.get("title") or "").strip()[:60]}
    return out


# ── selection ───────────────────────────────────────────────────────────────

def select_moments(raw_words: Sequence[Dict[str, Any]], n_sec: int, signals: Dict[str, Sequence[float]],
                   max_clips: int, cfg: Optional[sm.MomentConfig] = None, hits=None,
                   chat: Optional[Callable[..., Any]] = None,
                   ads: Sequence[Tuple[float, float]] = ()) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from .boundaries import normalize_words
    from .cut_selection import is_promo
    from . import stream_ads
    from . import stream_types as st
    cfg = cfg or sm.MomentConfig()
    words, _ = normalize_words(raw_words)
    floor = floor_for(n_sec, max_clips, cfg)
    ads = list(ads or [])
    meta: Dict[str, Any] = {"version": VERSION, "params": {k: v for k, v in asdict(cfg).items()},
                            "signals": sorted(k for k, v in signals.items() if v is not None and len(v)),
                            "llm": {"status": "off"}, "min_clips": floor, "fallback": [],
                            "ads": [[round(a, 1), round(b, 1)] for a, b in ads]}
    e = sm.excitement(signals, n_sec, cfg.weights)
    if not words:
        meta["selection_status"] = "no_words"
        return sm._energy_segments(words, n_sec, e, np.zeros(n_sec), floor, [], cfg, signals, meta), meta
    speech = sm._speech_mask(words, n_sec)
    tctx = st.Context(words, n_sec, signals, hits)
    hits = tctx.hits
    loud = signals.get("loud")
    raw = sm.build_candidates(words, cfg, hits) + story_candidates(words, cfg)
    seen, cands = set(), []
    for c in raw:
        if c["id"] not in seen:
            seen.add(c["id"])
            cands.append(c)
    meta["candidates"] = len(cands)
    any_type = not cfg.moment_types or "conflict" in cfg.moment_types

    def _score_all(cs, conf):
        res = []
        for c in cs:
            c["text"] = sm._text(words, c)
            low = c["text"].lower()
            if stream_ads.overlaps(c["start"], c["end"], ads, 0.5):
                continue
            if is_promo(c["text"]) or sum(low.count(x) for x in sm._ALERT) >= 2:
                continue
            sc, why = sm.score_candidate(c, e, speech, words, conf, hits)
            vs, vwhy = _story_score(c, loud)
            sc += 0.8 * vs
            why += vwhy
            pen, n_chat = chat_talk(c["text"], float(c.get("tight") or (c["end"] - c["start"])))
            c["chat_cues"] = n_chat
            if pen:
                sc -= pen
                if c.get("category") == "guest":           # «спасибо, привет, Антоха» is not a guest entrance
                    sc -= 0.9
                    c["category"] = None
                why.append(f"chat talk ×{n_chat} −{pen:.1f}")
            ts = st.classify(c, tctx)
            c["type_scores"] = {k: v for k, v in ts.items() if v > 0}
            best = st.best_type(ts, conf.moment_types)
            if best:
                if not c.get("category") or best in conf.moment_types:
                    c["category"] = best
                if best in conf.moment_types:
                    sc = max(sc, ts[best]) + 0.5
                    why.append(f"type {best} {ts[best]:.1f}")
            c["typed"] = any_type or c.get("category") in conf.moment_types
            res.append((c, round(sc, 4), why))
        res.sort(key=lambda t: (-t[1], t[0]["start"], t[0]["end"]))
        return res

    def _diverse(items, taken, gap, limit):
        got = []
        for c, sc, why in items:
            if not sm._overlaps(c, [p for p, _, _ in taken] + [p for p, _, _ in got], gap):
                got.append((c, sc, why))
                if len(got) >= limit:
                    break
        return got

    scored = _score_all(cands, cfg)
    limit = max(cfg.pool, max_clips)
    typed = [t for t in scored if t[0]["typed"]]
    meta["typed_candidates"] = len(typed)
    # requested moment types first; the rest compete on merit (the editor sees the request)
    pool = _diverse(typed, [], cfg.min_gap_sec, limit)
    if len(pool) < limit:
        pool += _diverse([t for t in scored if not t[0]["typed"]], pool, cfg.min_gap_sec, limit - len(pool))
    meta["pool"] = len(pool)
    chosen = [(c, sc, why + ["local order (no editor)"]) for c, sc, why in pool[:max(floor, min(max_clips, 6))]]
    extras: Dict[str, Dict[str, Any]] = {}
    if cfg.editor and pool:
        all_sents = sm._sentences(words)
        for c, _, _ in pool:
            _editor_view(c, words, all_sents, cfg, ads)
            c["tags"] = sm._tags(c, signals, hits or [])
            if c.get("chat_cues", 0) >= 3:
                c["tags"] += f", чат-фразы×{c['chat_cues']}"
            if c.get("lead_in"):
                a0, b0 = c["lead_in"]
                c["lead_in_txt"] = (f"за {c['start'] - words[a0].start:.0f} с до фрагмента: "
                                    + " ".join(w.text.strip() for w in words[a0:b0 + 1])[:200])
        items = [c for c, _, _ in pool]
        em = lambda p: editor_messages(p, max_clips, cfg.context, cfg.moment_types, cfg.max_duration)  # noqa: E731
        msgs = em(items)
        while len(items) > max_clips and sm.forecast_rub(msgs, len(items), cfg) > cfg.budget_rub * 0.8:
            items = items[:-1]
            msgs = em(items)
        emeta: Dict[str, Any] = {"pool": len(items), "budget_rub": cfg.budget_rub,
                                 "forecast_rub": sm.forecast_rub(msgs, len(items), cfg)}
        verdicts: List[Dict[str, Any]] = []
        if emeta["forecast_rub"] <= cfg.budget_rub * 0.8:
            txt, m = _call(chat, msgs, PROMPT_VERSION, cfg)
            emeta.update(m)
            verdicts = parse_editor(txt, {c["id"]: c for c in items}) if txt else []
            emeta["verdicts"] = len(verdicts)
        else:
            emeta["status"] = "skipped_budget"
        meta["llm"] = emeta
        if verdicts:
            chosen = _pick(verdicts, pool, words, cfg, max_clips, floor, extras, meta)
            meta["selection_method"] = "editor"
            chosen = _final_check(chosen, extras, words, cfg, floor, chat, meta)
    for c, _, why in chosen:
        c["tier"] = 1 if any(w.startswith("fallback") for w in why) else 0
    # NO_CLIPS only without speech: the stream tiers when nothing at all survived (or the floor is not met)
    if len(chosen) < floor:
        rcfg = replace(cfg, min_duration=max(12.0, cfg.min_duration - 5.0), max_duration=cfg.max_duration + 10.0,
                       skip_intro_sec=min(cfg.skip_intro_sec, 30.0), min_gap_sec=min(cfg.min_gap_sec, 8.0),
                       moment_types=[])
        rc = sm.build_candidates(words, rcfg, hits, relaxed=True)
        more = [t for t in _diverse(_score_all(rc, rcfg), chosen, rcfg.min_gap_sec, floor - len(chosen))
                if t[0].get("chat_cues", 0) < 3]
        for c, sc, why in more:
            c["tier"] = 2
            why.append("fallback: relaxed phrase rules")
        if more:
            meta["fallback"].append(f"relaxed+{len(more)}")
            chosen = chosen + [(c, sc - 2.0, why) for c, sc, why in more]
    if not chosen:
        ew = [c for c in sm.energy_windows(words, n_sec, e, speech, max(1, floor), [], cfg)
              if not stream_ads.overlaps(c["start"], c["end"], ads, 0.5)]
        for c in ew:
            c["tier"] = 3
            c["text"] = c.get("text") or ""
            chosen.append((c, -3.0, [f"fallback: energy peak at {c['peak_t']:.0f}s, phrase-snapped"]))
        if ew:
            meta["fallback"].append(f"energy+{len(ew)}")
    meta.update(selection_method=meta.get("selection_method", "local"), selection_status="ok" if chosen else "none",
                curve_p95=round(float(np.percentile(e, 95)), 3) if len(e) else 0.0,
                selected=[{"id": c["id"], "start": round(c["start"], 3), "end": round(c["end"], 3),
                           "score": round(sc, 4), "tier": c.get("tier", 0), "reason": "; ".join(why),
                           "hook": (extras.get(c["id"]) or {}).get("hook"), "text": (c.get("text") or "")[:300]}
                          for c, sc, why in chosen])
    c0_by = {p["id"]: p for p, _, _ in pool}
    segs = [{"start": c["start"], "duration": c["end"] - c["start"], "score": sc,
             "source": "stream_moment:" + ("editor" if c["id"] in extras else
                                           "local" if c.get("tier", 0) <= 1 else "fallback"),
             "tier": c.get("tier", 0), "engine": "viral",
             "candidate_id": c["id"], "selection_reason": "; ".join(why), "peak_t": c.get("peak_t"),
             "tight_sec": c.get("tight"),
             **(sm._lead_and_category(c, extras.get(c["id"]), words) if c.get("first") is not None else {}),
             **({"hook": extras[c["id"]]["hook"], "keywords": extras[c["id"]]["keywords"],
                 "editor_score": extras[c["id"]]["score"], **sm._marks(c0_by.get(c["id"]), extras[c["id"]], words)}
                if c["id"] in extras else {})}
            for c, sc, why in chosen]
    return segs, meta


def _pick(verdicts, pool, words, cfg, max_clips, floor, extras, meta):
    """Viral tiers: strong + self-contained first; weaker only up to the floor; ads never; chat talk last."""
    import os
    by_id = {c["id"]: (c, sc, why) for c, sc, why in pool}
    try:
        strong = float(os.environ.get("STREAMER_V3_STRONG") or 70.0)
    except ValueError:
        strong = 70.0
    picked: List[Tuple[Dict[str, Any], float, List[str]]] = []
    clean = lambda v: not v["ad"] and v["type"] not in ("скучно", "реклама")   # noqa: E731
    tiers = ((2.0, max_clips, "", lambda v: clean(v) and not v["chat_talk"] and v["self_contained"] and v["score"] >= strong),
             (1.8, max_clips, "", lambda v: clean(v) and not v["chat_talk"] and v["score"] >= max(strong, 80.0)),
             (1.4, floor, "fallback: below the strong bar", lambda v: clean(v) and not v["chat_talk"] and v["self_contained"] and v["score"] >= 50),
             (1.1, floor, "fallback: not self-contained", lambda v: clean(v) and not v["chat_talk"] and v["score"] >= 55),
             (0.8, floor, "fallback: weak", lambda v: clean(v) and v["score"] >= 35))
    for tier, cap, note, ok in tiers:
        for v in sorted(verdicts, key=lambda v: -v["score"]):
            if len(picked) >= cap:
                break
            if v["id"] in extras or not ok(v):
                continue
            c0, sc, why = by_id[v["id"]]
            c = _apply_trim(c0, v, words, cfg)
            if sm._overlaps(c, [p for p, _, _ in picked], cfg.min_gap_sec):
                continue
            flags = ("" if v["self_contained"] else " (needs context)") + (" (chat)" if v["chat_talk"] else "")
            picked.append((c, v["score"] / 100.0 + tier + 0.05 * sc,
                           why + [f"editor {v['score']:.0f} {v['type']}{flags}: {v['reason']}"]
                           + ([c["trim_note"]] if c.get("trim_note") else []) + ([note] if note else [])))
            extras[c["id"]] = v
    meta["editor_rejected"] = len(verdicts) - len(picked)
    meta["editor_flags"] = {"ad": sum(v["ad"] for v in verdicts), "chat_talk": sum(v["chat_talk"] for v in verdicts),
                            "not_self_contained": sum(not v["self_contained"] for v in verdicts)}
    picked.sort(key=lambda t: -t[1])
    return picked


def _final_check(chosen, extras, words, cfg, floor, chat, meta):
    """Second look at the exact final spans (after the trim): unclear / chat / ad → out, the floor is kept."""
    import os
    if not chosen or os.environ.get("STREAMER_V3_FINAL_CHECK", "on").lower() in ("0", "off", "false", "no"):
        return chosen
    items = []
    for c, _, _ in chosen:
        v = extras.get(c["id"]) or {}
        items.append({"id": c["id"], "title": v.get("hook", ""), "dur": round(float(c.get("tight") or c["end"] - c["start"])),
                      "text": sm._text(words, c)[:900]})
    msgs = check_messages(items)
    fc = sm.forecast_rub(msgs, len(items), cfg)
    if fc > cfg.budget_rub * 0.4:
        meta["final_check"] = {"status": "skipped_budget", "forecast_rub": fc}
        return chosen
    txt, m = _call(chat, msgs, CHECK_PROMPT_VERSION, cfg, timeout=90.0)
    m["forecast_rub"] = fc
    res = _parse_check(txt, [c["id"] for c, _, _ in chosen])
    meta["final_check"] = dict(m, verdicts=len(res))
    if not res:
        return chosen
    try:
        bar = float(os.environ.get("STREAMER_V3_CHECK_SCORE") or 55.0)
    except ValueError:
        bar = 55.0
    keep, weak, dropped = [], [], []
    for c, sc, why in chosen:
        r = res.get(c["id"])
        if r is None:
            keep.append((c, sc, why))
            continue
        v = extras.get(c["id"])
        if v is not None and r["title"] and not v.get("hook"):
            v["hook"] = r["title"]
        note = f"final check {r['score']:.0f}{'' if r['clear'] else ' unclear'}{' chat' if r['chat_talk'] else ''}{' AD' if r['ad'] else ''}"
        if r["ad"]:
            dropped.append(c["id"])
            continue
        if r["clear"] and not r["chat_talk"] and r["score"] >= bar:
            keep.append((c, sc, why + [note]))
        else:
            weak.append((c, sc + r["score"] / 100.0 - (0.5 if r["chat_talk"] else 0.0), why + [note, "fallback: final check (floor)"]))
    weak.sort(key=lambda t: -t[1])
    while len(keep) < floor and weak:
        keep.append(weak.pop(0))
    meta["final_check"].update(dropped_ad=dropped, dropped_weak=[c["id"] for c, _, _ in weak])
    return keep
