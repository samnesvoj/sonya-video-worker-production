"""
cut_selection.py
================
ASR-grounded clip selection for SONYA Cut (virality).

    words ──► sentences ──► candidates (complete phrases, stable IDs)
          ──► ranking: Gemini picks candidate IDs (opt-in)  |  local fallback
          ──► non-overlapping top N ──► boundaries-v1 ──► render

Timestamps ALWAYS come from ASR words: the LLM only returns candidate IDs
(+ score/reason). Unknown IDs are discarded; if nothing valid remains the
deterministic local ranking is used.

The local ranking is a TECHNICAL fallback, not a judgement of what is a
good moment: it prefers complete sentences, steady speech and durations
near the target, and breaks ties by start time then ID so the same input
always gives the same output. Real quality is decided by the user.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .boundaries import Word, group_sentences, normalize_words

logger = logging.getLogger(__name__)

CANDIDATES_VERSION = "cut-candidates-v2"  # v2: canonical integer-ms + NFC ids
LOCAL_RANK_VERSION = "cut-local-rank-v1"
RANKING_VERSION = "cut-rank-v2"
TIE_BREAK = ["score desc", "local_score desc", "start asc", "end asc", "candidate_id asc"]
PROMPT_VERSION = "cut-select-v1"
_TERMINAL = re.compile(r"[.!?…]+[\"'»”)\]]*$")


@dataclass
class CandidateConfig:
    min_duration: float = 15.0
    max_duration: float = 60.0
    target_duration: float = 35.0
    pause_sec: float = 0.7
    max_per_start: int = 3          # how many end points per sentence start
    max_for_llm: int = 120          # candidates sent to the LLM
    text_chars_for_llm: int = 600

    def __post_init__(self) -> None:
        if not (0 < self.min_duration <= self.max_duration):
            raise ValueError("candidates: need 0 < min_duration <= max_duration")
        if not (self.min_duration <= self.target_duration <= self.max_duration):
            raise ValueError("candidates: target_duration must be within [min, max]")
        if self.pause_sec <= 0 or self.max_per_start < 1 or self.max_for_llm < 1:
            raise ValueError("candidates: pause_sec > 0, max_per_start >= 1, max_for_llm >= 1")


@dataclass
class Candidate:
    id: str
    start: float
    end: float
    first_word: int
    last_word: int
    sentences: int
    words: int
    text: str
    complete: bool                  # ends on terminal punctuation

    @property
    def duration(self) -> float:
        return self.end - self.start


def _ms(t: float) -> int:
    """Canonical time: integer milliseconds (1.0, 1.0000001, "1.000" -> 1000)."""
    return int(round(float(t) * 1000))


def _canon_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def candidate_id(words: Sequence[Word], first: int, last: int) -> str:
    """Content id from canonical ASR data only: integer-ms start/end + NFC,
    whitespace-collapsed text. Independent of input word order, dict/set
    order and PYTHONHASHSEED (sha1, not hash())."""
    key = f"{_ms(words[first].start)}|{_ms(words[last].end)}|" + \
        " ".join(_canon_text(w.text) for w in words[first:last + 1])
    return "c" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def tie_key(c: "Candidate") -> Tuple[int, int, str]:
    """Final tie-break after equal scores: earlier start, then shorter, then id."""
    return (_ms(c.start), _ms(c.end), c.id)


def build_candidates(words: Sequence[Word], cfg: CandidateConfig) -> List[Candidate]:
    groups = group_sentences(words, cfg.pause_sec)
    out: Dict[str, Candidate] = {}
    for gi, (gs, _) in enumerate(groups):
        emitted = 0
        for gj in range(gi, len(groups)):
            ge = groups[gj][1]
            dur = words[ge].end - words[gs].start
            if dur > cfg.max_duration + 1e-6:
                break
            if dur + 1e-6 < cfg.min_duration:
                continue
            cid = candidate_id(words, gs, ge)
            out[cid] = Candidate(cid, words[gs].start, words[ge].end, gs, ge, gj - gi + 1, ge - gs + 1,
                                 " ".join(w.text.strip() for w in words[gs:ge + 1]),
                                 bool(_TERMINAL.search(words[ge].text.strip())))
            emitted += 1
            if emitted >= cfg.max_per_start:
                break
    return sorted(out.values(), key=lambda c: (c.start, c.end, c.id))


def local_score(c: Candidate, cfg: CandidateConfig) -> float:
    density = c.words / max(c.duration, 1e-6)                  # words per second
    density_term = -abs(density - 2.5) / 2.5                    # steady conversational pace
    length_term = -abs(c.duration - cfg.target_duration) / cfg.target_duration
    return round((1.0 if c.complete else 0.0) + density_term + length_term, 6)


def rank_local(cands: Sequence[Candidate], cfg: CandidateConfig) -> List[Tuple[Candidate, float, str]]:
    scored = [(c, local_score(c, cfg)) for c in cands]
    scored.sort(key=lambda t: (-t[1], *tie_key(t[0])))        # total order, see TIE_BREAK
    return [(c, s, "local: complete=%s, %.1f s" % (c.complete, c.duration)) for c, s in scored]


def pick_non_overlapping(ranked: Sequence[Tuple[Candidate, float, str]], n: int
                         ) -> List[Tuple[Candidate, float, str]]:
    chosen: List[Tuple[Candidate, float, str]] = []
    for c, s, r in ranked:
        if all(c.end <= k.start or c.start >= k.end for k, _, _ in chosen):
            chosen.append((c, s, r))
            if len(chosen) >= n:
                break
    return chosen


def build_prompt(cands: Sequence[Candidate], n: int, cfg: CandidateConfig) -> List[Dict[str, str]]:
    items = [{"id": c.id, "start": round(c.start, 2), "end": round(c.end, 2),
              "text": c.text[:cfg.text_chars_for_llm]} for c in cands]
    system = ("You select short-form clips from a long interview/podcast transcript for SONYA Cut "
              "(viral mode): emotional peaks, strong claims, reactions, arguments, punchlines, "
              "moments that work on their own without the rest of the video. "
              "You may ONLY choose from the candidate ids given. Never invent ids or timestamps. "
              'Answer with JSON: {"selected": [{"id": "...", "score": 0-10, "reason": "..."}]} '
              f"with at most {n} items, best first, non-overlapping in time.")
    user = "Candidates (JSON):\n" + json.dumps(items, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_llm_selection(text: str, valid: Dict[str, Candidate]) -> Tuple[List[Tuple[str, float, str]], List[str]]:
    """-> ([(id, score, reason)], problems). Only ids that exist survive."""
    problems: List[str] = []
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("selected"), list):
        return [], ["malformed_response"]
    best: Dict[str, Tuple[float, str]] = {}
    for item in data["selected"]:
        if not isinstance(item, dict):
            problems.append("non_object_item")
            continue
        cid = item.get("id")
        if cid not in valid:
            problems.append(f"unknown_id:{str(cid)[:40]}")
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        if score != score or score in (float("inf"), float("-inf")):  # NaN / inf
            problems.append(f"bad_score:{cid}")
            score = 0.0
        score = max(0.0, min(10.0, score))
        reason = str(item.get("reason", ""))[:500]
        if cid in best:
            problems.append(f"duplicate_id:{cid}")
            if (score, reason) <= best[cid]:   # deterministic: max score, then max reason text
                continue
        best[cid] = (score, reason)
    out = [(cid, s, r) for cid, (s, r) in sorted(best.items())]
    return out, problems


# ── v3: whole stories / question + answer ───────────────────────────────
#
# Owner's verdict on v2: clips did not feel finished — one started mid-sentence,
# one ended on a long question with no answer. v3 builds candidates from TRUE
# sentences only (start after terminal punctuation, end on . ! …), lets a clip
# run up to 90 s when an answer needs it, and scores the structure of the talk:
# a question near the start that is answered inside the clip, an ending at a
# natural closure (the next thing is a new question or a long pause), no
# fragment openers ("и", "а", "ну", …), never ending on an unanswered question.

LOCAL_RANK_V3 = "cut-local-rank-v3"
CANDIDATES_V3 = "cut-candidates-v3"
_Q_END = re.compile(r"\?[\"'»”)\]]*$")
_STATEMENT_END = re.compile(r"(?<![.…])[.!]+[\"'»”)\]]*$")   # "…" / "..." trails off: not an end
_TRAILING = re.compile(r"(\.\.|…)[\"'»”)\]]*$")
_FRAGMENT_OPENERS = {"и", "а", "но", "ну", "вот", "потому", "поэтому", "так", "то", "что", "это", "да", "нет",
                     "или", "же", "ещё", "еще", "тоже", "также", "который", "которая", "которые", "когда"}


@dataclass
class StoryConfig:
    min_duration: float = 20.0
    max_duration: float = 90.0
    sweet_min: float = 25.0           # no length penalty inside [sweet_min, sweet_max]
    sweet_max: float = 60.0
    closure_pause: float = 0.8        # a pause this long after the clip = natural end
    min_answer_sec: float = 8.0
    min_question_words: int = 4
    skip_intro_sec: float = 90.0      # YouTube interviews open with a teaser montage of later moments …
    skip_intro_min_source: float = 600.0   # … only long sources have one
    exclude: Optional[List[List[float]]] = None   # [start, end] already delivered: pick NEW moments
    # "qa" format (owner liked it): the clip OPENS with the interviewer's question and
    # the rest is the answer — no further questions inside, ends at a closure
    qa_only: bool = False
    editor: bool = False      # LLM editor re-ranks the local pool (scripts/shared/llm_editor.py)
    promo: Optional[Dict[str, Any]] = None   # advertiser spec (scripts/shared/promo.py): conditions → selection
    # Guide strategies (docs/MODES_SPEC.md §1): qa | story | viral | educational | hooks; "" = qa_only flag rules
    strategy: str = ""
    energy: Optional[List[float]] = None     # loudness z-scores per second (scripts/shared/audio_features.py)


def true_sentences(words: Sequence[Word]) -> List[Tuple[int, int]]:
    """Sentences by punctuation only (a pause inside a sentence is not an end)."""
    out: List[Tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        if _TERMINAL.search(w.text.strip()) or i == len(words) - 1:
            out.append((start, i))
            start = i + 1
    return out


def _first_word(text: str) -> str:
    m = re.match(r"[\W_]*([\w-]+)", text)
    return (m.group(1) if m else "").lower()


def build_story_candidates(words: Sequence[Word], cfg: StoryConfig) -> List[Dict[str, Any]]:
    sents = true_sentences(words)
    long_source = bool(words) and words[-1].end >= cfg.skip_intro_min_source
    excl = [(float(a), float(b)) for a, b in (cfg.exclude or [])]
    # a real question: ends with "?" and has some substance ("Да?" / "Карамзин?" are not)
    is_q = [bool(_Q_END.search(words[e].text.strip())) and e - b + 1 >= cfg.min_question_words for b, e in sents]
    lens = [words[e].end - words[b].start for b, e in sents]
    any_q = [bool(_Q_END.search(words[e].text.strip())) for _, e in sents]   # incl. "А почему?"
    out: Dict[str, Dict[str, Any]] = {}
    for i, (s0, _) in enumerate(sents):
        if long_source and words[s0].start < cfg.skip_intro_sec:
            continue
        for j in range(i, len(sents)):
            e = sents[j][1]
            dur = words[e].end - words[s0].start
            if dur > cfg.max_duration + 1e-6:
                break
            end_txt = words[e].text.strip()
            if dur + 1e-6 < cfg.min_duration or not _STATEMENT_END.search(end_txt) or _TRAILING.search(end_txt):
                continue                               # never end on a question or a cut-off word
            if any(words[s0].start < b and words[e].end > a for a, b in excl):
                continue
            cid = candidate_id(words, s0, e)
            nxt_gap = (words[e + 1].start - words[e].end) if e + 1 < len(words) else 99.0
            prev_gap = (words[s0].start - words[s0 - 1].end) if s0 > 0 else 99.0
            qs = [k for k in range(i, j + 1) if is_q[k]]
            out[cid] = {"id": cid, "start": words[s0].start, "end": words[e].end, "first": s0, "last": e,
                        "sentences": j - i + 1, "words": e - s0 + 1,
                        "text": " ".join(w.text.strip() for w in words[s0:e + 1]),
                        "opener": _first_word(words[s0].text), "questions": qs, "sent_range": (i, j),
                        "first_sentence": " ".join(w.text.strip() for w in words[s0:sents[i][1] + 1]),
                        "next_is_question": j + 1 < len(sents) and is_q[j + 1],
                        "next_gap": nxt_gap, "prev_gap": prev_gap,
                        "q_end_times": [words[sents[k][1]].end for k in qs],
                        "inner_any_q": sum(1 for k in range(i + 1, j + 1) if any_q[k]),
                        # short sentences in a row = teaser montage / ping-pong banter, not a moment
                        "short_ratio": sum(1 for k in range(i, j + 1) if lens[k] < 2.0) / (j - i + 1),
                        "longest_sentence": max(sents[k][1] - sents[k][0] + 1 for k in range(i, j + 1))}
    return sorted(out.values(), key=lambda c: (c["start"], c["end"], c["id"]))


_TO_GUEST = re.compile(r"(?i)(?<![\w-])(ты|тебя|тебе|тобой|твой|твоя|твоё|твое|твои|твоего|твоей|твоих|"
                       r"вы|вас|вам|вами|ваш|ваша|ваше|ваши|вашего|вашей|"
                       r"\w+(ешь|ишь|ёшь|ете|ите))(?![\w-])")
_SELF_Q = re.compile(r"(?i)(?<![\w-])(я|мне|меня|мной|мой|моя|моё|мое|мои)(?![\w-])")


def asks_the_guest(question: str) -> bool:
    """Interviewer's question: addressed to the guest ("ты", "вы", "-ешь/-ете"),
    not the guest asking himself ("как я оцениваю?")."""
    return bool(_TO_GUEST.search(question)) and not (_SELF_Q.search(question) and not
                                                    re.search(r"(?i)(?<![\w-])(ты|вы|тебя|вас)(?![\w-])", question))


_PROMO = re.compile(r"(?i)(промокод|по ссылке в описании|ссылк[аеуи] в описании|подписывайтесь|подпишись|"
                    r"telegram-канал|телеграм-канал|не пропустите|розыгрыш|конкурс|скидк[аиу]|"
                    r"спонсор|реклам[аы]|партн[её]р выпуска|переходите по ссылке|"
                    # sponsor reads for subscriptions / marketplaces (kishlak_15m control, ggsel × YouTube Premium:
                    # «доступно лишь с подпиской Премиум… маркетплейс… все ссылки оставили в описании»)
                    r"маркетплейс|подписк\w*\s+(премиум|premium|плюс)|(youtube|ютуб)\s+(premium|премиум)|"
                    r"ссылк\w*\s+(\w+\s+){0,3}в\s+описани|отключ\w*\s+рекламу|промо-код|erid)")


# «А как ты…?» / «Но почему…?» are normal interviewer openers; these are continuations
_QA_BAD_OPENERS = {"и", "ну", "вот", "потому", "поэтому", "то", "это", "значит"}


_DEICTIC = {"ещё", "еще", "это", "этот", "эта", "эти", "там", "тоже", "такое", "так", "тогда", "туда", "оттуда"}
_QA_VAGUE_OPENERS = {"просто", "типа", "тоже", "ещё", "еще", "кстати", "короче"}


def self_contained_question(q: str) -> bool:
    """Owner lesson (MADK1D × VSRAP, clips 1–2): «У тебя есть ещё?» / «Просто прийти в детский
    дом и там…?» mean nothing without the previous minutes. The opening question must carry its
    own topic: at least 6 words, no vague opener, and a short question must not lean on
    «ещё / это / там / тоже…» for what it is about."""
    ws = [re.sub(r"[^\w-]", "", w).lower().replace("ё", "е") for w in q.split()]
    ws = [w for w in ws if w]
    if len(ws) < 6 or (ws and ws[0] in _QA_VAGUE_OPENERS):
        return False
    deictic = {d.replace("ё", "е") for d in _DEICTIC}
    if len(ws) < 10 and any(w in deictic for w in ws):
        return False
    return True


def is_promo(text: str) -> bool:
    """Ad integration / sponsor read / giveaway: never a clip (owner lesson, Alisha × Джарахов)."""
    return bool(_PROMO.search(text))


def qa_ok(c: Dict[str, Any], cfg: StoryConfig) -> bool:
    i = c["sent_range"][0]
    first = c.get("first_sentence", "")
    if not asks_the_guest(first):
        return False
    # a question inside reported speech («И значит, мне пишут: „как ты можешь…?“») is not the
    # interviewer's question: no fragment opener, no quote in the opening sentence
    if c["opener"] in _QA_BAD_OPENERS or re.search(r"[«\"„“]", first):
        return False
    if not self_contained_question(first):
        return False
    return bool(c["questions"]) and c["questions"][0] == i and c["inner_any_q"] == 0 and c["end"] - c["q_end_times"][0] >= cfg.min_answer_sec and \
        (c["next_is_question"] or c["next_gap"] >= cfg.closure_pause)


_EXPLAIN = re.compile(r"\b(потому что|смысл в том|во-первых|во-вторых|главное|важно|нужно|надо|совет|"
                      r"например|то есть|значит|секрет|правило|ошибка|если ты|если вы)\b", re.I)


def story_score(c: Dict[str, Any], cfg: StoryConfig) -> Tuple[float, List[str]]:
    why: List[str] = []
    score = 0.0
    st = cfg.strategy
    if cfg.energy and st in ("viral", "hooks", "story"):
        from .audio_features import candidate_energy
        e = candidate_energy(cfg.energy, c["start"], c["end"])
        c.update(e)
        w = 0.35 if st == "viral" else 0.15
        score += w * max(-1.0, min(3.0, e["energy_peak"]))
        if e["energy_peak"] >= 1.5:
            why.append(f"energy peak {e['energy_peak']:.1f}")
    if st == "educational":
        n = len(_EXPLAIN.findall(c["text"]))
        score += min(1.2, 0.25 * n)
        if n:
            why.append(f"explains ({n} markers)")
    if st == "hooks":
        first = c["text"].split(".")[0].split("?")[0].split("!")[0]
        if 3 <= len(first.split()) <= 12:
            score += 0.4
            why.append("short opening line")
    dur = c["end"] - c["start"]
    if c["opener"] in _FRAGMENT_OPENERS:
        score -= 0.8
        why.append(f"fragment opener «{c['opener']}»")
    else:
        score += 0.3
    if c["questions"]:
        q_end = c["q_end_times"][0]
        answer = c["end"] - q_end
        if q_end - c["start"] <= 0.45 * dur and answer >= cfg.min_answer_sec:
            score += 1.0
            why.append(f"question + answer ({answer:.0f} s)")
        if c["sent_range"][0] in c["questions"] or c["questions"][0] - c["sent_range"][0] <= 1:
            score += 0.3
            why.append("opens with the question")
    if c["next_is_question"] or c["next_gap"] >= cfg.closure_pause:
        score += 0.5
        why.append("ends at a closure")
    else:
        score -= 0.2
    if c["prev_gap"] >= cfg.closure_pause:
        score += 0.2
    if c["short_ratio"] > 0.4:
        score -= 1.0 * c["short_ratio"]
        why.append(f"choppy ({c['short_ratio']:.0%} short sentences)")
    if c["longest_sentence"] >= 20:                    # someone actually tells something
        score += 0.3
    density = c["words"] / max(dur, 1e-6)
    score -= 0.5 * abs(density - 2.5) / 2.5
    score -= (max(0.0, cfg.sweet_min - dur) + max(0.0, dur - cfg.sweet_max)) / 30.0
    why.append(f"{dur:.0f} s")
    return round(score, 6), why


def _diverse(sc_list, limit):
    """Best local candidates without overlaps (the editor should see different moments)."""
    out = []
    for c, sc, why in sc_list:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k, _, _ in out):
            out.append((c, sc, why))
            if len(out) >= limit:
                break
    return out


def select_story_clips(raw_words: Sequence[Dict[str, Any]], max_clips: int,
                       cfg: Optional[StoryConfig] = None,
                       chat: Optional[Callable[..., Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = cfg or StoryConfig()
    words, _ = normalize_words(raw_words)
    meta: Dict[str, Any] = {"candidates_version": CANDIDATES_V3, "local_rank_version": LOCAL_RANK_V3,
                            "params": {k: v for k, v in asdict(cfg).items() if k != "energy"},
                            "audio_energy": bool(cfg.energy), "llm": {"status": "not_opted_in"}}
    cands = build_story_candidates(words, cfg) if words else []
    meta["candidates"] = len(cands)
    if not cands:
        meta["selection_status"] = "no_candidates" if words else "no_words"
        return [], meta
    cands = [c for c in cands if not is_promo(c["text"])]
    brief = ""
    if cfg.promo:
        from .promo import PromoSpec, filter_candidates
        spec = PromoSpec.from_dict(cfg.promo)
        if spec.problems():
            raise ValueError("; ".join(spec.problems()))
        cands, dropped = filter_candidates(cands, spec)
        brief = spec.editor_brief()
        meta["promo"] = {"dropped_by_conditions": dropped, "brief": brief}

    def rank_pick(pool, picked, limit, tag):
        sc_list = []
        for c in pool:
            sc, why = story_score(c, cfg)
            sc_list.append((c, sc, why + ([tag] if tag else [])))
        sc_list.sort(key=lambda t: (-t[1], _ms(t[0]["start"]), _ms(t[0]["end"]), t[0]["id"]))
        for c, sc, why in sc_list:
            if len(picked) >= limit:
                break
            if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k, _, _ in picked):
                picked.append((c, sc, why))
        return sc_list

    chosen: List[Tuple[Dict[str, Any], float, List[str]]] = []
    if cfg.qa_only or cfg.strategy == "qa":
        qa = [c for c in cands if qa_ok(c, cfg)]
        meta["qa_candidates"] = len(qa)
        scored = rank_pick(qa, chosen, max_clips, "")
        if len(chosen) < max_clips:
            # not enough clean question → answer pairs: fill with whole stories (same rules:
            # true sentence start/end, no promo), marked so the owner can tell them apart
            meta["story_fill"] = max_clips - len(chosen)
            scored += rank_pick([c for c in cands if not qa_ok(c, cfg)], chosen, max_clips, "story fill")
    else:
        scored = rank_pick(cands, chosen, max_clips, "")
    extras: Dict[str, Dict[str, Any]] = {}
    if cfg.editor:
        from . import llm_editor
        if chat is None:
            from . import llm_provider
            chat = llm_provider.chat_json
        # ≤ 30 candidates: the answer (one verdict each) must arrive before OpenRouter drops an idle
        # connection (~135 s, seen with 37 candidates and long reasons)
        # the pool must be larger than the request (32 clips → ~42 candidates); short reasons keep
        # the answer fast enough for OpenRouter
        n_pool = max(30, max_clips + 10)
        qa_pool = _diverse([t for t in scored if "story fill" not in t[2]], int(n_pool * 0.75))
        fill_pool = _diverse([t for t in scored if "story fill" in t[2]], n_pool - len(qa_pool))
        pool = [c for c, _, _ in qa_pool + fill_pool]
        verdicts, emeta = llm_editor.rerank(pool, max_clips, chat, pool=len(pool), brief=brief,
                                            strategy=cfg.strategy or ("qa" if cfg.qa_only else ""))
        meta["llm"] = emeta
        if verdicts:
            by_id = {c["id"]: (c, sc, why) for c, sc, why in qa_pool + fill_pool}
            picked: List[Tuple[Dict[str, Any], float, List[str]]] = []
            # Owner 2026-11-13 «моменты некачественные … лучше меньше, но сильных»: only the editor's STRONG
            # picks fill the request; weaker ones only up to a floor of `keep` clips (min 5 by default).
            strong = _f_env("SONYA_CUT_STRONG_SCORE", 75.0)
            keep = min(max_clips, int(_f_env("SONYA_CUT_MIN_CLIPS", 5.0)))
            tiers = ((2.0, max_clips, lambda v: v["self_contained"] and v["score"] >= strong),
                     (1.8, max_clips, lambda v: not v["self_contained"] and v["score"] >= max(strong, 80.0)),
                     (1.5, keep, lambda v: v["self_contained"] and v["score"] >= 50),
                     (1.2, keep, lambda v: not v["self_contained"] and v["score"] >= 60))
            for tier, cap, ok in tiers:
                for v in verdicts:
                    if len(picked) >= cap:
                        break
                    if not ok(v) or v["id"] in extras or (brief and not v.get("ad_fit", True)):
                        continue
                    c, sc, why = by_id[v["id"]]
                    if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k, _, _ in picked):
                        picked.append((c, v["score"] / 100.0 + tier,
                                       why + [f"editor {v['score']:.0f}"
                                              f"{'' if v['self_contained'] else ' (needs context?)'}: {v['reason']}"]))
                        extras[c["id"]] = v
            # the editor rejected too many: top up with the local order up to the floor (never overlapping)
            meta["editor_dropped_weak"] = max(0, max_clips - max(len(picked), keep))
            for c, sc, why in chosen:
                if len(picked) >= keep:
                    break
                if c["id"] not in extras and all(c["end"] <= k["start"] or c["start"] >= k["end"] for k, _, _ in picked):
                    picked.append((c, sc, why + ["local (editor short)"]))
            chosen = picked
            meta["selection_method"] = "editor"
    if brief and len(extras) < len(chosen):
        # brand fit of these clips was checked by the local rules only (editor failed / too few
        # ad_fit picks): say so, the owner/advertiser reviews them by hand
        meta.setdefault("warnings", []).append(
            f"promo: {len(chosen) - len(extras)} clip(s) not checked by the editor for advertiser fit")
    rank = {c["id"]: i + 1 for i, (c, _, _) in enumerate(scored)}
    meta.update(selection_method=meta.get("selection_method", "local"), selection_status="ok",
                ranking_version=LOCAL_RANK_V3,
                selected=[{"id": c["id"], "rank": rank[c["id"]], "score": sc, "start": round(c["start"], 3),
                           "end": round(c["end"], 3), "reason": "; ".join(why), "text": c["text"][:300]}
                          for c, sc, why in chosen])
    segs = [{"start": c["start"], "duration": c["end"] - c["start"], "score": sc,
             "source": "asr_candidate:" + ("editor" if c["id"] in extras else "local-v3"), "candidate_id": c["id"],
             "selection_reason": "local-v3: " + "; ".join(why),
             **({"hook": extras[c["id"]]["hook"], "keywords": extras[c["id"]]["keywords"],
                 "editor_score": extras[c["id"]]["score"]} if c["id"] in extras else {}),
             **({"ad_check": "editor" if c["id"] in extras else "local_only"} if brief else {})}
            for c, sc, why in chosen]
    return segs, meta


def _f_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def select_clips(raw_words: Sequence[Dict[str, Any]], max_clips: int, *, llm_opt_in: bool = False,
                 cfg: Optional[CandidateConfig] = None,
                 chat: Optional[Callable[..., Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """-> (segments for the runner, selection metadata).

    segments: [{start, duration, score, source, candidate_id, selection_reason}]
    """
    cfg = cfg or CandidateConfig()
    words, _ = normalize_words(raw_words)
    meta: Dict[str, Any] = {"candidates_version": CANDIDATES_VERSION, "params": asdict(cfg)}
    if not words:
        meta["selection_status"] = "no_words"
        return [], meta
    cands = build_candidates(words, cfg)
    meta["candidates"] = len(cands)
    if not cands:
        meta["selection_status"] = "no_candidates"
        return [], meta
    by_id = {c.id: c for c in cands}
    local = rank_local(cands, cfg)
    chosen: List[Tuple[Candidate, float, str]] = []
    ranked: List[Tuple[Candidate, float, str]] = []
    method = "local"

    if llm_opt_in:
        from . import llm_provider
        chat = chat or llm_provider.chat_json
        pool = [c for c, _, _ in local[:cfg.max_for_llm]]
        pool.sort(key=lambda c: (c.start, c.id))               # chronological for the model
        try:
            res = chat(build_prompt(pool, max_clips, cfg), prompt_version=PROMPT_VERSION, opt_in=True)
            picks, problems = parse_llm_selection(res.text, {c.id: c for c in pool})
            meta["llm"] = dict(res.telemetry(), problems=problems)
            # Order-independent: the same {id: score} set always ranks the same.
            local_by_id = {c.id: s for c, s, _ in local}
            ranked = sorted(((by_id[i], s, r) for i, s, r in picks),
                            key=lambda t: (-t[1], -local_by_id.get(t[0].id, 0.0), *tie_key(t[0])))
            chosen = pick_non_overlapping(ranked, max_clips)
            if chosen:
                method = "llm"
            else:
                meta["llm"]["fallback_reason"] = "no_valid_ids"
        except llm_provider.LLMDisabled as e:
            meta["llm"] = {"status": "disabled", "reason": str(e)}
        except llm_provider.LLMError as e:
            meta["llm"] = {"status": "failed", "attempts": e.attempts}
        except Exception as e:  # never let the semantic layer kill the run
            logger.warning("[cut_selection] LLM selection error: %s", e)
            meta["llm"] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
    else:
        meta["llm"] = {"status": "not_opted_in"}

    if not chosen:
        chosen = pick_non_overlapping(local, max_clips)
        meta["local_rank_version"] = LOCAL_RANK_VERSION
    meta["selection_method"] = method
    meta["selection_status"] = "ok"
    meta["ranking_version"] = RANKING_VERSION
    meta["tie_break"] = TIE_BREAK
    order = {c.id: i + 1 for i, (c, _, _) in enumerate(ranked if method == "llm" else local)}
    local_scores = {c.id: s for c, s, _ in local}
    meta["selected"] = [{"id": c.id, "rank": order[c.id], "score_source": method, "score": s,
                         "local_score": local_scores[c.id], "start": round(c.start, 3),
                         "end": round(c.end, 3), "reason": r, "text": c.text[:300]} for c, s, r in chosen]
    segs = [{"start": c.start, "duration": c.duration, "score": s, "source": f"asr_candidate:{method}",
             "candidate_id": c.id, "selection_reason": r} for c, s, r in chosen]
    return segs, meta
