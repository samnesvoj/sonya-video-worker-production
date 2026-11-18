"""
stream_moments.py
=================
Moment search for long stream VODs (modes/streamer v2).

  1. excitement curve per second = weighted sum of the stream's own z-scored signals
     (scripts/shared/stream_signals): voice loudness + pitch (shouting), speech rate, reaction words
     (mat, laughter, «что?!», «да ладно»), game events (hard cuts, flashes), chat density (when the VOD has chat);
  2. candidates = whole sentences only (cut_selection.true_sentences / boundaries): start at a sentence start,
     end at a sentence end, 25–60 s;
  3. local score: the peak must be inside with time for the reaction after it, the opening 4 s must already
     grab (a hook: voice energy / a reaction / a short punchy line, no dead air), no long silences, no fragment
     openers («и», «ну», «вот»…), no promo / donation-alert reads;
  4. pool of ≤ 30 diverse non-overlapping candidates → Gemini editor (analysis only: ids, scores, hook,
     reason; timestamps never come from the model), with a hard ₽ cap per stream; any editor failure → local order.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VERSION = "stream-moments-v2"
PROMPT_VERSION = "stream-editor-v3"

# signal weights for the excitement curve (all inputs are robust z-scores, clipped below)
WEIGHTS = {"loud": 0.30, "pitch": 0.25, "rate": 0.10, "react": 0.45, "flash": 0.10, "cut": 0.10, "chat": 0.35}


@dataclass
class MomentConfig:
    # owner 2026-11-10: «динамично, быстро понятна суть» — 20–45 s after the jump cuts, the event in the first 3 s
    min_duration: float = 20.0
    max_duration: float = 45.0
    hook_sec: float = 3.0             # the opening that must grab
    tail_sec: float = 3.0             # time after the peak for the reaction / payoff
    skip_intro_sec: float = 120.0     # «стрим скоро начнётся» screens, intro music
    moment_types: List[str] = field(default_factory=list)   # stream_types: philosophy / zavoz / chaos / guest / ...; [] = any
    min_gap_sec: float = 20.0         # between two selected clips
    pool: int = 30                    # candidates shown to the editor (≤ 30: OpenRouter idle limit, CUT_PLAYBOOK)
    editor: bool = False
    budget_rub: float = 5.0           # hard cap of editor spend per stream (owner 2026-11-10: ≤ 5 ₽ per stream)
    rub_per_usd: float = 100.0
    min_editor_score: float = 62.0    # «лучше меньше, но сильных»: below this the editor's pick is dropped
    min_keep: int = 2                 # ... but never fewer than this many clips when the stream has candidates
    jump_gap: float = 0.7             # pauses longer than this are cut out at render (scripts/shared/jumpcut)
    context: str = ""                 # stream title / game, for the editor only
    weights: Dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))
    # owner 2026-11-18 («режим обязан всегда выдавать клипы»): the floor of clips when the stream has speech.
    # None = by stream length (min_clips_for); strict rules first, then relaxed tiers fill up to the floor.
    min_clips: Optional[int] = None

    def __post_init__(self) -> None:
        from .stream_types import PHILOSOPHY_MAX_SEC, normalize_types
        self.moment_types = normalize_types(self.moment_types)
        if "philosophy" in self.moment_types:          # a finished thought runs up to 60 s
            self.max_duration = max(self.max_duration, PHILOSOPHY_MAX_SEC)


# ── 1. excitement curve ─────────────────────────────────────────────────────

def excitement(signals: Dict[str, Sequence[float]], n_sec: int,
               weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Weighted sum of the available per-second signals (missing ones count 0), lightly smoothed (3 s)."""
    w = weights or WEIGHTS
    e = np.zeros(n_sec)
    for k, wk in w.items():
        v = signals.get(k)
        if v is None or not len(v):
            continue
        a = np.zeros(n_sec)
        m = min(n_sec, len(v))
        a[:m] = np.asarray(v[:m], dtype=np.float64)
        if k == "react":
            a = np.minimum(a, 4.0) * 1.2                 # counts → z-like scale
        elif k == "cut":
            a = a * 1.5
        else:
            a = np.clip(a, -1.5, 4.0)
        if k == "pitch":                                 # high pitch only counts when the voice is also up
            loud = np.zeros(n_sec)
            lv = signals.get("loud") or []
            loud[:min(n_sec, len(lv))] = np.asarray(lv[:n_sec], dtype=np.float64)
            a = np.where(loud > 0.0, a, np.minimum(a, 0.0))
        e += wk * a
    return np.convolve(e, np.ones(3) / 3.0, mode="same")


# ── 2. candidates from whole sentences ─────────────────────────────────────

def _sentences(words) -> List[Tuple[int, int]]:
    """True sentences by punctuation; GigaAM streams are often run-on, so a sentence also ends at a 1.2 s pause."""
    from .boundaries import group_sentences
    return group_sentences(words, pause_sec=1.2)


def build_candidates(words, cfg: MomentConfig, hits=None, relaxed: bool = False) -> List[Dict[str, Any]]:
    """relaxed (fallback tier): sentences also end at a 0.7 s pause and a clip may end on any word followed by a
    ≥ 0.6 s pause — still a finished phrase, never cut mid-word."""
    from .cut_selection import _STATEMENT_END, _TRAILING, _first_word, candidate_id
    if relaxed:
        from .boundaries import group_sentences
        sents = group_sentences(words, pause_sec=0.7)
    else:
        sents = _sentences(words)
    end_pause = 0.6 if relaxed else 1.2
    out: List[Dict[str, Any]] = []
    # the stream's opening is skipped (intro screens) — except a «wow» entrance (owner: «начало стрима с вау-выходом»)
    reveals = [t for t, kind, _w in (hits or []) if kind in ("reveal", "laugh", "joke") and t < cfg.skip_intro_sec]
    for i, (s0, _) in enumerate(sents):
        if words[s0].start < cfg.skip_intro_sec and not any(-2.0 <= t - words[s0].start <= 8.0 for t in reveals):
            continue
        for j in range(i, len(sents)):
            e = sents[j][1]
            raw = words[e].end - words[s0].start
            if raw > cfg.max_duration + 15.0:
                break
            dur = _tight(words, s0, e, cfg.jump_gap)     # what the viewer gets after the jump cuts
            if dur > cfg.max_duration + 1e-6 or dur + 1e-6 < cfg.min_duration:
                continue
            end_txt = words[e].text.strip()
            nxt_gap = (words[e + 1].start - words[e].end) if e + 1 < len(words) else 99.0
            # ends at a sentence end: terminal punctuation (not «…»), or a clear pause after it
            if relaxed:
                if not (_STATEMENT_END.search(end_txt) or nxt_gap >= end_pause):
                    continue
            elif _TRAILING.search(end_txt) or not (_STATEMENT_END.search(end_txt) or nxt_gap >= end_pause):
                continue
            out.append({"id": candidate_id(words, s0, e), "start": words[s0].start, "end": words[e].end, "tight": dur,
                        "first": s0, "last": e, "opener": _first_word(words[s0].text),
                        "first_sentence": " ".join(w.text.strip() for w in words[s0:sents[i][1] + 1]),
                        "next_gap": nxt_gap,
                        "prev_gap": (words[s0].start - words[s0 - 1].end) if s0 > 0 else 99.0})
    return out


def _tight(words, s0: int, e: int, gap: float, keep: float = 0.25) -> float:
    """Clip length after cutting every pause > gap down to `keep` (scripts/shared/jumpcut.pieces, word-level)."""
    d = words[e].end - words[s0].start
    for k in range(s0, e):
        g = words[k + 1].start - words[k].end
        if g > gap:
            d -= g - keep
    return round(d, 3)


# ── 3. local score ──────────────────────────────────────────────────────────

_BAD_OPENERS = {"и", "а", "но", "ну", "вот", "потому", "поэтому", "то", "это", "значит", "тоже", "типа", "короче"}
_ALERT = ("спасибо за донат", "спасибо за подписку", "спасибо за фоллоу", "задонатил", "донат от", "подписался",
          "спасибо за сабку", "спасибо за платную")


def _speech_mask(words, n_sec: int) -> np.ndarray:
    m = np.zeros(n_sec)
    for w in words:
        a, b = int(w.start), min(n_sec - 1, int(w.end))
        if 0 <= a < n_sec:
            m[a:b + 1] = 1.0
    return m


def score_candidate(c: Dict[str, Any], e: np.ndarray, speech: np.ndarray, words,
                    cfg: MomentConfig, hits=None) -> Tuple[float, List[str]]:
    """v2 (owner: «интересное, динамично, быстро понятна суть»): the opening carries the most weight — the
    event (a shout, a laugh, a reaction word, a game hit) must already be in the first ~3 s or come within
    ~10 s; a slow build-up, dead air and a long tail lose points."""
    why: List[str] = []
    a, b = int(c["start"]), min(len(e), int(np.ceil(c["end"])))
    seg = e[a:b]
    if len(seg) < 5:
        return -9.0, ["too short"]
    win = np.convolve(seg, np.ones(3) / 3.0, mode="valid") if len(seg) >= 3 else seg
    k = int(np.argmax(win))
    peak, t_peak = float(win[k]), a + k + 1
    lead = t_peak - c["start"]
    score = 0.9 * peak
    why.append(f"peak {peak:.1f} at +{lead:.0f}s")
    # the opening: energy of the first hook_sec + reaction words said in it
    h = e[a:min(b, a + int(np.ceil(cfg.hook_sec)))]
    hook = float(h.max()) if len(h) else 0.0
    score += 0.6 * max(-1.0, min(2.0, hook))
    early = [kind for t, kind, _w in (hits or []) if c["start"] - 0.3 <= t < c["start"] + cfg.hook_sec + 0.5]
    if early:
        score += min(0.6, 0.25 * len(early))
        why.append("reaction in the first 3 s: " + ",".join(sorted(set(early))))
    if hook >= 0.8:
        why.append(f"hook {hook:.1f}")
    # the strongest moment must come soon: no long build-up
    if lead > 12.0:
        score -= min(1.2, (lead - 12.0) / 10.0)
        why.append("slow build-up")
    after = c["end"] - t_peak
    if after < 2.0:
        score -= 0.5
        why.append("peak at the very end")
    elif after > 25.0:
        score -= min(0.6, (after - 25.0) / 20.0)
        why.append("long tail after the peak")
    cat_bonus, cat = _category(c, words, hits, cfg)
    c["category"] = cat
    if cat:
        score += cat_bonus
        why.append(f"category {cat}")
    fs = c["first_sentence"].split()
    if 1 <= len(fs) <= 9 and fs[-1][-1:] in "!?":
        score += 0.35
        why.append("punchy first line")
    elif len(fs) > 25:
        score -= 0.3                                     # a long first sentence is a slow start
    mean = float(seg.mean())
    score += 0.3 * max(-1.0, min(2.0, mean))
    sp = speech[a:b]
    silent = 1.0 - float(sp.mean()) if len(sp) else 1.0
    if silent > 0.4:
        score -= 1.2 * (silent - 0.4)
        why.append(f"{silent:.0%} without speech")
    if c["opener"] in _BAD_OPENERS:
        score -= 0.6
        why.append(f"fragment opener «{c['opener']}»")
    if c["prev_gap"] >= 0.8:
        score += 0.1
    if c["next_gap"] >= 0.8:
        score += 0.1
    dur = float(c.get("tight") or (c["end"] - c["start"]))
    score -= abs(dur - 30.0) / 50.0                      # mild preference for ~30 s (after jump cuts)
    c.update(peak=round(peak, 3), peak_t=float(t_peak), hook=round(hook, 3), mean=round(mean, 3),
             silent=round(silent, 3))
    return round(score, 4), why + [f"{dur:.0f} s"]


# owner 2026-11-10 ~06:00: «подводка → событие → реакция» — explicit categories
CATEGORIES = {"guest": "приход гостя", "entrance": "эффектный выход", "joke": "жёсткая шутка",
              "philosophy": "философия", "zavoz": "завоз", "chaos": "хаос", "conflict": "конфликт"}


def _category(c: Dict[str, Any], words, hits, cfg: MomentConfig) -> Tuple[float, Optional[str]]:
    """Category hint from the reaction hits (the editor has the last word). A guest entrance with its lead-in
    («сейчас к нам зайдёт…») said 8–150 s before gets the lead-in sentence attached (c["lead_in"] = word span):
    the clip plays lead-in → entrance → reaction, the waiting in between is cut."""
    hs = hits or []
    a, b = c["start"], c["end"]
    head = [h for h in hs if a - 0.5 <= h[0] <= a + 10.0]
    inside = [h for h in hs if a <= h[0] <= b]
    if any(h[1] == "guest_in" for h in head):
        g = min(h[0] for h in head if h[1] == "guest_in")
        leads = [h for h in hs if h[1] == "guest_lead" and 0.0 < g - h[0] <= 150.0]
        if leads:
            lt = leads[-1][0]
            if lt < a - 4.0:                             # outside the clip: attach its sentence
                span = _sentence_at(words, lt)
                if span and words[span[1]].end - words[span[0]].start <= 9.0 and words[span[1]].end < a - 1.0:
                    c["lead_in"] = span
            return 0.9, "guest"
        if sum(1 for h in head if h[1] == "guest_in") >= 2 and any(h[1] in ("laugh", "react", "shout") for h in inside):
            return 0.4, "guest"
    if any(h[1] == "reveal" for h in head) and any(h[1] in ("laugh", "react", "shout", "phrase") for h in inside):
        return (0.9 if a < cfg.skip_intro_sec + 600 else 0.6), "entrance"
    if any(h[1] == "joke" for h in inside):
        return 0.6, "joke"
    return 0.0, None


def _sentence_at(words, t: float) -> Optional[Tuple[int, int]]:
    for s0, e0 in _sentences(words):
        if words[s0].start - 0.3 <= t <= words[e0].end + 0.3:
            return s0, e0
    return None


def _overlaps(c, picked, gap):
    return any(c["start"] < k["end"] + gap and c["end"] > k["start"] - gap for k in picked)


def _text(words, c) -> str:
    return " ".join(w.text.strip() for w in words[c["first"]:c["last"] + 1])


# ── 4. editor (analysis only) ───────────────────────────────────────────────

EDITOR_SYSTEM = (
    "Ты — строгий редактор вирусных вертикальных видео (Shorts/Reels/TikTok) из записей русскоязычных игровых "
    "стримов. Зритель листает ленту и видит ТОЛЬКО этот фрагмент (вебка стримера сверху, игра снизу), паузы "
    "вырезаны. Интересно ТОЛЬКО: сильная эмоция (ярость, шок, истерический смех), конфликт или перепалка, смешное "
    "(шутка, абсурд, нелепость), неожиданный поворот, эпичный провал или клатч, короткая яркая история с развязкой, "
    "приход гостя С ПОДВОДКОЙ («сейчас к нам зайдёт…» → гость заходит → первая реакция), эффектный выход или "
    "сюрприз стримера (в образе/костюме, вау-начало стрима, все в шоке и смеются), пошлая или жёсткая шутка и "
    "реакция на неё (смех в комнате, «ты чё», «ахаха»). Схема сильного клипа: подводка → событие → реакция, без "
    "мёртвой тишины. "
    "Главное правило: по первым 2–3 секундам (первой фразе) уже понятно, что происходит и почему стоит смотреть — "
    "клип начинается с события или реакции, а не с разгона. Скучно (отбрасывай, score < 40): обсуждение настроек, "
    "железа, расписания, чтение доната/рекламы, спокойная болтовня, комментирование без события, обрыв мысли, "
    "непонятно без предыстории. Мат сам по себе не плюс и не минус. Будь строгим: из 30 фрагментов хороших обычно "
    "3–6. Ты НЕ придумываешь время — выбираешь номера фраз. Отвечай только JSON.")


def _sentence_list(words, c: Dict[str, Any], pause: float = 1.2) -> List[Tuple[int, int]]:
    """Sentence spans (word indices) inside a candidate, for the editor's start / end choice."""
    from .boundaries import group_sentences
    sub = words[c["first"]:c["last"] + 1]
    return [(c["first"] + a, c["first"] + b) for a, b in group_sentences(sub, pause_sec=pause)]


def editor_messages(pool: List[Dict[str, Any]], want: int, context: str = "",
                    types: Sequence[str] = (), max_sec: float = 45.0) -> List[Dict[str, str]]:
    items = []
    for c in pool:
        sents = c.get("sentences_txt") or [(0.0, c["text"])]
        items.append({"id": c["id"], "signals": c.get("tags", ""),
                      **({"lead_in": c["lead_in_txt"]} if c.get("lead_in_txt") else {}),
                      "before": c.get("before", "")[-220:],
                      "phrases": [f"{i}) +{t:.0f}s {txt[:260]}" for i, (t, txt) in enumerate(sents)]})
    from .stream_types import editor_types_line
    head = (f"Стрим: {context[:200]}\n" if context else "") + editor_types_line(types, max_sec)
    user = (head + f"Нужно найти до {want} СИЛЬНЫХ клипов (лучше меньше, но сильных). Для каждого фрагмента: "
            "phrases — фразы с номером и временем от начала фрагмента; before — что было перед ним (контекст, "
            "в клип не входит). Выбери start (номер фразы, с которой клип начинается — с самого события/реакции, "
            "без разгона) и end (номер фразы, на которой он заканчивается — после развязки), чтобы вышло 20–" + str(int(max_sec)) + " с.\n"
            "Фрагменты:\n" + json.dumps(items, ensure_ascii=False) +
            "\n\nВерни JSON: {\"clips\": [{\"id\": str, \"score\": 0-100, "
            "\"type\": \"эмоция|конфликт|смешное|неожиданное|фейл|клатч|история|гость|выход|шутка|философия|завоз|хаос|скучно\", "
            "\"use_lead_in\": bool (только если у фрагмента есть lead_in — подводка к гостю: ставить её в начало), "
            "\"clear_3s\": bool (понятно ли по первой фразе, что происходит и почему интересно), "
            "\"start\": int, \"end\": int, "
            "\"payoff\": int (номер фразы, где РАЗВЯЗКА/кульминация: исход, ответ, результат, главная реакция), "
            "\"punchline\": int (номер последней ударной фразы-панчлайна, обычно ближе к концу), \"title\": \"крючок-заголовок 3–6 слов по-русски, только то, что "
            "правда есть в клипе\", \"why\": \"почему интересно, до 12 слов\", "
            "\"keywords\": [до 3 слов ИЗ ТЕКСТА]}]} — для КАЖДОГО фрагмента, по убыванию score.")
    return [{"role": "system", "content": EDITOR_SYSTEM}, {"role": "user", "content": user}]


def parse_editor(text: str, pool: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Editor JSON → verdicts; unknown ids / bad sentence numbers are dropped or clamped (fail-closed)."""
    import re
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
            st = int(x.get("start", 0))
            en = int(x.get("end", n - 1))
        except (TypeError, ValueError):
            continue
        st, en = max(0, min(n - 1, st)), max(0, min(n - 1, en))
        if en < st:
            st, en = 0, n - 1
        said = {re.sub(r"[^\w-]", "", w).lower() for w in c["text"].split()}
        kws = [k for k in (x.get("keywords") or [])[:3]
               if isinstance(k, str) and re.sub(r"[^\w-]", "", k).lower() in said]
        typ = str(x.get("type") or "").strip().lower()[:20]

        def _idx(k):
            try:
                i = int(x.get(k))
            except (TypeError, ValueError):
                return None
            return i if st <= i <= en else None
        out.append({"id": x["id"], "score": score, "type": typ, "clear_3s": x.get("clear_3s") is True,
                    "start": st, "end": en, "hook": str(x.get("title") or "").strip()[:60],
                    "reason": str(x.get("why") or "").strip()[:160], "keywords": kws,
                    "self_contained": x.get("clear_3s") is True,
                    "payoff": _idx("payoff"), "punchline": _idx("punchline"),
                    "use_lead_in": x.get("use_lead_in") is True})
    return out


# conservative list prices (USD per 1M tokens) used ONLY for the pre-call forecast; the real cost comes from
# the provider's usage report when it gives one
_PRICE_IN, _PRICE_OUT = 0.50, 3.00


def forecast_rub(messages: List[Dict[str, str]], n_items: int, cfg: MomentConfig) -> float:
    chars = sum(len(m["content"]) for m in messages)
    tokens_in = chars / 2.5                              # Cyrillic JSON: ~2.5 chars per token (pessimistic)
    tokens_out = 110 * n_items + 300
    return round((tokens_in * _PRICE_IN + tokens_out * _PRICE_OUT) / 1e6 * cfg.rub_per_usd, 3)


def _tags(c: Dict[str, Any], sig: Dict[str, Sequence[float]], hits) -> str:
    a, b = int(c["start"]), int(np.ceil(c["end"]))
    tags = []
    kinds: Dict[str, int] = {}
    for t, kind, _w in hits:
        if a <= t < b:
            kinds[kind] = kinds.get(kind, 0) + 1
    names = {"wisdom": "мотивационная фраза", "wisdom_soft": "слово о жизни", "zavoz_lead": "подводка к выходу", "chaos_word": "слово-хаос",
             "laugh": "смех", "mat": "мат", "react": "реакция", "shout": "крик-слово", "phrase": "«да ладно»-фраза",
             "guest_lead": "подводка к гостю", "guest_in": "приветствие гостя", "reveal": "«смотрите/как я вам»",
             "joke": "мат+смех"}
    tags += [f"{names[k]}×{n}" for k, n in sorted(kinds.items())]
    for k, label in (("loud", "громко"), ("pitch", "высокий голос"), ("chat", "чат"), ("flash", "вспышка")):
        v = sig.get(k)
        if v is not None and len(v) > a:
            m = max(v[a:b]) if len(v[a:b]) else 0
            if m >= 2.5:
                tags.append(f"{label} z{m:.0f}")
    tags.append(f"пик на {c['peak_t'] - c['start']:.0f} с")
    if c.get("category"):
        tags.append("похоже на: " + CATEGORIES.get(c["category"], c["category"]))
    return ", ".join(tags)


def _editor_view(c: Dict[str, Any], words, all_sents: List[Tuple[int, int]], cfg: MomentConfig) -> None:
    """What the editor sees: the candidate extended by whole sentences up to ~55 s (so it can cut a slow start
    and still have 20–45 s), numbered phrases, and ~15 s of context before."""
    last = c["last"]
    for s0, e0 in all_sents:
        if s0 <= last:
            continue
        if _tight(words, c["first"], e0, cfg.jump_gap) > cfg.max_duration + 10.0:
            break
        last = e0
    c["view_last"] = last
    view = dict(c, last=last)
    sents = _sentence_list(words, view)
    c["sentences"] = sents
    c["sentences_txt"] = [(words[a].start - c["start"], " ".join(w.text.strip() for w in words[a:b + 1]))
                          for a, b in sents]
    k = c["first"]
    while k > 0 and words[k - 1].start > c["start"] - 15.0:
        k -= 1
    c["before"] = " ".join(w.text.strip() for w in words[k:c["first"]])


_TYPE_TO_CAT = {"гость": "guest", "выход": "entrance", "шутка": "joke", "философия": "philosophy",
                "завоз": "zavoz", "хаос": "chaos", "конфликт": "conflict"}


def _lead_and_category(c: Dict[str, Any], v: Optional[Dict[str, Any]], words) -> Dict[str, Any]:
    """category (editor type, else the local hint) and the guest lead-in [start, end] s when it is used."""
    out: Dict[str, Any] = {}
    typ = (v or {}).get("type") or ""
    cat = _TYPE_TO_CAT.get(typ) or c.get("category")
    if typ:
        out["moment_type"] = typ
    if cat:
        out["category"] = cat
    if cat == "philosophy":
        from .stream_types import MUSIC_NOTE
        out["music"] = MUSIC_NOTE                      # no safe music library in the project: rendered without music
    li = c.get("lead_in")
    if li and (v.get("use_lead_in") if v else cat == "guest") and words[li[1]].end < c["start"] - 1.0:
        out["lead_in"] = [round(float(words[li[0]].start), 3), round(float(words[li[1]].end), 3)]
    return out


def _marks(c0: Optional[Dict[str, Any]], v: Dict[str, Any], words) -> Dict[str, Any]:
    """Editor's payoff / punchline phrase numbers → source seconds (phrase start), for the ad-slot presets."""
    out: Dict[str, Any] = {}
    sents = (c0 or {}).get("sentences") or []
    for k in ("payoff", "punchline"):
        i = v.get(k)
        if i is not None and 0 <= i < len(sents):
            out[f"{k}_t"] = round(float(words[sents[i][0]].start), 3)
    return out


def _apply_trim(c: Dict[str, Any], v: Dict[str, Any], words, cfg: MomentConfig) -> Dict[str, Any]:
    """The editor's start / end phrase → word indices; out of 18–48 s after jump cuts → keep the local span."""
    sents = c.get("sentences") or []
    if not sents:
        return c
    a, b = sents[v["start"]][0], sents[v["end"]][1]
    # too short: grow the end by whole phrases; too long: shrink it
    j = v["end"]
    while _tight(words, a, b, cfg.jump_gap) < cfg.min_duration and j + 1 < len(sents):
        j += 1
        b = sents[j][1]
    while _tight(words, a, b, cfg.jump_gap) > cfg.max_duration + 3.0 and j > v["start"]:
        j -= 1
        b = sents[j][1]
    t = _tight(words, a, b, cfg.jump_gap)
    if not (cfg.min_duration - 2.0 <= t <= cfg.max_duration + 3.0):
        return dict(c, trim_note="editor span rejected (length)")
    if a == c["first"] and b == c["last"]:
        return c
    from .cut_selection import _first_word
    n = dict(c, first=a, last=b, start=words[a].start, end=words[b].end, tight=t, opener=_first_word(words[a].text),
             text=" ".join(w.text.strip() for w in words[a:b + 1]),
             trim_note=f"editor trim: phrases {v['start']}-{j}")
    return n


def min_clips_for(n_sec: float, max_clips: int, cfg: Optional[MomentConfig] = None) -> int:
    """Floor of clips (owner 2026-11-18: ≥ 5–8 from a 30+ min stream; NO_CLIPS only without speech / action).
    cfg.min_clips or STREAMER_MIN_CLIPS (for 30+ min) override the default."""
    import os
    if cfg is not None and cfg.min_clips is not None:
        return max(0, min(max_clips, int(cfg.min_clips)))
    try:
        long_floor = int(os.environ.get("STREAMER_MIN_CLIPS") or 8)
    except ValueError:
        long_floor = 8
    if n_sec >= 1800:
        f = long_floor
    elif n_sec >= 900:
        f = min(long_floor, 6)
    elif n_sec >= 420:
        f = min(long_floor, 4)
    elif n_sec >= 150:
        f = 2
    else:
        f = 1
    return max(0, min(max_clips, f))


def _phrase_start(words, lo: float, hi: float) -> Optional[int]:
    """Word index starting in [lo, hi] right after the longest pause (a phrase start)."""
    best, bg = None, -1.0
    for i, w in enumerate(words):
        if w.start < lo:
            continue
        if w.start > hi:
            break
        g = (w.start - words[i - 1].end) if i > 0 else 99.0
        if g > bg + 1e-6:
            best, bg = i, g
    return best


def _phrase_end(words, s0: int, lo: float, hi: float) -> Optional[int]:
    """Word index ending in [lo, hi] (seconds after words[s0].start): terminal punctuation first, else the
    longest pause after it (a finished phrase, never mid-word)."""
    from .cut_selection import _STATEMENT_END
    t0 = words[s0].start
    best, bkey = None, None
    for e in range(s0, len(words)):
        d = words[e].end - t0
        if d < lo:
            continue
        if d > hi:
            break
        g = (words[e + 1].start - words[e].end) if e + 1 < len(words) else 99.0
        key = (bool(_STATEMENT_END.search(words[e].text.strip())) or g >= 1.0, round(g, 2))
        if bkey is None or key > bkey:
            best, bkey = e, key
    return best


def energy_windows(words, n_sec: int, e: np.ndarray, speech: np.ndarray, want: int,
                   taken: List[Dict[str, Any]], cfg: MomentConfig) -> List[Dict[str, Any]]:
    """Last tier: the loudest / most reactive stretches (audio, chat, reaction words — the excitement curve)
    with speech, snapped to phrase boundaries: start after a pause before the peak, end on a finished phrase
    20–45 s later. Without words: 30 s windows around the loudest peaks that are not silence."""
    from .cut_selection import _first_word, candidate_id
    out: List[Dict[str, Any]] = []
    if n_sec < 8:
        return out
    curve = np.asarray(e, dtype=np.float64).copy()
    if len(words):
        curve = curve + 0.6 * np.convolve(speech, np.ones(9) / 9.0, mode="same")   # talk > dead air
    order = np.argsort(-curve, kind="stable")
    intro = min(cfg.skip_intro_sec, max(0.0, n_sec * 0.1))
    lo_d, hi_d = max(12.0, cfg.min_duration - 5.0), cfg.max_duration
    for t in order:
        if len(out) >= want:
            break
        t = int(t)
        if t < intro and n_sec > 300:
            continue
        probe = {"start": t - 15.0, "end": t + 15.0}
        if _overlaps(probe, taken + out, 4.0):
            continue
        if len(words):
            s0 = _phrase_start(words, t - 12.0, t - 1.0)
            if s0 is None:
                s0 = _phrase_start(words, t - 25.0, t + 3.0)
            if s0 is None:
                continue
            e0 = _phrase_end(words, s0, lo_d, hi_d)
            if e0 is None:
                e0 = _phrase_end(words, s0, 8.0, hi_d + 10.0)
            if e0 is None:
                continue
            c = {"id": candidate_id(words, s0, e0), "start": words[s0].start, "end": words[e0].end,
                 "first": s0, "last": e0, "tight": _tight(words, s0, e0, cfg.jump_gap),
                 "opener": _first_word(words[s0].text), "text": _text(words, {"first": s0, "last": e0})}
        else:
            a = float(max(0, min(n_sec - 30, t - 10)))
            c = {"id": f"energy_{int(a)}", "start": a, "end": min(float(n_sec), a + 30.0), "first": None,
                 "last": None, "tight": min(30.0, n_sec - a), "opener": "", "text": ""}
        if _overlaps(c, taken + out, 4.0):
            continue
        c["peak_t"] = float(t)
        out.append(c)
    return out


def select_moments(raw_words: Sequence[Dict[str, Any]], n_sec: int, signals: Dict[str, Sequence[float]],
                   max_clips: int, cfg: Optional[MomentConfig] = None, hits=None,
                   chat: Optional[Callable[..., Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """-> (segments [{start, duration, score, source, candidate_id, selection_reason, hook?, keywords?}], meta).

    Tiers (owner 2026-11-18: the mode must always give clips when the stream has speech / action):
      0 strict: whole sentences 20–45 s, the requested moment types, the editor's «strong» picks;
      1 the floor (min_clips_for) is filled from the same candidates: other moment types, the editor's rejected
        (not boring first), the local order;
      2 relaxed candidates (0.7 s phrase pauses, 15–55 s, shorter intro skip, closer spacing);
      3 energy windows (audio / chat / reaction peaks with speech, snapped to finished phrases).
    Each clip's "tier" says which rule found it; meta["fallback"] lists what was used."""
    from dataclasses import replace
    from .boundaries import normalize_words
    from .cut_selection import is_promo
    cfg = cfg or MomentConfig()
    words, _ = normalize_words(raw_words)
    floor = min_clips_for(n_sec, max_clips, cfg)
    meta: Dict[str, Any] = {"version": VERSION, "params": {k: v for k, v in asdict(cfg).items()},
                            "signals": sorted(k for k, v in signals.items() if v is not None and len(v)),
                            "llm": {"status": "off"}, "min_clips": floor, "fallback": []}
    e = excitement(signals, n_sec, cfg.weights)
    if not words:
        meta["selection_status"] = "no_words"
        segs = _energy_segments(words, n_sec, e, np.zeros(n_sec), floor, [], cfg, signals, meta)
        return segs, meta
    from . import stream_types as st
    speech = _speech_mask(words, n_sec)
    tctx = st.Context(words, n_sec, signals, hits)
    hits = tctx.hits                                         # + wisdom / zavoz_lead / chaos_word hits
    cands = build_candidates(words, cfg, hits)
    meta["candidates"] = len(cands)
    any_type = not cfg.moment_types or "conflict" in cfg.moment_types

    def _score_all(cs, conf) -> List[Tuple[Dict[str, Any], float, List[str]]]:
        res = []
        for c in cs:
            c["text"] = _text(words, c)
            low = c["text"].lower()
            if is_promo(c["text"]) or sum(low.count(x) for x in _ALERT) >= 2:
                continue
            sc, why = score_candidate(c, e, speech, words, conf, hits)
            ts = st.classify(c, tctx)
            c["type_scores"] = {k: v for k, v in ts.items() if v > 0}
            best = st.best_type(ts, conf.moment_types)
            if best:
                if not c.get("category") or best in conf.moment_types:
                    c["category"] = best
                if best in conf.moment_types:
                    sc = max(sc, ts[best]) + 0.5             # asked for this type: it outranks generic excitement
                    why.append(f"type {best} {ts[best]:.1f}")
            c["typed"] = any_type or c.get("category") in conf.moment_types
            res.append((c, sc, why))
        res.sort(key=lambda t: (-t[1], t[0]["start"], t[0]["end"]))
        return res

    def _diverse(items, taken, gap, limit):
        got: List[Tuple[Dict[str, Any], float, List[str]]] = []
        for c, sc, why in items:
            if not _overlaps(c, [p for p, _, _ in taken] + [p for p, _, _ in got], gap):
                got.append((c, sc, why))
                if len(got) >= limit:
                    break
        return got

    scored = _score_all(cands, cfg)
    typed = [t for t in scored if t[0]["typed"]]
    meta["typed_candidates"] = len(typed)
    limit = max(cfg.pool, max_clips)
    # diverse pool: best per region, never overlapping, spaced. The requested types first; when they cannot reach
    # the floor, the other moments follow them (the editor still sees the request and ranks them)
    pool = _diverse(typed, [], cfg.min_gap_sec, limit)
    if len(pool) < floor and len(typed) < len(scored):
        others = [t for t in scored if not t[0]["typed"]]
        for c, sc, why in others:
            why.append("fallback: other moment type (requested types too few)")
        pool += _diverse(others, pool, cfg.min_gap_sec, limit - len(pool))
        if cfg.moment_types:
            meta["fallback"].append("other_types")
    meta["pool"] = len(pool)
    chosen = [t for t in pool if t[0]["typed"]][:max_clips]
    if len(chosen) < floor:
        chosen = pool[:max(floor, len(chosen))][:max_clips]
    extras: Dict[str, Dict[str, Any]] = {}
    if cfg.editor and pool:
        all_sents = _sentences(words)
        for c, _, _ in pool:
            _editor_view(c, words, all_sents, cfg)
            c["tags"] = _tags(c, signals, hits or [])
            if c.get("lead_in"):
                a0, b0 = c["lead_in"]
                c["lead_in_txt"] = (f"за {c['start'] - words[a0].start:.0f} с до фрагмента: "
                                    + " ".join(w.text.strip() for w in words[a0:b0 + 1])[:200])
        verdicts, emeta = _run_editor([c for c, _, _ in pool], max_clips, cfg, chat)
        meta["llm"] = emeta
        if verdicts:
            by_id = {c["id"]: (c, sc, why) for c, sc, why in pool}
            picked: List[Tuple[Dict[str, Any], float, List[str]]] = []
            reserve: List[Tuple[Dict[str, Any], str]] = []
            rejected = 0
            # strict: the editor's order; boring / not clear in 3 s / weak → out («лучше меньше, но сильных»)
            ranked = sorted(verdicts, key=lambda v: (-(v["clear_3s"] and v["type"] != "скучно"), -v["score"]))

            def _take(v, note=""):
                c0, sc, why = by_id[v["id"]]
                c = _apply_trim(c0, v, words, cfg)
                if _overlaps(c, [p for p, _, _ in picked], cfg.min_gap_sec):
                    return False
                picked.append((c, v["score"] / 100.0 * 3 + sc * 0.2 - (1.0 if note else 0.0),
                               why + [f"editor {v['score']:.0f} {v['type']}{'' if v['clear_3s'] else ' (not clear in 3 s)'}: "
                                      f"{v['reason']}"] + ([c["trim_note"]] if c.get("trim_note") else [])
                               + ([note] if note else [])))
                extras[c["id"]] = v
                return True

            for v in ranked:
                if len(picked) >= max_clips:
                    break
                c0 = by_id[v["id"]][0]
                vcat = _TYPE_TO_CAT.get(v["type"]) or c0.get("category")
                if cfg.moment_types and vcat not in cfg.moment_types and v["type"] not in ("", "скучно") \
                        and not (c0.get("category") in cfg.moment_types):
                    rejected += 1                            # a good clip, but not of the requested type
                    reserve.append((v, "fallback: editor type not requested"))
                    continue
                strong = v["clear_3s"] and v["type"] != "скучно" and v["score"] >= cfg.min_editor_score
                if not strong and len(picked) >= cfg.min_keep:
                    rejected += 1
                    reserve.append((v, "fallback: below the editor's bar"))
                    continue
                _take(v)
            meta["editor_rejected"] = rejected
            if len(picked) < floor and reserve:              # tier 1: the floor from the editor's rejects
                reserve.sort(key=lambda r: (-(r[0]["type"] != "скучно"), -r[0]["clear_3s"], -r[0]["score"]))
                n0 = len(picked)
                for v, note in reserve:
                    if len(picked) >= floor:
                        break
                    _take(v, note)
                if len(picked) > n0:
                    meta["fallback"].append(f"editor_rejects+{len(picked) - n0}")
            if len(picked) < floor:                          # pool moments the editor did not return
                seen = {v["id"] for v in verdicts}
                n0 = len(picked)
                for c, sc, why in pool:
                    if len(picked) >= floor:
                        break
                    if c["id"] in seen or _overlaps(c, [p for p, _, _ in picked], cfg.min_gap_sec):
                        continue
                    picked.append((c, sc * 0.2 - 1.0, why + ["fallback: local order (no editor verdict)"]))
                if len(picked) > n0:
                    meta["fallback"].append(f"local_pool+{len(picked) - n0}")
            chosen = picked
            meta["selection_method"] = "editor"
    for c, _, why in chosen:
        c["tier"] = 1 if any(w.startswith("fallback") for w in why) else 0
    # tier 2: relaxed candidates (shorter pauses end a phrase, 15–55 s, closer spacing)
    if len(chosen) < floor:
        rcfg = replace(cfg, min_duration=max(12.0, cfg.min_duration - 5.0), max_duration=cfg.max_duration + 10.0,
                       skip_intro_sec=min(cfg.skip_intro_sec, 30.0), min_gap_sec=min(cfg.min_gap_sec, 8.0),
                       moment_types=[])
        rc = build_candidates(words, rcfg, hits, relaxed=True)
        meta["relaxed_candidates"] = len(rc)
        more = _diverse(_score_all(rc, rcfg), chosen, rcfg.min_gap_sec, floor - len(chosen))
        for c, sc, why in more:
            c["tier"] = 2
            why.append("fallback: relaxed phrase rules")
        if more:
            meta["fallback"].append(f"relaxed+{len(more)}")
            chosen = chosen + [(c, sc - 2.0, why) for c, sc, why in more]
    # tier 3: energy windows snapped to finished phrases
    if len(chosen) < floor:
        ew = energy_windows(words, n_sec, e, speech, floor - len(chosen), [c for c, _, _ in chosen], cfg)
        for c in ew:
            c["tier"] = 3
            c["text"] = c.get("text") or ""
            chosen.append((c, -3.0 + float(e[min(len(e) - 1, int(c["peak_t"]))]) * 0.1,
                           [f"fallback: energy peak at {c['peak_t']:.0f}s, phrase-snapped"]))
        if ew:
            meta["fallback"].append(f"energy+{len(ew)}")
    meta.update(selection_method=meta.get("selection_method", "local"), selection_status="ok" if chosen else "none",
                curve_p95=round(float(np.percentile(e, 95)), 3) if len(e) else 0.0,
                selected=[{"id": c["id"], "start": round(c["start"], 3), "end": round(c["end"], 3),
                           "score": round(sc, 4), "tier": c.get("tier", 0), "reason": "; ".join(why),
                           "text": (c.get("text") or "")[:300]}
                          for c, sc, why in chosen])
    c0_by = {p["id"]: p for p, _, _ in pool}
    segs = [{"start": c["start"], "duration": c["end"] - c["start"], "score": sc,
             "source": "stream_moment:" + ("editor" if c["id"] in extras else
                                           "local" if c.get("tier", 0) <= 1 else "fallback"),
             "tier": c.get("tier", 0),
             "candidate_id": c["id"], "selection_reason": "; ".join(why), "peak_t": c.get("peak_t"),
             "tight_sec": c.get("tight"),
             **(_lead_and_category(c, extras.get(c["id"]), words) if c.get("first") is not None else {}),
             **({"hook": extras[c["id"]]["hook"], "keywords": extras[c["id"]]["keywords"],
                 "editor_score": extras[c["id"]]["score"], **_marks(c0_by.get(c["id"]), extras[c["id"]], words)}
                if c["id"] in extras else {})}
            for c, sc, why in chosen]
    return segs, meta


def _energy_segments(words, n_sec, e, speech, want, taken, cfg, signals, meta) -> List[Dict[str, Any]]:
    """No ASR words at all: clips only when the audio has action (loudness that is not flat silence)."""
    loud = np.asarray(signals.get("loud") or [], dtype=np.float64)
    db = np.asarray(signals.get("db") or [], dtype=np.float64)
    if want <= 0 or not len(loud):
        return []
    if len(db):          # raw level: ≥ 10 % of the video is audible (game / music / shouting, not silence)
        if float((db > -50.0).mean()) < 0.10:
            return []
    elif float(loud.max()) < 1.0:
        return []
    ew = energy_windows(words, n_sec, e, speech, want, taken, cfg)
    if ew:
        meta["fallback"].append(f"energy_no_speech+{len(ew)}")
        meta["selection_status"] = "ok"
    return [{"start": c["start"], "duration": c["end"] - c["start"], "score": -3.0, "tier": 3,
             "source": "stream_moment:fallback", "candidate_id": c["id"], "peak_t": c["peak_t"],
             "tight_sec": c["tight"], "selection_reason": f"fallback: energy peak at {c['peak_t']:.0f}s (no speech)"}
            for c in ew]


def _run_editor(pool: List[Dict[str, Any]], want: int, cfg: MomentConfig,
                chat: Optional[Callable[..., Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {"prompt_version": PROMPT_VERSION, "pool": len(pool), "budget_rub": cfg.budget_rub}
    em = lambda p: editor_messages(p, want, cfg.context, cfg.moment_types, cfg.max_duration)   # noqa: E731
    msgs = em(pool)
    while len(pool) > want and forecast_rub(msgs, len(pool), cfg) > cfg.budget_rub:
        pool = pool[:-1]                                    # shrink until the forecast fits the cap
        msgs = em(pool)
    meta["forecast_rub"] = forecast_rub(msgs, len(pool), cfg)
    if meta["forecast_rub"] > cfg.budget_rub:
        meta["status"] = "skipped_budget"
        return [], meta
    if chat is None:
        from . import llm_provider
        chat = llm_provider.chat_json
    try:
        res = chat(msgs, prompt_version=PROMPT_VERSION, opt_in=True, timeout=180.0,
                   extra={"reasoning": {"effort": "low"}})
        tel = res.telemetry() if hasattr(res, "telemetry") else {}
        meta.update(tel)
        cost = tel.get("cost_usd")
        meta["cost_rub"] = round(float(cost) * cfg.rub_per_usd, 3) if cost is not None else None
        if cost is None and tel.get("input_tokens") is not None:     # Timeweb reports tokens, not money
            est = (float(tel["input_tokens"]) * _PRICE_IN + float(tel.get("output_tokens") or 0) * _PRICE_OUT) / 1e6
            meta["cost_rub_estimate"] = round(est * cfg.rub_per_usd, 3)
        verdicts = parse_editor(res.text, {c["id"]: c for c in pool})
    except Exception as exc:                                # never let the editor kill the run
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return [], meta
    verdicts.sort(key=lambda v: (-(v["self_contained"]), -v["score"]))
    meta.update(status="ok", verdicts=len(verdicts))
    return verdicts, meta
