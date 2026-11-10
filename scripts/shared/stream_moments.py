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

VERSION = "stream-moments-v1"
PROMPT_VERSION = "stream-editor-v1"

# signal weights for the excitement curve (all inputs are robust z-scores, clipped below)
WEIGHTS = {"loud": 0.30, "pitch": 0.25, "rate": 0.10, "react": 0.45, "flash": 0.10, "cut": 0.10, "chat": 0.35}


@dataclass
class MomentConfig:
    min_duration: float = 25.0
    max_duration: float = 60.0
    hook_sec: float = 4.0             # the opening that must grab
    tail_sec: float = 3.0             # time after the peak for the reaction / payoff
    skip_intro_sec: float = 120.0     # «стрим скоро начнётся» screens, intro music
    min_gap_sec: float = 20.0         # between two selected clips
    pool: int = 30                    # candidates shown to the editor (≤ 30: OpenRouter idle limit, CUT_PLAYBOOK)
    editor: bool = False
    budget_rub: float = 10.0          # hard cap of editor spend per stream (owner: ≤ 10 ₽ per stream)
    rub_per_usd: float = 100.0
    weights: Dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))


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


def build_candidates(words, cfg: MomentConfig) -> List[Dict[str, Any]]:
    from .cut_selection import _STATEMENT_END, _TRAILING, _first_word, candidate_id
    sents = _sentences(words)
    out: List[Dict[str, Any]] = []
    for i, (s0, _) in enumerate(sents):
        if words[s0].start < cfg.skip_intro_sec:
            continue
        for j in range(i, len(sents)):
            e = sents[j][1]
            dur = words[e].end - words[s0].start
            if dur > cfg.max_duration + 1e-6:
                break
            if dur + 1e-6 < cfg.min_duration:
                continue
            end_txt = words[e].text.strip()
            nxt_gap = (words[e + 1].start - words[e].end) if e + 1 < len(words) else 99.0
            # ends at a sentence end: terminal punctuation (not «…»), or a clear pause after it
            if _TRAILING.search(end_txt) or not (_STATEMENT_END.search(end_txt) or nxt_gap >= 1.2):
                continue
            out.append({"id": candidate_id(words, s0, e), "start": words[s0].start, "end": words[e].end,
                        "first": s0, "last": e, "opener": _first_word(words[s0].text),
                        "first_sentence": " ".join(w.text.strip() for w in words[s0:sents[i][1] + 1]),
                        "next_gap": nxt_gap,
                        "prev_gap": (words[s0].start - words[s0 - 1].end) if s0 > 0 else 99.0})
    return out


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
                    cfg: MomentConfig) -> Tuple[float, List[str]]:
    why: List[str] = []
    a, b = int(c["start"]), min(len(e), int(np.ceil(c["end"])))
    seg = e[a:b]
    if len(seg) < 5:
        return -9.0, ["too short"]
    # strongest 3 s inside
    win = np.convolve(seg, np.ones(3) / 3.0, mode="valid") if len(seg) >= 3 else seg
    k = int(np.argmax(win))
    peak, t_peak = float(win[k]), a + k + 1
    score = 1.0 * peak
    why.append(f"peak {peak:.1f} at +{t_peak - c['start']:.0f}s")
    # the reaction needs time after the peak
    after = c["end"] - t_peak
    if after < cfg.tail_sec:
        score -= 0.6
        why.append("peak at the very end")
    # hook: the opening already grabs
    h = e[a:min(b, a + int(cfg.hook_sec))]
    hook = float(h.mean()) if len(h) else 0.0
    # saturating: a lively start is enough, starting ON the peak loses the build-up
    score += 0.35 * max(-1.0, min(1.2, hook))
    if hook >= 0.8:
        why.append(f"hook {hook:.1f}")
    lead = t_peak - c["start"]
    if 3.0 <= lead <= 0.6 * (c["end"] - c["start"]):
        score += 0.3                                     # a short build-up, then the peak, then the reaction
        why.append("build-up → peak → reaction")
    fs = c["first_sentence"].split()
    if 2 <= len(fs) <= 10 and fs[-1][-1:] in "!?":
        score += 0.3
        why.append("punchy first line")
    # sustained interest, not one spike in a dull minute
    mean = float(seg.mean())
    score += 0.35 * max(-1.0, min(2.0, mean))
    # dead air
    sp = speech[a:b]
    silent = 1.0 - float(sp.mean()) if len(sp) else 1.0
    if silent > 0.45:
        score -= 1.2 * (silent - 0.45)
        why.append(f"{silent:.0%} without speech")
    if c["opener"] in _BAD_OPENERS:
        score -= 0.5
        why.append(f"fragment opener «{c['opener']}»")
    if c["prev_gap"] >= 0.8:
        score += 0.15
    if c["next_gap"] >= 0.8:
        score += 0.15
    dur = c["end"] - c["start"]
    score -= abs(dur - 40.0) / 80.0                      # mild preference for ~40 s
    c.update(peak=round(peak, 3), peak_t=float(t_peak), hook=round(hook, 3), mean=round(mean, 3),
             silent=round(silent, 3))
    return round(score, 4), why + [f"{dur:.0f} s"]


def _overlaps(c, picked, gap):
    return any(c["start"] < k["end"] + gap and c["end"] > k["start"] - gap for k in picked)


def _text(words, c) -> str:
    return " ".join(w.text.strip() for w in words[c["first"]:c["last"] + 1])


# ── 4. editor (analysis only) ───────────────────────────────────────────────

EDITOR_SYSTEM = (
    "Ты — редактор коротких вертикальных видео (Shorts/Reels/TikTok) из записей русскоязычных игровых стримов. "
    "Тебе дают фрагменты расшифровки речи стримера и метки сигналов (крик, смех, реакция чата, вспышки в игре). "
    "Зритель увидит ТОЛЬКО этот фрагмент: вебку стримера сверху и игру снизу. Хороший клип: яркая реакция "
    "(крик, смех, ярость, шок), эпичный или провальный игровой момент, смешная перепалка или короткая история, "
    "понятные без остального стрима; первые 3 секунды цепляют. Плохой клип: чтение доната/рекламы, скучный "
    "разговор о настройках, обрыв мысли, непонятно о чём. Мат сам по себе не минус. Отвечай только JSON.")


def editor_messages(pool: List[Dict[str, Any]], want: int) -> List[Dict[str, str]]:
    items = [{"id": c["id"], "dur": round(c["end"] - c["start"]), "signals": c.get("tags", ""),
              "text": c["text"][:700]} for c in pool]
    user = (f"Нужно выбрать {want} лучших клипов. Фрагменты:\n" + json.dumps(items, ensure_ascii=False) +
            "\n\nВерни JSON: {\"clips\": [{\"id\": str, \"score\": 0-100, \"self_contained\": bool, "
            "\"hook\": \"заголовок до 7 слов\", \"keywords\": [до 3 слов ИЗ ТЕКСТА], \"reason\": \"до 8 слов\"}]} "
            "— для КАЖДОГО фрагмента, по убыванию score.")
    return [{"role": "system", "content": EDITOR_SYSTEM}, {"role": "user", "content": user}]


# conservative list prices (USD per 1M tokens) used ONLY for the pre-call forecast; the real cost comes from
# the provider's usage report when it gives one
_PRICE_IN, _PRICE_OUT = 0.50, 3.00


def forecast_rub(messages: List[Dict[str, str]], n_items: int, cfg: MomentConfig) -> float:
    chars = sum(len(m["content"]) for m in messages)
    tokens_in = chars / 2.5                              # Cyrillic JSON: ~2.5 chars per token (pessimistic)
    tokens_out = 90 * n_items + 300
    return round((tokens_in * _PRICE_IN + tokens_out * _PRICE_OUT) / 1e6 * cfg.rub_per_usd, 3)


def _tags(c: Dict[str, Any], sig: Dict[str, Sequence[float]], hits) -> str:
    a, b = int(c["start"]), int(np.ceil(c["end"]))
    tags = []
    kinds: Dict[str, int] = {}
    for t, kind, _w in hits:
        if a <= t < b:
            kinds[kind] = kinds.get(kind, 0) + 1
    names = {"laugh": "смех", "mat": "мат", "react": "реакция", "shout": "крик-слово", "phrase": "«да ладно»-фраза"}
    tags += [f"{names[k]}×{n}" for k, n in sorted(kinds.items())]
    for k, label in (("loud", "громко"), ("pitch", "высокий голос"), ("chat", "чат"), ("flash", "вспышка")):
        v = sig.get(k)
        if v is not None and len(v) > a:
            m = max(v[a:b]) if len(v[a:b]) else 0
            if m >= 2.5:
                tags.append(f"{label} z{m:.0f}")
    tags.append(f"пик на {c['peak_t'] - c['start']:.0f} с")
    return ", ".join(tags)


def select_moments(raw_words: Sequence[Dict[str, Any]], n_sec: int, signals: Dict[str, Sequence[float]],
                   max_clips: int, cfg: Optional[MomentConfig] = None, hits=None,
                   chat: Optional[Callable[..., Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """-> (segments [{start, duration, score, source, candidate_id, selection_reason, hook?, keywords?}], meta)."""
    from .boundaries import normalize_words
    from .cut_selection import is_promo
    cfg = cfg or MomentConfig()
    words, _ = normalize_words(raw_words)
    meta: Dict[str, Any] = {"version": VERSION, "params": {k: v for k, v in asdict(cfg).items()},
                            "signals": sorted(k for k, v in signals.items() if v is not None and len(v)),
                            "llm": {"status": "off"}}
    if not words:
        meta["selection_status"] = "no_words"
        return [], meta
    e = excitement(signals, n_sec, cfg.weights)
    speech = _speech_mask(words, n_sec)
    cands = build_candidates(words, cfg)
    meta["candidates"] = len(cands)
    scored = []
    for c in cands:
        c["text"] = _text(words, c)
        low = c["text"].lower()
        if is_promo(c["text"]) or sum(low.count(x) for x in _ALERT) >= 2:
            continue
        sc, why = score_candidate(c, e, speech, words, cfg)
        scored.append((c, sc, why))
    scored.sort(key=lambda t: (-t[1], t[0]["start"], t[0]["end"]))
    # diverse pool: best per region, never overlapping, spaced
    pool: List[Tuple[Dict[str, Any], float, List[str]]] = []
    for c, sc, why in scored:
        if not _overlaps(c, [p for p, _, _ in pool], cfg.min_gap_sec):
            pool.append((c, sc, why))
            if len(pool) >= max(cfg.pool, max_clips):
                break
    meta["pool"] = len(pool)
    chosen = pool[:max_clips]
    extras: Dict[str, Dict[str, Any]] = {}
    if cfg.editor and pool:
        for c, _, _ in pool:
            c["tags"] = _tags(c, signals, hits or [])
        verdicts, emeta = _run_editor([c for c, _, _ in pool], max_clips, cfg, chat)
        meta["llm"] = emeta
        if verdicts:
            by_id = {c["id"]: (c, sc, why) for c, sc, why in pool}
            picked: List[Tuple[Dict[str, Any], float, List[str]]] = []
            for v in verdicts:                              # editor order, its score blended with the local one
                if len(picked) >= max_clips:
                    break
                c, sc, why = by_id[v["id"]]
                if v["score"] < 35 or _overlaps(c, [p for p, _, _ in picked], cfg.min_gap_sec):
                    continue
                picked.append((c, v["score"] / 100.0 * 3 + sc * 0.25,
                               why + [f"editor {v['score']:.0f}: {v['reason']}"]))
                extras[c["id"]] = v
            for c, sc, why in pool:                         # editor rejected too many → local order
                if len(picked) >= max_clips:
                    break
                if c["id"] not in extras and not _overlaps(c, [p for p, _, _ in picked], cfg.min_gap_sec):
                    picked.append((c, sc, why + ["local (editor short)"]))
            chosen = picked
            meta["selection_method"] = "editor"
    meta.update(selection_method=meta.get("selection_method", "local"), selection_status="ok" if chosen else "none",
                curve_p95=round(float(np.percentile(e, 95)), 3) if len(e) else 0.0,
                selected=[{"id": c["id"], "start": round(c["start"], 3), "end": round(c["end"], 3),
                           "score": round(sc, 4), "reason": "; ".join(why), "text": c["text"][:300]}
                          for c, sc, why in chosen])
    segs = [{"start": c["start"], "duration": c["end"] - c["start"], "score": sc,
             "source": "stream_moment:" + ("editor" if c["id"] in extras else "local"),
             "candidate_id": c["id"], "selection_reason": "; ".join(why), "peak_t": c.get("peak_t"),
             **({"hook": extras[c["id"]]["hook"], "keywords": extras[c["id"]]["keywords"],
                 "editor_score": extras[c["id"]]["score"]} if c["id"] in extras else {})}
            for c, sc, why in chosen]
    return segs, meta


def _run_editor(pool: List[Dict[str, Any]], want: int, cfg: MomentConfig,
                chat: Optional[Callable[..., Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from . import llm_editor
    meta: Dict[str, Any] = {"prompt_version": PROMPT_VERSION, "pool": len(pool), "budget_rub": cfg.budget_rub}
    msgs = editor_messages(pool, want)
    while len(pool) > want and forecast_rub(msgs, len(pool), cfg) > cfg.budget_rub:
        pool = pool[:-1]                                    # shrink until the forecast fits the cap
        msgs = editor_messages(pool, want)
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
        verdicts = llm_editor.parse(res.text, {c["id"]: c for c in pool})
    except Exception as exc:                                # never let the editor kill the run
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return [], meta
    verdicts.sort(key=lambda v: (-(v["self_contained"]), -v["score"]))
    meta.update(status="ok", verdicts=len(verdicts))
    return verdicts, meta
