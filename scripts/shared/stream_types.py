"""
stream_types.py
===============
The «spectrum» of stream moment types (parameter ``moment_types``, any combination):

  philosophy  a finished thought of 20–60 s: motivation / life talk («мне никто не верил… а потом поверили»,
              «верьте в себя»). Calm monologue, no jokes, no shouting. Music: only from a safe library — the project
              has none, so clips are rendered WITHOUT music (segment["music"] = "none").
  zavoz       an effective entrance of a guest / the streamer to music: lead-in → music → entrance → reaction.
  chaos       mayhem: things get broken, shouting, sharp loudness spikes and fast motion.
  guest / entrance / joke / conflict   the earlier categories (stream_moments); ``conflict`` is the editor's call.

Local signals only (no model): regex hits on the ASR text + the per-second loudness / visual curves. The Gemini
editor (analysis only, ≤ 5 RUB per stream) has the last word. ``classify`` → {type: score}, 0 = not this type.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ALL_TYPES = ("philosophy", "zavoz", "chaos", "guest", "entrance", "joke", "conflict")
NEW_TYPES = ("philosophy", "zavoz", "chaos")
RU = {"philosophy": "философия", "zavoz": "завоз", "chaos": "хаос", "guest": "гость с подводкой",
      "entrance": "выход в образе", "joke": "пошлая шутка", "conflict": "конфликт"}
TYPE_FROM_RU = {"философия": "philosophy", "завоз": "zavoz", "хаос": "chaos", "гость": "guest", "выход": "entrance",
                "шутка": "joke", "конфликт": "conflict"}
PHILOSOPHY_MAX_SEC = 60.0
MUSIC_NOTE = "none"          # no royalty-free music library in the project → philosophy clips go without music


def normalize_types(value: Any) -> List[str]:
    """list / «a, b» string → known type names in a stable order; unknown names dropped; [] = no filter."""
    if isinstance(value, str):
        value = re.split(r"[,\s;]+", value)
    got = {str(v).strip().lower() for v in (value or [])}
    got |= {TYPE_FROM_RU[g] for g in list(got) if g in TYPE_FROM_RU}
    return [t for t in ALL_TYPES if t in got]


# ── text hits ───────────────────────────────────────────────────────────────

_WISDOM = re.compile(
    r"верьте в себя|верь в себя|поверьте в себя|никто (в меня )?не верил|никто не верил|а потом (все )?поверили|"
    r"не сдавайтесь|не сдавайся|главное не сдаваться|главное (это )?верить|главное не бояться|не бойтесь|не бойся|"
    r"идите к (своей )?(цели|мечте)|своей мечт|ваша мечта|моя мечта|все получится|получится у (каждого|вас|всех)|"
    r"жизнь (одна|коротка|такая)|в жизни (главное|нужно)|путь к успеху|путь (к|в) (цели|мечте)|"
    r"я хочу (вам |всем )?сказать|запомните|помните что|каждый (из вас|человек)|человек (может|способен)|"
    r"сила в (том|тебе)|успех (приходит|это)|смысл жизни|не бывает (поздно|невозможного)|никогда не поздно|"
    r"сомневал|смеялись|говорили что (я|у меня) не|ничего не получится")
_ZAVOZ = re.compile(r"встречайте|на сцену|выходит|заходит|заезжает|появляется|поприветствуем|аплодисменты|"
                    r"завоз|завозим|завез|представляю|прямо сейчас к нам|дамы и господа|барабанная дробь|"
                    r"а теперь|и вот он|и вот она|сюрприз")
_CHAOS = re.compile(r"слома|ломай|ломаю|разнес|разнесу|разбил|разбива|бей|бью|взорв|взрыв|громи|круши|ломают|"
                    r"хаос|бардак|пожар|горит|псих|с ума сош|беги|бегите|держи его|стреля|убью")


def _w(w: Any, k: str, default: Any = None) -> Any:
    """Field of a word given as a dict (ASR) or as a normalized Word object (stream_moments)."""
    if isinstance(w, dict):
        return w.get(k, w.get("word") if k == "text" else default)
    return getattr(w, k, default)


def _norm(w: str) -> str:
    return re.sub(r"[^\w-]", "", str(w).lower().replace("ё", "е"))


def type_hits(words: Sequence[Dict[str, Any]], n_sec: int = 0) -> List[Tuple[float, str, str]]:
    """Extra (time, kind, text) hits: wisdom (a motivational phrase), zavoz_lead, chaos_word. Window of 4 words."""
    seq = [w for w in words if _w(w, "start") is not None]
    toks = [_norm(_w(w, "text", "")) for w in seq]
    out: List[Tuple[float, str, str]] = []
    last: Dict[str, float] = {}
    for i, w in enumerate(seq):
        win = " ".join(t for t in toks[i:i + 4] if t)
        t = round(float(_w(w, "start")), 2)
        for kind, rx, gap in (("wisdom", _WISDOM, 5.0), ("zavoz_lead", _ZAVOZ, 6.0), ("chaos_word", _CHAOS, 3.0)):
            if rx.search(win) and t - last.get(kind, -99.0) >= gap:
                out.append((t, kind, win))
                last[kind] = t
    return out


def music_runs(words: Sequence[Dict[str, Any]], loud: Optional[Sequence[float]], n_sec: int,
               min_len: int = 4, min_loud: float = 0.3) -> List[Tuple[int, int]]:
    """Stretches ≥ min_len s without speech where the sound is still above the stream's median level — music /
    sound effects (an entrance track), not silence. -> [(t0, t1)] seconds."""
    if loud is None or not len(loud):
        return []
    talk = np.zeros(n_sec, dtype=bool)
    for w in words:
        a, b = int(float(_w(w, "start", 0))), int(float(_w(w, "end", _w(w, "start", 0))))
        talk[max(0, a):min(n_sec, b + 1)] = True
    lv = np.zeros(n_sec)
    lv[:min(n_sec, len(loud))] = np.asarray(loud[:n_sec], dtype=np.float64)
    quiet_music = (~talk) & (lv > min_loud)
    runs, s = [], None
    for t in range(n_sec + 1):
        on = t < n_sec and quiet_music[t]
        if on and s is None:
            s = t
        elif not on and s is not None:
            if t - s >= min_len:
                runs.append((s, t))
            s = None
    return runs


# ── per-candidate scores ────────────────────────────────────────────────────

class Context:
    """Per-stream data shared by all candidates."""

    def __init__(self, words, n_sec: int, signals: Dict[str, Sequence[float]], hits):
        self.hits = list(hits or []) + type_hits(words, n_sec)
        self.signals = signals
        self.n_sec = n_sec
        self.music = music_runs(words, signals.get("loud"), n_sec)

    def span(self, a: float, b: float, kinds) -> List[Tuple[float, str, str]]:
        return [h for h in self.hits if a <= h[0] <= b and h[1] in kinds]


def _curve(sig, k, a, b):
    v = sig.get(k)
    if v is None or len(v) <= a:
        return np.zeros(0)
    return np.asarray(v[a:b], dtype=np.float64)


def classify(c: Dict[str, Any], ctx: Context) -> Dict[str, float]:
    """{type: score} for philosophy / zavoz / chaos; 0.0 when the candidate does not look like that type."""
    a, b = float(c["start"]), float(c["end"])
    ia, ib = int(a), int(np.ceil(b))
    dur = float(c.get("tight") or (b - a))
    loud, flash = _curve(ctx.signals, "loud", ia, ib), _curve(ctx.signals, "flash", ia, ib)
    loud_max = float(loud.max()) if len(loud) else 0.0
    n_hot = int((loud > 1.5).sum()) if len(loud) else 0
    emo = ctx.span(a, b, ("laugh", "mat", "shout", "react", "phrase"))
    out = {"philosophy": 0.0, "zavoz": 0.0, "chaos": 0.0}

    # philosophy: ≥ 2 motivational phrases (or 1 strong + length), calm voice, almost no pauses, no jokes / shouts
    wis = ctx.span(a, b, ("wisdom",))
    silent = float(c.get("silent", 0.0))
    if wis and 20.0 <= dur <= PHILOSOPHY_MAX_SEC + 1e-6 and len(emo) <= 2 and loud_max < 3.0 and silent < 0.25 \
            and not any(h[1] in ("laugh", "joke") for h in emo):
        n = len(wis)
        if n >= 2 or (n == 1 and dur >= 30.0 and len(emo) == 0):
            out["philosophy"] = round(1.2 + 0.45 * min(n, 4) + (0.3 if len(emo) == 0 else 0.0)
                                      - abs(dur - 40.0) / 80.0, 3)

    # zavoz: lead-in → music → entrance → reaction. Needs the lead-in AND a reaction AND (music or a greeting)
    lead = ctx.span(a - 30.0, a + 10.0, ("zavoz_lead", "guest_lead"))
    music = [r for r in ctx.music if r[1] >= a - 4.0 and r[0] <= a + 14.0]
    reveal = ctx.span(a - 5.0, a + 10.0, ("reveal",))
    greet = ctx.span(a, b, ("guest_in",))
    react = ctx.span(a, b, ("laugh", "react", "shout", "guest_in", "phrase"))
    if (lead or (reveal and music)) and react and (music or greet):
        out["zavoz"] = round(1.2 + 0.4 * bool(lead) + 0.4 * bool(music) + 0.3 * bool(greet), 3)

    # chaos: loudness spikes + fast motion + dense shouting / mat, or breaking words — at least three of four
    hot_words = ctx.span(a, b, ("chaos_word",))
    dense = max((sum(1 for h in emo if t0 <= h[0] < t0 + 8.0) for t0 in [h[0] for h in emo]), default=0)
    motion = float(flash.max()) if len(flash) else 0.0
    pts = (1 if loud_max >= 3.5 or n_hot >= 6 else 0) + (1 if dense >= 5 else 0) + (1 if hot_words else 0) \
        + (1 if motion >= 2.5 else 0)
    if pts >= 3 or (hot_words and loud_max >= 3.0 and dense >= 3):
        out["chaos"] = round(1.2 + 0.4 * pts + min(0.5, 0.08 * n_hot), 3)
    return out


def best_type(scores: Dict[str, float], wanted: Sequence[str]) -> Optional[str]:
    cand = {k: v for k, v in scores.items() if v > 0 and (not wanted or k in wanted)}
    return max(cand, key=cand.get) if cand else None


# ── editor prompt part ──────────────────────────────────────────────────────

EDITOR_TYPES = {
    "philosophy": "философия — законченное философское/мотивационное рассуждение (20–60 с): «мне никто не верил… а "
                  "потом поверили», «верьте в себя»; цельная мысль с началом и итогом, спокойно, без шуток",
    "zavoz": "завоз — эффектное появление гостя или стримера под музыку: подводка → музыка/вход → реакция",
    "chaos": "хаос — жесть: ломают, кричат, резкие всплески громкости и движения, всё рушится",
    "guest": "гость — приход гостя с подводкой", "entrance": "выход — выход в образе/костюме",
    "joke": "шутка — пошлая/жёсткая шутка и реакция", "conflict": "конфликт — перепалка, ссора"}


def editor_types_line(types: Sequence[str], max_sec: float) -> str:
    if not types:
        return ""
    lines = "; ".join(EDITOR_TYPES[t] for t in types if t in EDITOR_TYPES)
    return (f"Нужны ТОЛЬКО моменты этих типов (поле type строго из списка, иначе «скучно»): {lines}. "
            f"Длина клипа 20–{int(max_sec)} с.\n")
