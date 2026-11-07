"""
evidence.py — ties an overlay to what was actually said. No model, no network: plain text checks.

Everything shown on screen has to be traceable here: a quote must exist in the ASR words of the clip
(its time comes from those words, not from the LLM), numbers must be spoken in the quote, and the
words of a definition must come from the speaker. Hedged speech ("кажется", "не помню") is never evidence.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[^\W\d_]+", re.U)

QUOTE_MIN_RATIO = 0.86          # tolerance for ASR/LLM spelling differences, not for rephrasing
HEDGES = (
    "кажется", "вроде", "наверное", "не помню", "не уверен", "не уверена", "говорят", "якобы", "по слухам",
    "может быть", "если не ошибаюсь", "точно не знаю", "что-то там", "что то там", "кто-то рассказывал",
    "i think", "not sure", "i don't remember", "they say", "supposedly", "rumor",
)


def norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е").replace("–", "-").replace("—", "-")


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(norm(text))


@dataclass
class Segment:
    id: str
    start: float
    end: float
    text: str


@dataclass
class QuoteMatch:
    start: float
    end: float
    ratio: float
    probability: Optional[float]
    segment_id: Optional[str]
    text: str


def clip_local_words(words: Iterable[Dict[str, Any]], clip_start: float, clip_end: float) -> List[Dict[str, Any]]:
    """ASR words inside the clip, re-timed to clip-local seconds."""
    out = []
    for w in words or []:
        s, e = float(w.get("start", 0.0)), float(w.get("end", w.get("start", 0.0)))
        if e <= clip_start or s >= clip_end:
            continue
        out.append({**w, "start": max(0.0, s - clip_start), "end": min(clip_end, e) - clip_start})
    return out


def build_segments(words: List[Dict[str, Any]], *, pause_sec: float = 0.7, max_sec: float = 14.0) -> List[Segment]:
    """Sentences (ASR punctuation or a pause) with stable ids: the `source_segment_id` of the manifest."""
    segs: List[Segment] = []
    cur: List[Dict[str, Any]] = []

    def flush() -> None:
        if cur:
            segs.append(Segment(f"s{len(segs) + 1:04d}", float(cur[0]["start"]), float(cur[-1]["end"]),
                                " ".join(str(w.get("word", "")).strip() for w in cur)))
            cur.clear()

    for w in words:
        if cur and (float(w["start"]) - float(cur[-1]["end"]) > pause_sec
                    or float(w["end"]) - float(cur[0]["start"]) > max_sec):
            flush()
        cur.append(w)
        if re.search(r"[.!?…]$", str(w.get("word", "")).strip()):
            flush()
    flush()
    return segs


def find_quote(words: List[Dict[str, Any]], quote: str, segments: Optional[List[Segment]] = None,
               min_ratio: float = QUOTE_MIN_RATIO) -> Optional[QuoteMatch]:
    """Locate `quote` in the ASR words. None = this was not said: the overlay has no evidence."""
    q = tokens(quote)
    if len(q) < 2 or not words:
        return None
    flat: List[Tuple[str, int]] = [(t, i) for i, w in enumerate(words) for t in tokens(str(w.get("word", "")))]
    if not flat:
        return None
    best: Optional[Tuple[float, int, int]] = None
    n = len(flat)
    for size in {len(q), max(2, len(q) - 1), len(q) + 1, len(q) + 2}:
        for a in range(0, max(1, n - size + 1)):
            window = [t for t, _ in flat[a:a + size]]
            if q[0][:3] != window[0][:3] and q[-1][:3] != window[-1][:3]:
                continue
            r = difflib.SequenceMatcher(None, q, window, autojunk=False).ratio()
            if best is None or r > best[0] + 1e-9:
                best = (r, a, a + size - 1)
    if best is None or best[0] < min_ratio:
        return None
    ws = words[flat[best[1]][1]:flat[best[2]][1] + 1]
    probs = [float(w["probability"]) for w in ws if w.get("probability") is not None]
    start, end = float(ws[0]["start"]), float(ws[-1]["end"])
    seg_id = None
    for s in segments or []:
        if s.start - 0.05 <= start <= s.end + 0.05:
            seg_id = s.id
            break
    return QuoteMatch(start, end, best[0], sum(probs) / len(probs) if probs else None, seg_id,
                      " ".join(str(w.get("word", "")).strip() for w in ws))


def has_hedge(text: str) -> bool:
    t = " ".join(tokens(text))
    return any(" ".join(tokens(h)) in t for h in HEDGES)


def context_around(words: List[Dict[str, Any]], start: float, end: float, pad_sec: float = 1.0) -> str:
    """The quote plus a second either side: a claim hedged in the same breath is still hedged."""
    return " ".join(str(w.get("word", "")) for w in words
                    if float(w["end"]) >= start - pad_sec and float(w["start"]) <= end + pad_sec)


# ── numbers ──────────────────────────────────────────────────────────────────

_EXACT = {
    "ноль": 0, "нуль": 0, "один": 1, "одна": 1, "одно": 1, "одного": 1, "первый": 1, "первом": 1, "первого": 1, "первой": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "двумя": 2, "второй": 2, "втором": 2, "второго": 2,
    "три": 3, "трех": 3, "трем": 3, "тремя": 3, "третий": 3, "третьем": 3, "третьего": 3, "третьей": 3,
    "четыре": 4, "четырех": 4, "четырем": 4, "четвертый": 4, "четвертом": 4, "четвертого": 4, "четвертой": 4,
    "пять": 5, "пяти": 5, "пятый": 5, "пятом": 5, "пятого": 5, "пятой": 5,
    "шесть": 6, "шести": 6, "шестой": 6, "шестом": 6, "шестого": 6,
    "семь": 7, "семи": 7, "седьмой": 7, "седьмом": 7, "седьмого": 7,
    "восемь": 8, "восьми": 8, "восьмой": 8, "восьмом": 8, "восьмого": 8,
    "девять": 9, "девяти": 9, "девятый": 9, "девятом": 9, "девятого": 9,
    "десять": 10, "десяти": 10, "десятый": 10, "десятом": 10, "десятого": 10,
    "сто": 100, "ста": 100, "сотый": 100, "сотом": 100, "двести": 200, "двухсот": 200, "триста": 300, "трехсот": 300,
    "четыреста": 400, "четырехсот": 400, "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "шестисот": 600,
    "семьсот": 700, "семисот": 700, "восемьсот": 800, "восьмисот": 800, "девятьсот": 900, "девятисот": 900,
    "полтора": 1.5, "полторы": 1.5, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_PREFIX = (("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14), ("пятнадцат", 15),
           ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18), ("девятнадцат", 19), ("двадцат", 20),
           ("тридцат", 30), ("сорок", 40), ("пятьдесят", 50), ("пятидесят", 50), ("шестьдесят", 60),
           ("шестидесят", 60), ("семьдесят", 70), ("семидесят", 70), ("восемьдесят", 80), ("восьмидесят", 80),
           ("девяност", 90))
_SCALE_PREFIX = (("тысяч", 1e3), ("тыс", 1e3), ("миллион", 1e6), ("млн", 1e6), ("миллиард", 1e9), ("млрд", 1e9),
                 ("триллион", 1e12), ("thousand", 1e3), ("million", 1e6), ("billion", 1e9))


def _word_value(tok: str) -> Optional[float]:
    if tok in _EXACT:
        return float(_EXACT[tok])
    for p, v in _PREFIX:
        if tok.startswith(p):
            return float(v)
    return None


def _scale(tok: str) -> Optional[float]:
    for p, v in _SCALE_PREFIX:
        if tok.startswith(p):
            return v
    return None


def extract_numbers(text: str) -> Set[float]:
    """Numbers spoken in digits or words (RU/EN), with and without their scale:
    «сто двадцать тысяч» → {120, 120000}; «в две тысячи двадцать третьем» → {2, 2000, 2023, 23}."""
    found: Set[float] = set()
    total = cur = 0.0
    active = False

    def close() -> None:
        nonlocal total, cur, active
        if active:
            if cur:
                found.add(round(cur, 6))
            found.add(round(total + cur, 6))
        total = cur = 0.0
        active = False

    prev_digit_group: Optional[float] = None
    for t in tokens(re.sub(r"(?<=\d) ?, ?(?=\d)", ",", text or "")):      # Whisper writes «2 ,5»
        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", t):
            v = float(t.replace(",", "."))
            if prev_digit_group is not None and re.fullmatch(r"[0-9]{3}", t):   # «120 000»
                merged = prev_digit_group * 1000 + v
                found.add(merged)
                cur = merged
                prev_digit_group = merged
                active = True
                continue
            close()
            cur, active = v, True
            prev_digit_group = v
            continue
        prev_digit_group = None
        sc = _scale(t)
        if sc:
            if not active:
                cur, active = 1.0, True
            found.add(round(cur, 6))
            total += (cur or 1.0) * sc
            found.add(round(total, 6))
            cur = 0.0
            continue
        v = _word_value(t)
        if v is not None:
            active = True
            cur += v
            continue
        close()
    close()
    return found


def numbers_in_display(text: str) -> Set[float]:
    """Numbers printed on a card (digits only; a card never spells numbers out)."""
    out: Set[float] = set()
    for m in re.finditer(r"[0-9]+(?:[  ][0-9]{3})*(?:[.,][0-9]+)?", text or ""):
        out.add(float(m.group(0).replace(" ", "").replace(" ", "").replace(",", ".")))
    return out


def number_supported(value: float, spoken: Set[float], tolerance: float = 0.001) -> bool:
    return any(abs(value - s) <= max(abs(s) * tolerance, 1e-9) for s in spoken)


# ── content words ────────────────────────────────────────────────────────────

_STOP = {"это", "этот", "эта", "эти", "того", "чтобы", "который", "которая", "которое", "такой", "такое", "также",
         "very", "that", "this", "with", "from", "have", "были", "было", "будет", "можно", "нужно", "когда", "если",
         "процесс", "просто", "очень", "теперь", "поэтому"}


def _stem(tok: str) -> str:
    return tok[:5]


def content_stems(text: str) -> List[str]:
    return [_stem(t) for t in tokens(text) if len(t) >= 4 and t not in _STOP and not t[0].isdigit()]


def coverage(claim: str, source: str) -> float:
    """Share of the claim's content words that the source says too (1.0 when the claim has none)."""
    c = content_stems(claim)
    if not c:
        return 1.0
    have = set(content_stems(source))
    return sum(1 for s in c if s in have) / len(c)
