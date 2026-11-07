"""
profanity.py
============
Russian profanity (мат) detection on ASR words for SONYA BEEP.

    word ──► variants (lowercase, ё→е, homoglyphs/digits → cyrillic, stretched
             letters collapsed, masks "х*й" expanded) ──► anchored root patterns
             with Russian prefixes ──► exceptions ──► category

Every pattern is anchored at the start of the word (after an optional verbal
prefix such as на-/по-/за-/вы-/о-/ни-), so roots hidden inside ordinary words
never fire: «рубля», «корабля», «употреблять», «команда», «застрахуй», «хлеб»,
«небо», «мандарин». Categories:

    mat   — the core roots (хуй, пизд-, еб-/ёб-, бля-, муд-, манда)
    rude  — strong insults usually bleeped too (сука, пидор, гандон, шлюха, залупа)
"""
from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

DETECTOR_VERSION = "ru-profanity-v1"

_PREFIX = (r"(?:а|в|вз|вы|до|за|из|ис|на|над|недо|не|ни|о|об|объ|от|отъ|о?ту|пере|по|под|подъ|пре|при|про|"
           r"раз|разъ|рас|с|съ|въ|взъ|у|долбо|мудо|пизд[ао]?|ху[ея]?)?")
# «с-» is left out for еб-: «себя»/«себе»; the real form is «съеб-»
_PREFIX_EB = _PREFIX.replace("|с|", "|")
_MAT = [
    rf"^{_PREFIX}ху[йеёяию]",                      # хуй, нахуй, похуй, охуенно, нихуя, хуёвый, хуйня
    rf"^{_PREFIX}пизд",                            # пизда, пиздец, спиздил, распиздяй
    rf"^{_PREFIX_EB}[её]б(?:[аеёиоуыюя]|л|н|т|ш|ц|к|ст)",  # ебать, заебал, выёбывается, уёбок, долбоёб(…)
    rf"^{_PREFIX_EB}[её]б$",                       # «долбоёб», «уёб»
    r"^бля(?:$|д|т|ха$|ть|я|ди|ск|х?а?$)",          # бля, блять, блядь, блядский (not «бляха», see exceptions)
    r"^муд(?:ак|ач|ил|о|ень|изм)",                  # мудак, мудила, мудозвон
    r"^манд(?:а|ы|е|у|ой|ёж|еж|ец)$",               # манда (not «мандарин», «команда»)
    r"^[её]п(?:т|ть)а$",                            # ёпта
]
_RUDE = [
    r"^сук(?:а|и|е|у|ой|ам|ами|ах)$",              # not «сук» (a branch, gen. pl.)
    r"^суч(?:ка|ки|ке|ку|кой|ара|ары|ий|ье|онок)",
    r"^пид(?:о|а)?р",                               # пидор, пидар, пидорас
    r"^гандон",
    r"^шлюх",
    r"^залуп",
]
_MAT_RE = [re.compile(p) for p in _MAT]
_RUDE_RE = [re.compile(p) for p in _RUDE]

# Real words that a pattern above would otherwise catch.
_EXCEPTIONS = re.compile(
    r"^(?:бляха|бляхи|бляху|бляхой|бляхе|"           # belt buckle / «бляха-муха» euphemism
    r"мудр\w*|мудре\w*|"                            # мудрый, мудрец
    r"ебонит\w*|"
    r"хуан\w*|хуанхэ|"                              # names / places
    r"сукно|сукна|сукном|сукне|сукон\w*|"           # cloth
    r"сукк\w*|"
    r"потребля\w*|употребля\w*|оскорбля\w*|истребля\w*|"
    r"застрах\w*|страх\w*|"
    r"небо\w*|"
    r"ребят\w*|ребен\w*|ребён\w*"
    r")$")

_HOMO = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "k": "к",
                       "m": "м", "t": "т", "b": "б", "h": "н", "u": "и", "0": "о", "3": "з", "4": "ч",
                       "6": "б", "@": "а", "ё": "е", "1": "и", "i": "и"})
# whole-latin spellings (transliteration, not lookalikes)
_LATIN = [re.compile(p) for p in (r"^bl(?:y|ya|ia)(?:t|d)", r"^bl[iy]a$", r"^(?:na|po|ni|o)?(?:h|kh|x)u(?:i|y|j|e)",
                                  r"^(?:s|ras|za|vy)?pizd", r"^(?:za|vy|na|u|do|po)?(?:e|ye|yo|jo)ba", r"^suka$")]
_MASK = re.compile(r"[*#%$!_.]")
_FILL = "уеёияюоаы"


def _strip(word: str) -> str:
    w = unicodedata.normalize("NFC", word).lower().strip()
    return re.sub(r"^[^\w*#%$@]+|[^\w*#%$@]+$", "", w)


def _variants(word: str) -> List[str]:
    """Normalised forms of one token: plain, homoglyph-mapped, stretch-collapsed, masks filled."""
    base = _strip(word)
    if not base:
        return []
    out = [base.replace("ё", "е")]
    if re.search(r"[a-z0-9@]", base) and re.search(r"[а-яё]", base):   # mixed script: lookalikes
        out.append(base.translate(_HOMO))
    elif re.fullmatch(r"[0-9@а-яё*#%$_.]+", base) and re.search(r"[0-9@]", base):
        out.append(base.translate(_HOMO))
    for w in list(out):
        out.append(re.sub(r"(.)\1{2,}", r"\1", w))           # бляяяять -> блять
        out.append(re.sub(r"(.)\1+", r"\1", w))              # ппиздец -> пиздец
    masked = [w for w in out if _MASK.search(w[1:]) and re.search(r"[а-я]", w)]
    for w in masked:
        stars = [i for i, ch in enumerate(w) if i > 0 and _MASK.match(ch)]
        if 0 < len(stars) <= 3:
            for fill in itertools.product(_FILL + "дзбклнт", repeat=len(stars)):
                lw = list(w)
                for i, ch in zip(stars, fill):
                    lw[i] = ch
                out.append("".join(lw))
    return list(dict.fromkeys(out))


def classify(word: str) -> Optional[str]:
    """'mat' | 'rude' | None for one ASR token. Hyphenated tokens are checked per part."""
    parts = [p for p in re.split(r"[-–—]", _strip(word)) if p] or [word]
    best = None
    for part in parts:
        for v in _variants(part):
            if _EXCEPTIONS.match(v):
                continue
            if re.fullmatch(r"[a-z]+", v):
                if any(p.search(v) for p in _LATIN):
                    return "mat" if not v.startswith("suka") else "rude"
                continue
            if any(p.search(v) for p in _MAT_RE):
                return "mat"
            if any(p.search(v) for p in _RUDE_RE):
                best = "rude"
    return best


@dataclass
class Hit:
    index: int
    word: str
    start: float
    end: float
    category: str


def detect(words: Sequence[Dict[str, object]], categories: Iterable[str] = ("mat", "rude")) -> List[Hit]:
    cats = set(categories)
    hits: List[Hit] = []
    for i, w in enumerate(words):
        text = str(w.get("word") or w.get("text") or "")
        cat = classify(text)
        if cat and cat in cats:
            hits.append(Hit(i, text.strip(), float(w["start"]), float(w["end"]), cat))
    return hits


def mask_word(word: str, style: str = "stars") -> str:
    """Mask one profane word, keeping the punctuation around it.

    style="stars" (BEEP mode, unchanged): «хуйня» -> «х****».
    style="hash" (Trailer): first two letters + ### + last letter: «пиздец» -> «пи###ц»;
        shorter words keep the same idea: 5 letters «блять» -> «бл#ть», 4 letters «суки» -> «с##и»,
        3 letters «хуй» -> «х##», 1–2 letters -> all «#». Hyphenated words are masked part by part.
    """
    if style not in ("stars", "hash"):
        raise ValueError("mask style must be stars|hash")
    if style == "stars":
        m = re.match(r"^(\W*)(\w)(\w*)(\W*)$", word.strip())
        if not m:
            return "*" * len(word.strip())
        return m.group(1) + m.group(2) + "*" * len(m.group(3)) + m.group(4)
    w = word.strip()
    parts = re.findall(r"\w+", w)
    flagged = [p for p in parts if classify(p)] if len(parts) > 1 else parts
    return re.sub(r"\w+", lambda m: _hash_core(m.group(0)) if (m.group(0) in flagged or not flagged) else m.group(0), w)


def _hash_core(w: str) -> str:
    n = len(w)
    if n >= 6:
        return w[:2] + "###" + w[-1]
    if n == 5:
        return w[:2] + "#" + w[-2:]
    if n == 4:
        return w[0] + "##" + w[-1]
    if n == 3:
        return w[0] + "##"
    return "#" * n
