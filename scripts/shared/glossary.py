"""
glossary.py
===========
Fix ASR mishearings of names (bloggers, streamers, artists) in word timestamps
before selection and subtitles.

Entries come from scripts/shared/data/names_glossary.json (owner-confirmed) plus
an optional per-run list. A global entry fires only if one of its context words
occurs within ±40 words, so «Вавилон» stays Babylon in a history talk but becomes
«Эвелон» next to «стример / вписка». Inflection and case are kept:
«Вавилоном» → «Эвелоном», «ВАВИЛОН» → «ЭВЕЛОН».
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GLOSSARY_PATH = Path(__file__).resolve().parent / "data" / "names_glossary.json"
WINDOW = 40


def load_entries(extra: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    try:
        base = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8")).get("entries", [])
    except (OSError, ValueError):
        base = []
    return list(base) + [dict(e) for e in (extra or [])]


def _norm(t: str) -> str:
    return re.sub(r"[^\w]", "", t.lower()).replace("ё", "е")


def _recase(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst.lower() if src.islower() else dst


def apply(words: List[Dict[str, Any]], extra: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """New list of word dicts with corrected text (timestamps untouched); 'glossary' notes on changed words."""
    entries = load_entries(extra)
    if not entries or not words:
        return words
    normed = [_norm(str(w.get("word", ""))) for w in words]
    out = []
    for i, w in enumerate(words):
        text = str(w.get("word", ""))
        for e in entries:
            stem = e["wrong"].lower().replace("ё", "е")
            n = normed[i]
            if not n.startswith(stem) or len(n) - len(stem) > 3:      # stem + short case ending only
                continue
            ctx = [c.lower() for c in e.get("context") or []]
            if ctx:
                around = normed[max(0, i - WINDOW):i + WINDOW + 1]
                if not any(a.startswith(c) for a in around for c in ctx):
                    continue
            m = re.search(re.escape(e["wrong"][:1]) + r"\w*", text, re.IGNORECASE)
            if not m:
                continue
            core = m.group(0)
            new_core = _recase(core, e["right"] + core[len(e["wrong"]):])
            text = text[:m.start()] + new_core + text[m.end():]
            w = dict(w, word=text, glossary=f"{core}→{new_core}")
            break
        out.append(w)
    return out
