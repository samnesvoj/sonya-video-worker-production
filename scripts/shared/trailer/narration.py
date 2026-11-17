"""
Trailer narration WITHOUT a voice (owner 2026-11-17: «голоса пока не делать» — the narrator voice is chosen later).

The narrator's text is told by the subtitles on the picture while the film plays with its own sound. This module gives
every segment the same shape voice.synthesize() returns — {"sec", "wav", "words": [(w, s, e)]} — so the edit plan,
subtitles, shorts and render do not care whether a voice exists: when the owner picks a TTS, the pipeline swaps
narration.reading() for voice.synthesize() and nothing else changes (the timeline is the «slot» for the voice).

Reading timing: READ_PACE words/s (the viewer reads bold mid-size subtitles; the owner's voice formula is 2.9–3.4,
reading is a bit slower), a word's time grows with its length, a sentence end gets a short beat, a segment never
shorter than MIN_SEG s.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

NARRATION_VERSION = "trailer-reading-v1"
READ_PACE = 2.7          # words per second of on-screen narration
MIN_SEG = 1.6
SENT_BEAT = 0.22         # extra time after . ! ? … inside a segment
TAIL = 0.25              # the last word stays a moment before the next segment
TOKEN_RE = re.compile(r"[\w]+(?:[-'][\w]+)*([^\w\s]*)", re.U)


def word_weights(text: str) -> List[Tuple[str, float, bool]]:
    """(word, weight, ends_sentence): weight ∝ reading effort (short words are glanced, long ones read)."""
    out = []
    for m in TOKEN_RE.finditer(text or ""):
        w = re.sub(r"[^\w'-]", "", m.group(0))
        if not w:
            continue
        out.append((w, 0.55 + 0.09 * min(len(w), 14), bool(re.search(r"[.!?…]", m.group(1) or ""))))
    return out


def segment_timing(text: str, pace: float = READ_PACE) -> Dict[str, Any]:
    ws = word_weights(text)
    if not ws:
        return {"sec": MIN_SEG, "wav": None, "words": []}
    n = len(ws)
    sents = sum(1 for _, _, e in ws[:-1] if e)
    body = max(MIN_SEG - TAIL, n / pace)
    unit = (body - SENT_BEAT * sents) / sum(w for _, w, _ in ws)
    t = 0.0
    words = []
    for w, k, end in ws:
        d = k * unit
        words.append((w, round(t, 3), round(t + d * 0.92, 3)))
        t += d + (SENT_BEAT if end else 0.0)
    return {"sec": round(max(MIN_SEG, words[-1][2] + TAIL), 3), "wav": None, "words": words}


def reading(segments: Sequence[Dict[str, Any]], pace: float = READ_PACE) -> Dict[str, Any]:
    """segments [{"n", "text"}] → the voice.synthesize() shape, no audio."""
    out = {s["n"]: segment_timing(s["text"], pace) for s in segments}
    words = sum(len(v["words"]) for v in out.values())
    sec = sum(v["sec"] for v in out.values())
    return {"segments": out, "version": NARRATION_VERSION,
            "tts": {"engine": "none", "mode": "subtitles", "pace": round(words / max(0.1, sec), 2), "words": words,
                    "voice_sec": round(sec, 2)}}
