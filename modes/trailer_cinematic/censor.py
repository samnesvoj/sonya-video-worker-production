"""
censor.py — SONYA BEEP inside the Trailer: profanity in the narrator's voice is bleeped, in the subtitles it is masked.

Nothing new is detected here: `scripts.shared.profanity` (detector, mask) and `scripts.shared.beep` (intervals, audio
filter, tv beep) are reused as they are.

    narration (sentences with measured TTS times) ──► words with estimated times (the same ones the subtitles use)
        ──► profanity.detect()  ──► refine each hit with the ASR word times of the narrator's own voice (local, free)
        ──► beep.censor_intervals() ──► voice.wav with the bleeps  +  subtitle tokens masked as «пи###ц»

The voice is bleeped BEFORE the mix, so the original word is not in the final audio at all.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scripts.shared import beep as bp
from scripts.shared.profanity import Hit, classify, detect, mask_word

MASK_STYLE = "hash"
EST_PAD = 0.2        # no ASR match: the per-word estimate is proportional to letters, so widen it
CLAMP = 0.015        # keep the bleep inside the sentence (sentence-edge level checks stay valid)

Word = Tuple[str, float, float]


def subtitle_words(vt: Dict[str, Any]) -> Tuple[List[Word], List[Tuple[float, float]]]:
    """(token, start, end) per spoken word, proportional to its length inside the measured sentence — exactly what the
    subtitles use — plus the (t0, t1) of the sentence each token belongs to."""
    sw: List[Word] = []
    bounds: List[Tuple[float, float]] = []
    for s in vt["segments"]:
        for r in s["sentences"]:
            toks = r["text"].split()
            tot = sum(len(x) + 1 for x in toks) or 1
            tt = r["t0"]
            for x in toks:
                d = r["sec"] * (len(x) + 1) / tot
                sw.append((x, round(tt, 3), round(tt + d, 3)))
                bounds.append((r["t0"], r["t1"]))
                tt += d
    return sw, bounds


def find_hits(sw: Sequence[Word], categories=("mat", "rude")) -> List[Hit]:
    return detect([{"word": w, "start": a, "end": b} for w, a, b in sw], categories)


def mask_tokens(sw: Sequence[Word], hits: Sequence[Hit]) -> List[Word]:
    bad = {h.index for h in hits}
    return [(mask_word(w, MASK_STYLE) if i in bad else w, a, b) for i, (w, a, b) in enumerate(sw)]


def without_profanity(text: str) -> str:
    """Script text for the ASR comparison: the bleeped words are not expected to be heard."""
    return " ".join(w for w in text.split() if not classify(w))


def refine_hits(hits: Sequence[Hit], asr_words: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Exact word times from ASR of the narrator's voice when a matching word is near the estimate, else the estimate
    widened. Returns [{index, word, start, end, source}]."""
    out: List[Dict[str, Any]] = []
    used: set = set()
    cand = []
    for j, w in enumerate(asr_words or []):
        txt = str(w.get("word") or w.get("text") or "")
        cand.append((j, txt, float(w["start"]), float(w["end"]), bool(classify(txt))))
    for h in hits:
        mid = (h.start + h.end) / 2
        best = None
        for j, txt, a, b, prof in cand:
            if j in used or abs((a + b) / 2 - mid) > 1.0:
                continue
            same = re.sub(r"\W", "", txt.lower())[:2] == re.sub(r"\W", "", h.word.lower())[:2]
            if prof or (same and abs((a + b) / 2 - mid) <= 0.6):
                d = abs((a + b) / 2 - mid)
                if best is None or d < best[0]:
                    best = (d, j, a, b)
        if best:
            used.add(best[1])
            out.append({"index": h.index, "word": h.word, "start": best[2], "end": best[3], "source": "asr"})
        else:
            out.append({"index": h.index, "word": h.word, "start": h.start - EST_PAD, "end": h.end + EST_PAD,
                        "source": "estimate"})
    return out


@dataclass
class Plan:
    hits: List[Hit]
    refined: List[Dict[str, Any]]
    intervals: List[List[float]]
    words: List[Word]                       # masked subtitle tokens
    cfg: bp.BeepConfig = field(default_factory=bp.BeepConfig)

    def report(self) -> Dict[str, Any]:
        return {"beep_version": bp.BEEP_VERSION, "mask_style": MASK_STYLE, "count": len(self.hits),
                "words": [{"word": r["word"], "masked": mask_word(r["word"], MASK_STYLE), "start": round(r["start"], 3),
                           "end": round(r["end"], 3), "source": r["source"]} for r in self.refined],
                "intervals": self.intervals, "censored_sec": round(sum(b - a for a, b in self.intervals), 3)}


def plan(vt: Dict[str, Any], asr_words: Optional[Sequence[Dict[str, Any]]] = None, total: Optional[float] = None,
         cfg: Optional[bp.BeepConfig] = None) -> Plan:
    cfg = cfg or bp.BeepConfig()
    sw, bounds = subtitle_words(vt)
    hits = find_hits(sw, cfg.categories)
    refined = refine_hits(hits, asr_words)
    spans = []
    for r in refined:
        lo, hi = bounds[r["index"]]
        r["start"] = max(r["start"], lo + CLAMP)
        r["end"] = min(r["end"], hi - CLAMP)
        spans.append(Hit(r["index"], r["word"], r["start"], r["end"], "mat"))
    # pads are applied by censor_intervals; keep the result inside the sentences too
    iv = bp.censor_intervals(spans, total or vt["total"], cfg)
    ivs: List[List[float]] = []
    for a, b in iv:
        owner = next((bounds[r["index"]] for r in refined if r["start"] - 0.2 <= a <= r["end"] + 0.2), None)
        if owner:
            a, b = max(a, owner[0] + CLAMP), min(b, owner[1] - CLAMP)
        if b > a:
            ivs.append([round(a, 3), round(b, 3)])
    return Plan(hits, refined, ivs, mask_tokens(sw, hits), cfg)


def beep_voice(voice_wav: str, intervals: Sequence[Sequence[float]], out_wav: str, cfg: Optional[bp.BeepConfig] = None) -> str:
    """voice.wav → same voice with the tv beep over each interval (BEEP's own filter graph)."""
    cfg = cfg or bp.BeepConfig()
    graph = bp.audio_filter(intervals, cfg)
    # BEEP's beep source is stereo: make the voice stereo first (as the mix does anyway) so amix never downmixes (-3 dB)
    st = str(Path(out_wav).with_suffix(".st.wav"))
    r0 = subprocess.run([bp.ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", voice_wav,
                         "-af", "aformat=channel_layouts=stereo", "-ar", "48000", "-c:a", "pcm_s16le", st], capture_output=True, text=True)
    if r0.returncode != 0:
        raise RuntimeError(f"beep voice failed: {r0.stderr[-400:]}")
    cmd = [bp.ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", st,
           "-filter_complex", graph, "-map", "[aout]", "-ar", "48000", "-c:a", "pcm_s16le", out_wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"beep voice failed: {r.stderr[-400:]}")
    Path(st).unlink(missing_ok=True)
    return out_wav


def asr_words_fn(language: str, cache_dir: Path) -> Callable[[str], List[Dict[str, Any]]]:
    """Local ASR with word times (same desktop backend as the validator). Free."""
    def run(path: str) -> List[Dict[str, Any]]:
        import os
        import sys
        from . import media
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "desktop"))
        from sonya_desktop.asr.backends import select_backend
        from sonya_desktop.compute import detect_hardware
        backend = select_backend(detect_hardware(probe_torch=False), "quality", language=language)
        wav = media.extract_wav16k(path, str(cache_dir / "voice-asr.wav"))
        d = backend.transcribe(wav, "quality", language=language, word_timestamps=True)
        os.remove(wav)
        return list(d.get("words") or [])
    return run
