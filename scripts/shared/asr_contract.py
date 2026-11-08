"""
asr_contract.py - the ASR interface SONYA's modes rely on, as executable checks. Pure Python (no torch), importable everywhere.

Owner decision (docs/OWNER_DECISIONS.md): the primary ASR is GigaAM v3 (`v3_e2e_rnnt`); faster-whisper is the fallback only. The BACKEND
(loading GigaAM on CUDA, the -150 ms start correction) is implemented by the Mac mini branch; this module does not implement it - it
defines what any backend must return and how the worker verifies it, so both sides test against the same rules (docs/ASR_INTERFACE.md).

Result schema = the format of the proven Mac mini Desktop backend (desktop/sonya_desktop/asr/backends.py `_assemble`), a superset of what
scripts/shared/enhancers/sonya_enhancer.py reads from Transcriber (`text`, `language`, `words`):
  {"text": str, "language": "ru", "backend": "gigaam", "model": "gigaam-v3_e2e_rnnt-ts3",
   "segments": [{"start","end","text","words":[{"word","start","end"}]}],
   "words":    [{"word": str, "start": float, "end": float}, ...]}
`model` carries the post-processing version (`-tsN`): ts3 = start lead 0.15 s + minimum word length 0.12 s + punctuation-token merge. All times are
seconds from the start of the audio; a word text may carry a leading space (Cut strips it).
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

BACKENDS = ("gigaam", "faster_whisper_cuda", "faster_whisper_cpu")      # names as in the Mac mini backend table
DEFAULT_BACKEND = "gigaam"
GIGAAM_MODEL = "v3_e2e_rnnt"
RNNT_START_SHIFT_SEC = 0.15          # GigaAM RNNT word starts run ~150 ms late (CUT_PLAYBOOK); GIGAAM_START_LEAD in the reference backend
MIN_WORD_SEC = 0.12                  # GIGAAM_MIN_WORD: near-zero-length RNNT words get a real extent so Cut does not drop them
MIN_POSTPROC_TAG = 3                 # `-ts3` or newer in meta model name = shift + min length + punct merge applied
GIGAAM_MODEL_RE = r"^gigaam-v3_e2e_rnnt-ts(\d+)$"
GIGAAM_FILES = {                     # relative to the models root ($MODELS_LOCAL_DIR); keys in modes/*/mode.yaml + release/models.required.json
    "gigaam_v3_e2e_rnnt_ckpt": "gigaam/v3_e2e_rnnt.ckpt",
    "gigaam_v3_e2e_rnnt_tokenizer": "gigaam/v3_e2e_rnnt_tokenizer.model",
}
FRAME_SEC = 0.04                     # one GigaAM encoder frame
OVERLAP_TOLERANCE_SEC = 0.08         # two RNNT frames; Cut's normalize_words trims small overlaps


def backend_name(env: Mapping[str, str] = os.environ) -> str:
    name = (env.get("SONYA_ASR_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"SONYA_ASR_BACKEND must be one of {BACKENDS}, got {name!r}")
    return name


def expected_model_files(models_root: Path, backend: Optional[str] = None) -> List[Path]:
    """Files that must already be on disk (verified by SHA-256 from our S3) before the worker may accept jobs."""
    return [models_root / rel for rel in GIGAAM_FILES.values()] if (backend or backend_name()) == "gigaam" else []


def validate_result(result: Any, audio_duration: Optional[float] = None) -> List[str]:
    """-> list of problems (empty = the result honours the interface)."""
    p: List[str] = []
    if not isinstance(result, dict):
        return ["result is not a dict"]
    if not isinstance(result.get("text"), str):
        p.append("text must be a string")
    if not isinstance(result.get("language"), str) or not result.get("language"):
        p.append("language must be a non-empty string")
    words = result.get("words")
    if not isinstance(words, list):
        return p + ["words must be a list"]
    if not words and (result.get("text") or "").strip():
        p.append("text is non-empty but words is empty (Cut needs word timestamps)")
    prev_s = prev_e = None
    for i, w in enumerate(words):
        if not isinstance(w, dict) or not isinstance(w.get("word"), str) or not w["word"].strip():
            p.append(f"word[{i}]: missing/empty 'word'"); continue
        try:
            s, e = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            p.append(f"word[{i}]: start/end must be numbers"); continue
        if not (math.isfinite(s) and math.isfinite(e)) or s < 0 or e <= s:
            p.append(f"word[{i}] {w['word']!r}: need 0 <= start < end (got {s}, {e}) - Cut DROPS words with end <= start")
        if audio_duration is not None and e > audio_duration + 0.5:
            p.append(f"word[{i}]: end {e:.2f}s is beyond the audio ({audio_duration:.2f}s)")
        if prev_s is not None and s < prev_s - 1e-9:
            p.append(f"word[{i}]: not sorted by start")
        if prev_e is not None and s < prev_e - OVERLAP_TOLERANCE_SEC:
            p.append(f"word[{i}] {w['word']!r}: overlaps the previous word by {prev_e - s:.3f}s (> {OVERLAP_TOLERANCE_SEC}s)")
        prev_s, prev_e = s, e
        if len(p) > 20:
            p.append("... (more problems truncated)"); break
    backend, model = result.get("backend"), result.get("model")
    if backend not in BACKENDS:
        p.append(f"backend must be one of {BACKENDS}")
    if not isinstance(model, str) or not model:
        p.append("model must be a non-empty string")
    elif backend == "gigaam":
        m = re.match(GIGAAM_MODEL_RE, model)
        if not m:
            p.append(f"model must look like 'gigaam-{GIGAAM_MODEL}-tsN' (got {model!r})")
        elif int(m.group(1)) < MIN_POSTPROC_TAG:
            p.append(f"post-processing tag ts{m.group(1)} < ts{MIN_POSTPROC_TAG}: the -150 ms start correction / minimum word length are missing")
    if not isinstance(result.get("segments"), list):
        p.append("segments must be a list")
    return p


def cut_compat(words: List[Dict[str, Any]], max_dropped_ratio: float = 0.01) -> Dict[str, Any]:
    """Run the words through the SAME code Cut uses (boundaries.normalize_words -> sentences -> candidates). Reports how many words Cut would
    silently drop. RNNT can emit several short words inside one 40 ms frame; unless the backend gives every word a minimum length and
    de-overlaps, Cut drops them."""
    from scripts.shared.boundaries import group_sentences, normalize_words
    from scripts.shared.cut_selection import CandidateConfig, build_candidates

    kept, stats = normalize_words(words)
    total = max(1, stats["input"])
    dropped = stats["dropped_malformed"] + stats["dropped_overlap"]
    sentences = group_sentences(kept, 0.7) if kept else []
    cands = build_candidates(kept, CandidateConfig()) if kept else []
    ratio = dropped / total
    return {"input": stats["input"], "kept": len(kept), "dropped": dropped, "dropped_ratio": round(ratio, 4), "trimmed_overlap": stats["trimmed_overlap"],
            "sentences": len(sentences), "candidates": len(cands), "ok": ratio <= max_dropped_ratio}
