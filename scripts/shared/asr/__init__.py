"""
scripts.shared.asr - the ASR entry point of the worker (contract: scripts/shared/asr_contract.py, docs/ASR_INTERFACE.md).

    transcribe(audio_path, language="ru") -> {"text", "language", "backend", "model", "segments", "words"}

Owner decision 2026-11-07 (docs/OWNER_DECISIONS.md): GigaAM v3 (`v3_e2e_rnnt`) is the PRIMARY ASR for virality and beep,
on CUDA as on the Mac mini; faster-whisper is the fallback only.
  SONYA_ASR_BACKEND   gigaam (default) | faster_whisper_cuda | faster_whisper_cpu
  SONYA_ASR_FALLBACK  1 = if GigaAM cannot run (weights/runtime missing) use faster-whisper and mark the result
                      `fallback_from`; default 0 = raise (a silent quality drop is worse than a clear failure)
GigaAM is Russian-only: for another explicit language the whisper path is used (as in the Desktop backend order);
language None/"auto" means SONYA_ASR_LANGUAGE (default "ru").
Nothing here downloads a model: GigaAM reads $MODELS_LOCAL_DIR/gigaam/, faster-whisper honours SONYA_ASR_OFFLINE=1.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from scripts.shared.asr_contract import backend_name

log = logging.getLogger(__name__)
GIGAAM_LANGS = ("ru",)


def _whisper(audio_path: str, language: Optional[str], backend: str) -> Dict[str, Any]:
    from scripts.shared.transcription.transcriber import Transcriber
    cuda = backend == "faster_whisper_cuda"
    t = Transcriber(device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8", language=language)
    res = t.transcribe(audio_file=audio_path, language=language)
    model = os.environ.get("SONYA_WHISPER_MODEL", "").strip() or "base"
    return dict(res, backend=backend, model=f"faster-whisper-{model}")


def transcribe(audio_path: str, language: Optional[str] = "ru", backend: Optional[str] = None,
               progress: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    name = backend or backend_name()
    lang = (language or "auto").strip().lower()
    if lang == "auto":                         # SONYA's sources are Russian unless the job says otherwise
        lang = os.environ.get("SONYA_ASR_LANGUAGE", "ru").strip().lower() or "ru"
    if name == "gigaam" and lang in GIGAAM_LANGS:
        from scripts.shared.asr import gigaam_backend as g
        try:
            return g.transcribe(audio_path, progress=progress)
        except g.GigaAMUnavailable as e:
            if os.environ.get("SONYA_ASR_FALLBACK", "0") != "1":
                raise
            log.warning("GigaAM unavailable (%s) - falling back to faster-whisper (SONYA_ASR_FALLBACK=1)", e)
            res = _whisper(audio_path, lang, "faster_whisper_cuda" if _cuda() else "faster_whisper_cpu")
            return dict(res, fallback_from="gigaam", fallback_reason=str(e))
    if name == "gigaam":                       # non-Russian request: GigaAM cannot do it
        name = "faster_whisper_cuda" if _cuda() else "faster_whisper_cpu"
    return _whisper(audio_path, lang, name)


def _cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
