# -*- coding: utf-8 -*-
"""
transcriber.py
==============
Audio transcription module for Boosta backend.

Reconstructed from: transcriber.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary.

Confirmed recovered elements:
  - Class Transcriber  (4 methods from binary)
  - OpenAI Whisper API  (original engine: aOpenAI, aOPENAI_API_KEY,
    averbose_json, atimestamp_granularities, aclient, acreate)
  - Model name: "whisper-1" (OpenAI model ID)
  - Chunk limit: 25 MB  (docstring: "Splits audio into chunks < 25MB")
  - Audio extraction: ffmpeg -vn -acodec libmp3lame -> .mp3
  - ffprobe: format=duration, noprint_wrappers=1:nokey=1
  - Steps: "Step 2: Separating Audio Track..." / "Step 3: Transcribing Content..."
  - Output keys: seg_text, seg_start, seg_end, w_text, w_start, w_end
  - dotenv: loads OPENAI_API_KEY from .env

This reconstruction adds faster-whisper as the default local engine
(faster_whisper 1.2.1 is present in the bundled runtime) with the
OpenAI API as the optional cloud fallback — identical output format.

Output format compatible with clipper.py (reads transcript["words"]):
{
    "text":     "full transcript text",
    "language": "en",
    "segments": [{"start": 0.0, "end": 2.5, "text": "...",
                  "words": [{"word": "...", "start": 0.0, "end": 0.4}]}],
    "words":    [{"word": "...", "start": 0.0, "end": 0.4}]
}

Dependencies:
    faster-whisper >= 0.9.0   (local engine, bundled in backend/python/)
    openai >= 1.0.0            (optional cloud fallback)
    ffmpeg + ffprobe           (bundled as ffmpeg.exe / ffprobe.exe)
    python-dotenv              (optional, for .env loading)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dotenv
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# faster-whisper  (primary local engine)
# ---------------------------------------------------------------------------
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel as _WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _WhisperModel = None  # type: ignore

# ---------------------------------------------------------------------------
# OpenAI API  (optional cloud fallback — original binary engine)
# ---------------------------------------------------------------------------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI as _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None  # type: ignore

__all__ = ["Transcriber"]

# ---------------------------------------------------------------------------
# Constants decoded from binary
# ---------------------------------------------------------------------------
DEFAULT_MODEL       = "whisper-1"           # OpenAI model name (binary: uwhisper-1)
LOCAL_MODEL_MAP     = {                     # OpenAI name -> faster-whisper equivalent
    "whisper-1":    "base",
    "whisper-2":    "medium",
    "large":        "large-v2",
    "large-v2":     "large-v2",
    "medium":       "medium",
    "small":        "small",
    "base":         "base",
    "tiny":         "tiny",
}
CHUNK_LIMIT_MB      = 25.0                  # binary docstring: "chunks < 25MB"
CHUNK_DURATION_S    = 600                   # 10 min per chunk (safe for base model)
AUDIO_SAMPLE_RATE   = 16000                 # Whisper expects 16 kHz
AUDIO_CODEC         = "libmp3lame"          # binary: alibmp3lame


# ===========================================================================
# Transcriber
# ===========================================================================

class Transcriber:
    """
    Transcribes audio from video/audio file using OpenAI Whisper API.
    Handles large files by splitting.

    Primary engine: faster-whisper (local, bundled in backend/python/)
    Fallback engine: OpenAI Whisper API (requires OPENAI_API_KEY in .env)
    """

    def __init__(
        self,
        model:        str = DEFAULT_MODEL,
        device:       str = "cpu",
        compute_type: str = "int8",
        ffmpeg_path:  str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        language:     Optional[str] = None,
    ) -> None:
        self.model_name   = model
        self.device       = device
        self.compute_type = compute_type
        self.language     = language
        self._model       = None   # lazy-loaded

        # Locate bundled ffmpeg/ffprobe
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg   = self._find_bin(ffmpeg_path,  script_dir)
        self.ffprobe  = self._find_bin(ffprobe_path, script_dir)

        # OpenAI client (original engine)
        self._openai_client = None
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[WARN] Warning: OPENAI_API_KEY not found in .env file.")
        elif OPENAI_AVAILABLE and _OpenAI is not None:
            try:
                self._openai_client = _OpenAI(api_key=api_key)
            except Exception as exc:
                print(f"[WARN] OpenAI client init failed: {exc}")

        # Pre-warm faster-whisper if available
        if FASTER_WHISPER_AVAILABLE:
            self._model = self._load_faster_whisper(model, device, compute_type, script_dir)
        else:
            if not self._openai_client:
                print("[WARN] No transcription engine available "
                      "(install faster-whisper or set OPENAI_API_KEY).")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_duration(self, audio_file: str) -> float:
        """Get duration of video/audio file in seconds using ffprobe."""
        if not os.path.exists(audio_file):
            print(f"[WARN] File not found: {audio_file}")
            return 0.0
        try:
            cmd = [
                self.ffprobe,
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_file,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
            return float(out.strip())
        except Exception as exc:
            print(f"[WARN] Error getting duration: {exc}")

        # Fallback: ffprobe with JSON format
        try:
            cmd2 = [
                self.ffprobe, "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json",
                audio_file,
            ]
            out2 = subprocess.check_output(cmd2, stderr=subprocess.DEVNULL, timeout=15)
            data = json.loads(out2)
            return float(data["format"]["duration"])
        except Exception:
            pass

        print(f"[WARN] Could not determine duration, returning original file")
        return 0.0

    def split_audio(
        self,
        audio_file:      str,
        chunk_duration:  int   = CHUNK_DURATION_S,
        chunk_limit_mb:  float = CHUNK_LIMIT_MB,
    ) -> List[str]:
        """
        Splits audio into chunks < 25MB using ffmpeg.
        Returns list of file paths to chunks.
        """
        print(f"[INFO] Processing audio for splitting: {audio_file}")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb <= chunk_limit_mb:
            return [audio_file]

        print(f"[INFO] {file_size_mb:.1f} MB > 25MB. Splitting...")

        duration = self.get_duration(audio_file)
        if duration <= 0:
            return [audio_file]

        total_chunks = math.ceil(duration / chunk_duration)
        print(f"[INFO] Splitting into {total_chunks} chunks of {chunk_duration}s each")

        tmp_dir   = tempfile.mkdtemp(prefix="boosta_chunks_")
        base_name = Path(audio_file).stem
        chunks:   List[str] = []

        for i in range(total_chunks):
            ss          = i * chunk_duration
            chunk_name  = f"{base_name}_part{i+1:03d}.mp3"
            chunk_path  = os.path.join(tmp_dir, chunk_name)

            print(f"[INFO] Extracting chunk {i+1}/{total_chunks}: {ss}s - {ss+chunk_duration}s")

            cmd = [
                self.ffmpeg, "-y",
                "-ss", str(ss),
                "-i", audio_file,
                "-t", str(chunk_duration),
                "-vn",
                "-acodec", AUDIO_CODEC,
                "-ar",     str(AUDIO_SAMPLE_RATE),
                "-q:a",    "5",
                "-loglevel", "error",
                chunk_path,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                if os.path.exists(chunk_path):
                    chunks.append(chunk_path)
            except Exception as exc:
                print(f"[WARN] Failed to extract chunk {i+1}: {exc}")

        return chunks if chunks else [audio_file]

    def transcribe(
        self,
        audio_file:  str,
        language:    Optional[str] = None,
        beam_size:   int = 5,
        vad_filter:  bool = True,
    ) -> Dict:
        """
        Transcribes audio from video/audio file using OpenAI Whisper API.
        Handles large files by splitting.

        Returns dict compatible with clipper.py:
        {
            "text":     str,
            "language": str,
            "segments": [{"start", "end", "text", "words": [{"word", "start", "end"}]}],
            "words":    [{"word", "start", "end"}]
        }
        """
        lang = language or self.language

        if not os.path.exists(audio_file):
            print(f"[WARN] File not found: {audio_file}")
            return _empty_transcript(lang)

        # Step 2: Separating Audio Track
        print(f"[INFO] Step 2: Separating Audio Track...")
        audio_path, is_temp = self._extract_audio(audio_file)

        # Step 3: Transcribing Content
        print(f"[INFO] Step 3: Transcribing Content...")
        print(f"[INFO] Preparing audio for transcription...")

        try:
            # Check if chunking is needed
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > CHUNK_LIMIT_MB:
                result = self._transcribe_chunked(audio_path, lang, beam_size, vad_filter)
            else:
                result = self._transcribe_single(audio_path, lang, beam_size, vad_filter)

            return result

        except Exception as exc:
            print(f"[WARN] Transcription failed: {exc}")
            traceback.print_exc()
            return _empty_transcript(lang)
        finally:
            if is_temp and audio_path != audio_file:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    # -----------------------------------------------------------------------
    # faster-whisper loader (with local-path and proxy-bypass support)
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_faster_whisper(model_name: str, device: str, compute_type: str,
                             script_dir: str):
        """
        Load WhisperModel. Resolution order:
        1. Local path: <script_dir>/models/whisper-<name>/   (ct2 format)
        2. HuggingFace download with proxy env vars temporarily cleared.
        Returns model or None on failure.
        """
        local_name = LOCAL_MODEL_MAP.get(model_name, "base")

        # 1. Check bundled local model
        for candidate_name in (local_name, model_name):
            local_dir = os.path.join(script_dir, "models", f"whisper-{candidate_name}")
            if os.path.isdir(local_dir) and os.listdir(local_dir):
                try:
                    m = _WhisperModel(local_dir, device=device, compute_type=compute_type)
                    print(f"[OK] faster-whisper loaded from local: {local_dir}")
                    return m
                except Exception as exc:
                    print(f"[WARN] Local whisper model load failed ({local_dir}): {exc}")

        # 2. HuggingFace download — bypass SOCKS proxy (unsupported by httpx)
        #    Approach A: clear env vars
        #    Approach B: patch httpx trust_env=False so it ignores system proxy
        proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                      "ALL_PROXY", "all_proxy", "REQUESTS_CA_BUNDLE")
        saved_proxy = {k: os.environ.pop(k, None) for k in proxy_keys}
        # Set NO_PROXY to catch any residual proxy detection
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        try:
            # Patch huggingface_hub http session to disable system proxy
            try:
                import httpx
                import huggingface_hub.utils._http as _hf_http
                _orig_factory = getattr(_hf_http, "_GLOBAL_CLIENT_FACTORY", None)

                def _no_proxy_factory():
                    return httpx.Client(trust_env=False, timeout=120)

                _hf_http._GLOBAL_CLIENT_FACTORY = _no_proxy_factory
                _hf_http._GLOBAL_CLIENT = None  # force rebuild
            except Exception:
                pass  # patch failed — proceed anyway

            m = _WhisperModel(local_name, device=device, compute_type=compute_type)
            print(f"[OK] faster-whisper '{local_name}' on {device}/{compute_type}")
            return m

        except Exception as exc:
            print(f"[WARN] faster-whisper init failed: {exc}")
            print(f"[INFO] To use faster-whisper offline, download the model once:")
            print(f"[INFO]   python -c \"from faster_whisper import WhisperModel; "
                  f"WhisperModel('{local_name}')\"")
            print(f"[INFO] Or place a ct2-format model in:")
            print(f"[INFO]   {os.path.join(script_dir, 'models', f'whisper-{local_name}')}")
            return None
        finally:
            # Restore proxy settings
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
            for k, v in saved_proxy.items():
                if v is not None:
                    os.environ[k] = v

    # -----------------------------------------------------------------------
    # Internal — single file transcription
    # -----------------------------------------------------------------------

    def _transcribe_single(
        self,
        audio_path: str,
        language:   Optional[str],
        beam_size:  int,
        vad_filter: bool,
    ) -> Dict:
        """Transcribe a single audio file (< 25 MB)."""
        # Try faster-whisper first (local, no API key needed)
        if self._model is not None:
            return self._transcribe_with_faster_whisper(
                audio_path, language, beam_size, vad_filter, offset=0.0
            )

        # Fallback: OpenAI API
        if self._openai_client is not None:
            return self._transcribe_with_openai(audio_path, language)

        raise RuntimeError(
            "No transcription engine available. "
            "Install faster-whisper or set OPENAI_API_KEY."
        )

    def _transcribe_chunked(
        self,
        audio_path: str,
        language:   Optional[str],
        beam_size:  int,
        vad_filter: bool,
    ) -> Dict:
        """Split audio and transcribe each chunk, then merge with timestamps."""
        chunks = self.split_audio(audio_path, CHUNK_DURATION_S, CHUNK_LIMIT_MB)
        chunks_to_cleanup = [c for c in chunks if c != audio_path]

        full_transcript: Dict = {"text": "", "language": language or "unknown",
                                 "segments": [], "words": []}
        offset_seconds = 0.0

        for i, chunk_path in enumerate(chunks):
            print(f"[INFO] Chunk {i+1}/{len(chunks)}: {chunk_path}")
            try:
                if self._model is not None:
                    chunk_result = self._transcribe_with_faster_whisper(
                        chunk_path, language, beam_size, vad_filter, offset=offset_seconds
                    )
                elif self._openai_client is not None:
                    chunk_result = self._transcribe_with_openai(chunk_path, language)
                    chunk_result = _shift_timestamps(chunk_result, offset_seconds)
                else:
                    raise RuntimeError("No transcription engine available.")

                # Merge
                if full_transcript["text"]:
                    full_transcript["text"] += " " + chunk_result["text"]
                else:
                    full_transcript["text"]  = chunk_result["text"]
                full_transcript["language"]  = chunk_result.get("language") or language or "unknown"
                full_transcript["segments"].extend(chunk_result.get("segments", []))
                full_transcript["words"].extend(chunk_result.get("words", []))

                offset_seconds += self.get_duration(chunk_path)

            except Exception as exc:
                print(f"[WARN] Error transcribing chunk {i+1}: {exc}")

        # Cleanup temp chunks
        for c in chunks_to_cleanup:
            try:
                os.remove(c)
                print(f"[INFO] Cleaned up: {c}")
            except OSError:
                print(f"[WARN] Failed to cleanup: {c}")

        return full_transcript

    # -----------------------------------------------------------------------
    # faster-whisper engine
    # -----------------------------------------------------------------------

    def _transcribe_with_faster_whisper(
        self,
        audio_path: str,
        language:   Optional[str],
        beam_size:  int,
        vad_filter: bool,
        offset:     float = 0.0,
    ) -> Dict:
        """Transcribe using faster-whisper (local)."""
        segments_gen, info = self._model.transcribe(
            audio_path,
            beam_size        = beam_size,
            language         = language,
            vad_filter       = vad_filter,
            word_timestamps  = True,
            task             = "transcribe",
        )

        all_text      = ""
        all_segments: List[Dict] = []
        all_words:    List[Dict] = []

        for seg in segments_gen:
            seg_text  = seg.text.strip()
            seg_start = round(seg.start + offset, 3)
            seg_end   = round(seg.end   + offset, 3)
            all_text  = (all_text + " " + seg_text).strip()

            # Word-level
            seg_words: List[Dict] = []
            if seg.words:
                for w in seg.words:
                    word_entry = {
                        "word":  w.word.strip(),
                        "start": round(w.start + offset, 3),
                        "end":   round(w.end   + offset, 3),
                    }
                    seg_words.append(word_entry)
                    all_words.append(word_entry)
            else:
                # No word timestamps — synthesize from segment
                words_in_seg = seg_text.split()
                if words_in_seg:
                    step = (seg_end - seg_start) / len(words_in_seg)
                    for wi, wt in enumerate(words_in_seg):
                        we = {
                            "word":  wt,
                            "start": round(seg_start + wi * step, 3),
                            "end":   round(seg_start + (wi + 1) * step, 3),
                        }
                        seg_words.append(we)
                        all_words.append(we)

            all_segments.append({
                "start": seg_start,
                "end":   seg_end,
                "text":  seg_text,
                "words": seg_words,
            })

        n_segs = len(all_segments)
        n_words = len(all_words)
        if not all_words:
            print(f"[INFO] {n_segs} segments (no words).")
        else:
            print(f"[OK] Transcribed {n_segs} segments, {n_words} words "
                  f"(lang={info.language})")

        return {
            "text":     all_text,
            "language": info.language or language or "unknown",
            "segments": all_segments,
            "words":    all_words,
        }

    # -----------------------------------------------------------------------
    # OpenAI API engine  (original binary engine)
    # -----------------------------------------------------------------------

    def _transcribe_with_openai(
        self,
        audio_path: str,
        language:   Optional[str],
    ) -> Dict:
        """Transcribe using OpenAI Whisper API (original binary engine)."""
        with open(audio_path, "rb") as af:
            transcription = self._openai_client.audio.transcriptions.create(
                model                  = "whisper-1",
                file                   = af,
                response_format        = "verbose_json",
                timestamp_granularities = ["word"],
                language               = language,
            )

        # Parse verbose_json response
        raw = transcription.model_dump() if hasattr(transcription, "model_dump") else {}
        return _normalise_openai_response(raw, language)

    # -----------------------------------------------------------------------
    # Audio extraction helper
    # -----------------------------------------------------------------------

    def _extract_audio(self, video_path: str) -> Tuple[str, bool]:
        """
        Extract audio track from video to MP3.
        Returns (audio_path, is_temp).
        If file is already audio-only, returns (video_path, False).
        """
        ext = Path(video_path).suffix.lower()
        if ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"):
            return video_path, False

        tmp_audio = tempfile.mktemp(suffix=".mp3", prefix="boosta_audio_")
        cmd = [
            self.ffmpeg, "-y",
            "-i", video_path,
            "-vn",
            "-acodec", AUDIO_CODEC,
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-q:a", "5",
            "-loglevel", "error",
            tmp_audio,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if os.path.exists(tmp_audio):
                return tmp_audio, True
        except Exception as exc:
            print(f"[WARN] Audio extraction failed: {exc}")
        return video_path, False

    # -----------------------------------------------------------------------
    # Utils
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_bin(name: str, script_dir: str) -> str:
        """Find binary: local dir first, then PATH."""
        local = os.path.join(script_dir, name)
        if os.path.exists(local):
            return local
        local_exe = local + ".exe"
        if os.path.exists(local_exe):
            return local_exe
        return name  # rely on PATH


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _empty_transcript(language: Optional[str] = None) -> Dict:
    return {
        "text":     "",
        "language": language or "unknown",
        "segments": [],
        "words":    [],
    }


def _shift_timestamps(transcript: Dict, offset: float) -> Dict:
    """Shift all timestamps in transcript by offset seconds."""
    if offset == 0:
        return transcript
    result = dict(transcript)
    result["segments"] = [
        {**s,
         "start": round(s.get("start", 0) + offset, 3),
         "end":   round(s.get("end",   0) + offset, 3),
         "words": [{**w,
                    "start": round(w.get("start", 0) + offset, 3),
                    "end":   round(w.get("end",   0) + offset, 3)}
                   for w in s.get("words", [])]}
        for s in transcript.get("segments", [])
    ]
    result["words"] = [
        {**w,
         "start": round(w.get("start", 0) + offset, 3),
         "end":   round(w.get("end",   0) + offset, 3)}
        for w in transcript.get("words", [])
    ]
    return result


def _normalise_openai_response(raw: Dict, language: Optional[str]) -> Dict:
    """Normalise OpenAI verbose_json response to internal format."""
    # OpenAI words use {"word": ..., "start": ..., "end": ...}
    words: List[Dict] = []
    segments: List[Dict] = []

    for seg in raw.get("segments", []):
        seg_text  = seg.get("text", "").strip()
        seg_start = float(seg.get("start", 0))
        seg_end   = float(seg.get("end",   0))
        seg_words: List[Dict] = []

        for w in seg.get("words", []):
            # OpenAI returns w_text, w_start, w_end (decoded from binary)
            entry = {
                "word":  w.get("word", "").strip(),
                "start": round(float(w.get("start", 0)), 3),
                "end":   round(float(w.get("end",   0)), 3),
            }
            seg_words.append(entry)
            words.append(entry)

        segments.append({
            "start": seg_start,
            "end":   seg_end,
            "text":  seg_text,
            "words": seg_words,
        })

    # Top-level words from response (some OpenAI versions return this)
    if not words:
        for w in raw.get("words", []):
            words.append({
                "word":  w.get("word", "").strip(),
                "start": round(float(w.get("start", 0)), 3),
                "end":   round(float(w.get("end",   0)), 3),
            })

    return {
        "text":     raw.get("text", "").strip(),
        "language": raw.get("language") or language or "unknown",
        "segments": segments,
        "words":    words,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <video_or_audio.mp4> [language]")
        print("       language: en, ru, de, fr, ... (optional, auto-detect if omitted)")
        sys.exit(1)

    input_file = sys.argv[1]
    lang       = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    print("=" * 60)
    print("Boosta Transcriber")
    print("=" * 60)

    t = Transcriber()

    dur = t.get_duration(input_file)
    print(f"Duration: {dur:.1f}s")

    result = t.transcribe(input_file, language=lang)

    words_n    = len(result.get("words", []))
    segments_n = len(result.get("segments", []))
    print(f"\nResult:")
    print(f"  Language: {result.get('language')}")
    print(f"  Segments: {segments_n}")
    print(f"  Words:    {words_n}")
    print(f"  Text preview: {result.get('text', '')[:120]}...")

    # Save transcript.json next to input file
    out_path = str(Path(input_file).with_suffix(".transcript.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
