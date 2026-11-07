# -*- coding: utf-8 -*-
"""
gemini_analyzer.py
==================
Video-native viral moment analyzer for Boosta backend.

Reconstructed from: gemini_analyzer.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary.

Confirmed recovered elements:
  Class : GeminiVideoAnalyzer  (7 methods from binary)
  Methods:
    __init__
    analyze_video             (public API: aanalyze_video)
    analyze_video_with_gemini (internal: aanalyze_video_with_gemini)
    _build_analysis_prompt    (a_build_analysis_prompt)
    _parse_response           (a_parse_response)
    _generate_fallback_clips  (a_generate_fallback_clips)
    _get_video_duration       (a_get_video_duration)
    get_raw_response          (debugging: returns last raw Gemini response)

  Model   : gemini-2.5-flash  (exact from binary)
  Package : google.generativeai  (agenerativeai)
  Upload flow: upload_file → poll PROCESSING → generate_content → delete_file
  Safety: BLOCK_NONE for all 4 HARM categories (aBLOCK_NONE)

  JSON output schema (from binary examples):
    {"moments": [
      {"start": 34.0, "end": ..., "title": "...", "hook_phrase": "...",
       "reason": "...", "score": 7}
    ]}

  Validation (from binary):
    - Regex fallback: pattern matching start/end/title JSON objects
    - Min clip duration: 12 s  (binary: "s (min 12s)")
    - Max clip duration: 60 s  (binary: "Capped long moment to 60s")
    - Skip overlapping: aoverlap_start, aoverlap_end, aoverlap_duration
    - Skip beyond video: "w/u moments with INVALID timestamps (beyond"
    - Timestamp conversion: astart_min/astart_sec — detects "MM.SS" format from Gemini
    - Offsets: START -= 0.3s, END += 0.5s (binary: "add 0.3s before", "add 0.5s after")

  Prompt (reconstructed from binary strings):
    "You are an expert viral content analyst for TikTok, YouTube Shorts, and Instagram Reels."
    "WATCH THIS VIDEO CAREFULLY and identify ALL high-potential viral moments."
    1. DURATION: 15-60 seconds (sweet spot: 25-50s)
    - ALL timestamps must be in SECONDS (NOT minutes:seconds!)
    - CRITICAL TIMESTAMP FORMAT
    - WRONG: 2.17 → RIGHT: 137.0
    - START at FIRST word of sentence (add 0.3s before)
    - END at LAST word of sentence (add 0.5s after)
    - NEVER cut in middle of word/sentence/thought!
    - If someone says "I made $81k" - clip MUST show the proof screen
    - Keep JSON SHORT - no long descriptions

  Fallback (a_generate_fallback_clips):
    "Auto-generated fallback clip", "Viral Moment", "Highlight", "No title"
    Evenly distributes clips across video duration.

  Key difference from analyzer.py:
    analyzer.py        → transcript text → OpenAI GPT → moments
    gemini_analyzer.py → full video file → Gemini 2.5 Flash → moments
    "Much more accurate than transcript-only analysis because it can SEE the video"
    "Can understand visual content, emotions, reactions, and context"

Dependencies:
  google-generativeai OR google.genai  (Gemini API)
  ffprobe.exe  (duration, bundled)
  GEMINI_API_KEY env var
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dotenv
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# google.genai (new) OR google.generativeai (original binary)
# ---------------------------------------------------------------------------
GENAI_AVAILABLE = False
GENAI_NEW_SDK   = False
genai           = None  # type: ignore
HarmCategory    = None  # type: ignore
HarmBlockThreshold = None  # type: ignore

try:
    import google.genai as genai              # type: ignore
    from google.genai import types as _gt    # type: ignore
    GENAI_AVAILABLE = True
    GENAI_NEW_SDK   = True
except ImportError:
    pass

if not GENAI_AVAILABLE:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai   # type: ignore
            from google.generativeai.types import (
                HarmCategory, HarmBlockThreshold  # type: ignore
            )
        GENAI_AVAILABLE = True
        GENAI_NEW_SDK   = False
    except ImportError:
        pass

__all__ = ["GeminiVideoAnalyzer", "analyze_video"]

# ---------------------------------------------------------------------------
# Constants decoded from binary
# ---------------------------------------------------------------------------
MODEL_NAME        = "gemini-2.5-flash"   # exact from binary
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.7                  # creative viral detection
UPLOAD_WAIT_S     = 5
UPLOAD_TIMEOUT    = 180

MIN_CLIP_DURATION = 12.0    # binary: "s (min 12s)"
MAX_CLIP_DURATION = 60.0    # binary: "Capped long moment to 60s"

# Timestamp padding decoded from prompt strings
PAD_START = 0.3   # "add 0.3s before" (start of sentence)
PAD_END   = 0.5   # "add 0.5s after"  (end of sentence)


# ===========================================================================
# GeminiVideoAnalyzer
# ===========================================================================

class GeminiVideoAnalyzer:
    """
    Analyzes video content directly using Gemini 2.5 Flash.
    Uses Google Gemini 2.5 Flash to analyze video directly for viral moments.
    Much more accurate than transcript-only analysis because it can SEE the video.
    Can understand visual content, emotions, reactions, and context.
    """

    def __init__(
        self,
        api_key:      Optional[str] = None,
        model_name:   str = MODEL_NAME,
        ffprobe_path: str = "ffprobe",
    ) -> None:
        self.model_name  = model_name
        self._model      = None
        self._client     = None
        self._raw_response: Optional[str] = None

        script_dir   = os.path.dirname(os.path.abspath(__file__))
        self.ffprobe = self._find_bin(ffprobe_path, script_dir)

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            print("[WARN] GEMINI_API_KEY not found in environment variables")
        elif GENAI_AVAILABLE and genai is not None:
            try:
                if GENAI_NEW_SDK:
                    self._client = genai.Client(api_key=resolved_key)
                    self._model  = model_name
                else:
                    genai.configure(api_key=resolved_key)
                    _safety = {}
                    if HarmCategory is not None and HarmBlockThreshold is not None:
                        _safety = {
                            HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    self._model = genai.GenerativeModel(
                        model_name        = model_name,
                        generation_config = genai.GenerationConfig(
                            temperature        = TEMPERATURE,
                            max_output_tokens  = MAX_OUTPUT_TOKENS,
                            response_mime_type = "application/json",
                        ),
                        safety_settings = _safety,
                    )
                    self._client = None
                print(f"✅ Gemini 2.5 Flash initialized for video analysis")
            except Exception as exc:
                print(f"[WARN] Gemini init failed: {exc}")
                self._model = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def analyze_video(
        self,
        video_path:          str,
        transcript_segments: Optional[List[Dict]] = None,
        video_duration:      Optional[float]      = None,
        num_clips:           int                  = 5,
        clip_duration:       Optional[float]      = None,
    ) -> List[Dict]:
        """
        Analyze video for viral moments using Gemini 2.5 Flash.

        Args:
            video_path:          Path to the video file.
            transcript_segments: Optional transcript for context (may be None).
            video_duration:      Duration in seconds (auto-detected if None).
            num_clips:           Desired number of clips to extract.
            clip_duration:       Optional fixed clip length hint.

        Returns:
            List of moment dicts compatible with clipper.py:
            [{"start": float, "end": float, "title": str,
              "hook_phrase": str, "reason": str, "score": int}]
        """
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            return []

        duration = video_duration or self._get_video_duration(video_path)

        # ── Gemini path ──────────────────────────────────────────────────────
        if self._model is not None and GENAI_AVAILABLE:
            try:
                return self.analyze_video_with_gemini(
                    video_path, transcript_segments, duration, num_clips, clip_duration
                )
            except Exception as exc:
                print(f"[WARN] Gemini video analysis failed: {exc}")

        # ── Fallback path ────────────────────────────────────────────────────
        print(f"[WARN] Falling back to generated clips for {os.path.basename(video_path)}")
        return self._generate_fallback_clips(duration, num_clips, clip_duration)

    def analyze_video_with_gemini(
        self,
        video_path:          str,
        transcript_segments: Optional[List[Dict]],
        video_duration:      Optional[float],
        num_clips:           int,
        clip_duration:       Optional[float],
    ) -> List[Dict]:
        """Upload video to Gemini and analyze for viral moments."""
        if GENAI_NEW_SDK:
            return self._call_new_sdk(
                video_path, transcript_segments, video_duration, num_clips, clip_duration
            )
        else:
            return self._call_legacy_sdk(
                video_path, transcript_segments, video_duration, num_clips, clip_duration
            )

    def get_raw_response(self) -> Optional[str]:
        """Get the last raw response from Gemini for debugging."""
        return self._raw_response

    # -----------------------------------------------------------------------
    # SDK implementations
    # -----------------------------------------------------------------------

    def _call_legacy_sdk(
        self,
        video_path:          str,
        transcript_segments: Optional[List[Dict]],
        video_duration:      Optional[float],
        num_clips:           int,
        clip_duration:       Optional[float],
    ) -> List[Dict]:
        video_file = None
        try:
            print(f"    📤 Uploading video to Gemini for analysis...")
            video_file = genai.upload_file(video_path, mime_type="video/mp4")

            print(f"    ⏳ Waiting for Gemini to process video...")
            deadline = time.time() + UPLOAD_TIMEOUT
            while video_file.state.name == "PROCESSING":
                if time.time() > deadline:
                    raise TimeoutError("Upload timed out")
                time.sleep(UPLOAD_WAIT_S)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError("Video processing failed")

            prompt    = self._build_analysis_prompt(
                video_duration, num_clips, clip_duration, transcript_segments
            )
            print(f"    🔍 Analyzing video for viral moments...")
            response  = self._model.generate_content([video_file, prompt])
            self._raw_response = response.text.strip()

            return self._parse_response(self._raw_response, video_duration)
        finally:
            if video_file is not None:
                try: genai.delete_file(video_file.name)
                except Exception: pass

    def _call_new_sdk(
        self,
        video_path:          str,
        transcript_segments: Optional[List[Dict]],
        video_duration:      Optional[float],
        num_clips:           int,
        clip_duration:       Optional[float],
    ) -> List[Dict]:
        file_ref = None
        try:
            print(f"    📤 Uploading video to Gemini for analysis...")
            file_ref = self._client.files.upload(
                path   = video_path,
                config = {"mime_type": "video/mp4"},
            )
            print(f"    ⏳ Waiting for Gemini to process video...")
            deadline = time.time() + UPLOAD_TIMEOUT
            while file_ref.state.name not in ("ACTIVE", "FAILED"):
                if time.time() > deadline:
                    raise TimeoutError("Upload timed out")
                time.sleep(UPLOAD_WAIT_S)
                file_ref = self._client.files.get(name=file_ref.name)
            if file_ref.state.name == "FAILED":
                raise RuntimeError("Video processing failed")

            prompt = self._build_analysis_prompt(
                video_duration, num_clips, clip_duration, transcript_segments
            )
            print(f"    🔍 Analyzing video for viral moments...")
            response = self._client.models.generate_content(
                model    = self._model,
                contents = [file_ref, prompt],
                config   = {"response_mime_type": "application/json",
                            "temperature": TEMPERATURE},
            )
            self._raw_response = response.text.strip()
            return self._parse_response(self._raw_response, video_duration)
        finally:
            if file_ref is not None:
                try: self._client.files.delete(name=file_ref.name)
                except Exception: pass

    # -----------------------------------------------------------------------
    # Prompt builder
    # -----------------------------------------------------------------------

    def _build_analysis_prompt(
        self,
        video_duration:      Optional[float],
        num_clips:           int,
        clip_duration:       Optional[float],
        transcript_segments: Optional[List[Dict]],
    ) -> str:
        """Build the analysis prompt for Gemini."""
        dur_str = f"{video_duration:.0f}" if video_duration else "unknown"

        # Optional transcript context
        transcript_block = ""
        if transcript_segments:
            lines = []
            for seg in transcript_segments[:60]:   # cap to avoid huge prompts
                start = seg.get("start", 0)
                text  = seg.get("text", "").strip()
                if text:
                    lines.append(f"  [{start:.1f}s] {text}")
            if lines:
                transcript_block = (
                    "\n\nTRANSCRIPT CONTEXT (use for exact word timestamps):\n"
                    + "\n".join(lines)
                    + "\n"
                )

        clip_hint = ""
        if clip_duration:
            clip_hint = f"\nPreferred clip length: ~{clip_duration:.0f} seconds."

        return f"""You are an expert viral content analyst for TikTok, YouTube Shorts, and Instagram Reels.

WATCH THIS VIDEO CAREFULLY and identify ALL high-potential viral moments.
Video duration: {dur_str} seconds. Find {num_clips} best moments.{clip_hint}{transcript_block}

???????????? CRITICAL TIMESTAMP FORMAT! ????????????
- ALL timestamps must be in SECONDS (NOT minutes:seconds!)
- WRONG: 2.17 (this means 2.17 seconds, NOT 2 min 17 sec!)
- RIGHT: 137.0 (this means 2 min 17 sec = 137 seconds)
?????? TIMESTAMPS ARE IN SECONDS! Example: 2 minutes 30 seconds = 150.0 (NOT 2.30!)

RULES FOR EACH MOMENT:
1. DURATION: 15-60 seconds (sweet spot: 25-50s)
2. - START at FIRST word of sentence (add 0.3s before)
   - END at LAST word of sentence (add 0.5s after)
   - NEVER cut in middle of word/sentence/thought!
3. - If someone says "I made $81k" - clip MUST show the proof screen
4. - Keep JSON SHORT - no long descriptions
5. - Your timestamps must be between 0 and {dur_str}

Return ONLY valid JSON (timestamps in SECONDS!):

{{"moments": [
  {{
    "start": 34.0,
    "end": 67.0,
    "title": "Short title max 50 chars",
    "hook_phrase": "First words",
    "reason": "Why viral",
    "score": 8
  }},
  {{
    "start": 120.0,
    "end": 155.0,
    "title": "Another moment",
    "hook_phrase": "Hook words",
    "reason": "Why viral",
    "score": 7
  }}
]}}"""

    # -----------------------------------------------------------------------
    # Response parser
    # -----------------------------------------------------------------------

    def _parse_response(
        self,
        response_text: str,
        video_duration: Optional[float],
    ) -> List[Dict]:
        """Parse and validate layout response from Gemini."""
        # Strip markdown fences
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        # Parse JSON
        raw_moments: List[Dict] = []
        try:
            data = json.loads(clean)
            raw_moments = data.get("moments", data if isinstance(data, list) else [])
        except json.JSONDecodeError:
            # Try extracting individual moment objects with regex
            # Pattern recovered from binary:
            # \{\s*"start"\s*:\s*[\d.]+\s*,\s*"end"\s*:\s*[\d.]+\s*,\s*"title"\s*:\s*"[^"]*"[^}]*\}
            pattern = r'\{\s*"start"\s*:\s*[\d.]+\s*,\s*"end"\s*:\s*[\d.]+\s*,\s*"title"\s*:\s*"[^"]*"[^}]*\}'
            for m in re.finditer(pattern, clean, re.DOTALL):
                try:
                    raw_moments.append(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    pass

        if not raw_moments:
            print(f"[WARN] No moments parsed from response")
            return []

        validated: List[Dict] = []
        skipped_invalid = 0
        max_end = video_duration or float("inf")

        for raw in raw_moments:
            try:
                start_raw = float(raw.get("start", 0))
                end_raw   = float(raw.get("end",   0))

                # Detect and convert MM.SS format (astart_min, astart_sec)
                # e.g. Gemini returns 2.30 meaning 2 min 30 sec → 150.0
                start = self._fix_timestamp(start_raw, video_duration)
                end   = self._fix_timestamp(end_raw,   video_duration)

                # Apply padding offsets from binary
                start = max(0.0, start - PAD_START)
                end   = end + PAD_END

                # Validate against video duration
                if video_duration and start >= video_duration:
                    skipped_invalid += 1
                    continue

                # Cap end to video duration
                if video_duration and end > video_duration:
                    print(f"    ⚠️  Capped end to video duration: {end:.1f}s → {video_duration:.1f}s")
                    end = video_duration

                # Cap clip length to MAX
                if end - start > MAX_CLIP_DURATION:
                    print(f"    ⚠️  Capped long moment to 60s")
                    end = start + MAX_CLIP_DURATION

                # Skip too short
                if end - start < MIN_CLIP_DURATION:
                    print(f"    ⚠️  Skipping too short moment: {end - start:.1f}s (min 12s)")
                    continue

                validated_moment: Dict = {
                    "start":       round(start, 3),
                    "end":         round(end, 3),
                    "title":       str(raw.get("title",       "Viral Moment"))[:50],
                    "hook_phrase": str(raw.get("hook_phrase", "")),
                    "reason":      str(raw.get("reason",      "")),
                    "score":       int(raw.get("score",       7)),
                }
                validated.append(validated_moment)

            except (TypeError, ValueError):
                continue

        if skipped_invalid:
            print(f"    ⚠️  {skipped_invalid} moments with INVALID timestamps (beyond video duration)")

        # Remove overlapping moments (keep higher score)
        validated = self._remove_overlaps(validated)

        return validated

    @staticmethod
    def _fix_timestamp(ts: float, video_duration: Optional[float]) -> float:
        """
        Detect and fix MM.SS-style timestamps from Gemini.
        e.g. 2.17 → might mean 2 min 17 sec = 137.0 seconds
        Only converts when the result makes more sense in context.
        """
        if ts <= 0:
            return 0.0

        # Check if fractional part looks like seconds (0.00–0.59 range)
        frac = ts - math.floor(ts)
        if frac > 0.0 and frac <= 0.59 and math.floor(ts) <= 120:
            # Candidate: treat as MM.SS
            start_min = math.floor(ts)
            start_sec = round(frac * 100)
            converted = start_min * 60 + start_sec

            # Only use conversion if:
            # - original value < video duration (original could be valid)
            # - converted value is also < video duration
            # - AND converted > original significantly
            if video_duration:
                orig_valid      = ts < video_duration
                converted_valid = converted < video_duration
                # If original is valid and converted is too, keep original
                # If original is beyond duration but converted is not, convert
                if not orig_valid and converted_valid:
                    return float(converted)

        return float(ts)

    @staticmethod
    def _remove_overlaps(moments: List[Dict]) -> List[Dict]:
        """Remove overlapping moments, keeping the one with higher score."""
        if len(moments) <= 1:
            return moments

        # Sort by score descending
        sorted_m = sorted(moments, key=lambda m: m.get("score", 0), reverse=True)
        kept: List[Dict] = []

        for candidate in sorted_m:
            c_start = candidate["start"]
            c_end   = candidate["end"]
            overlaps = False
            for kept_m in kept:
                k_start = kept_m["start"]
                k_end   = kept_m["end"]
                # Check overlap
                overlap_start = max(c_start, k_start)
                overlap_end   = min(c_end,   k_end)
                overlap_dur   = overlap_end - overlap_start
                if overlap_dur > 5.0:   # more than 5s overlap = skip
                    print(f"    ⚠️  Skipping overlapping moment: {c_start:.1f}–{c_end:.1f}s "
                          f"overlaps with {k_start:.1f}–{k_end:.1f}s")
                    overlaps = True
                    break
            if not overlaps:
                kept.append(candidate)

        # Re-sort by start time for clipper.py
        kept.sort(key=lambda m: m["start"])
        return kept

    # -----------------------------------------------------------------------
    # Fallback clips
    # -----------------------------------------------------------------------

    def _generate_fallback_clips(
        self,
        video_duration: Optional[float],
        num_clips:      int,
        clip_duration:  Optional[float],
    ) -> List[Dict]:
        """Generate basic clips when analysis fails."""
        if not video_duration or video_duration <= 0:
            return []

        span    = clip_duration or min(MAX_CLIP_DURATION, max(MIN_CLIP_DURATION, video_duration / max(1, num_clips)))
        results: List[Dict] = []
        step    = (video_duration - span) / max(1, num_clips - 1) if num_clips > 1 else 0

        titles = ["Highlight", "Viral Moment", "Auto-generated fallback clip", "No title"]

        for i in range(num_clips):
            start = round(step * i, 2)
            end   = round(min(start + span, video_duration), 2)
            if end - start < MIN_CLIP_DURATION:
                break
            results.append({
                "start":       start,
                "end":         end,
                "title":       titles[i % len(titles)],
                "hook_phrase": "",
                "reason":      "Auto-generated fallback clip",
                "score":       5,
            })

        print(f"    ⚠️  Generated {len(results)} fallback clips")
        return results

    # -----------------------------------------------------------------------
    # Duration
    # -----------------------------------------------------------------------

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                self.ffprobe, "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
            return float(out.strip())
        except Exception:
            pass
        return 0.0

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_bin(name: str, script_dir: str) -> str:
        for candidate in (name, name + ".exe",
                          os.path.join(script_dir, name),
                          os.path.join(script_dir, name + ".exe")):
            if os.path.exists(candidate):
                return candidate
        return name


# ===========================================================================
# Module-level convenience
# ===========================================================================

def analyze_video(
    video_path:          str,
    transcript_segments: Optional[List[Dict]] = None,
    video_duration:      Optional[float]      = None,
    num_clips:           int                  = 5,
    clip_duration:       Optional[float]      = None,
    api_key:             Optional[str]        = None,
) -> List[Dict]:
    """
    Analyze video directly to find viral moments.

    Args:
        video_path:          Path to video file.
        transcript_segments: Optional transcript context.
        video_duration:      Duration in seconds (auto-detected if None).
        num_clips:           Number of clips to extract.
        clip_duration:       Optional fixed clip length hint.
        api_key:             Gemini API key (uses GEMINI_API_KEY env if None).

    Returns:
        List of moment dicts (start/end/title/hook_phrase/reason/score).
    """
    analyzer = GeminiVideoAnalyzer(api_key=api_key)
    return analyzer.analyze_video(
        video_path, transcript_segments, video_duration, num_clips, clip_duration
    )


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import io
    import sys

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("Usage: python gemini_analyzer.py <video_path>")

    if len(sys.argv) < 2:
        sys.exit(0)

    video = sys.argv[1]
    if not os.path.exists(video):
        print(f"File not found: {video}")
        sys.exit(1)

    print("=" * 60)
    print("Gemini Video Analyzer")
    print("=" * 60)

    analyzer = GeminiVideoAnalyzer()
    moments  = analyzer.analyze_video(video)
    print()
    print(json.dumps({"moments": moments}, indent=2, ensure_ascii=False))

    out_path = video + ".moments.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"moments": moments}, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
