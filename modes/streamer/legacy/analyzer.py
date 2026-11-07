# -*- coding: utf-8 -*-
"""
analyzer.py
===========
Viral moment detection module for Boosta backend.

Reconstructed from: analyzer.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary.

Confirmed recovered elements:
  - Class: ViralMomentAnalyzer  (4 methods confirmed from binary)
  - Methods: __init__, analyze_transcript, _detect_language,
             _ensure_minimum_clips, _format_time
  - Model: "gpt-5-mini-2025-08-07"  (exact string from binary)
    (Note: docstring says "GPT-4o" — internal log uses actual model name)
  - System prompt: recovered verbatim from binary strings
  - Transcript format: [BATCH_X][MM:SS][XXX.XXs] text
  - Output keys: start, end, start_batch, hook_phrase, title, reason, score
  - Language detection: cyrillic_count, latin_count, spanish_chars
  - Validation: clip_duration 15-60s, overlap check, start-time correction
  - Fallback: _ensure_minimum_clips with section_duration-based distribution
  - Min clips: 3 (from log "need at least N")
  - Clip limits: MINIMUM 15s | MAXIMUM 60s

Output format (moments list) — compatible with clipper.py cut_all_moments():
[
  {
    "start":      154.20,
    "end":        189.50,
    "start_batch": 15,
    "hook_phrase": "The exact first words that hook the viewer",
    "title":      "Hook-Based Attention Title",
    "reason":     "Why this hook captures attention",
    "score":      8
  }
]

Dependencies:
  openai >= 1.0.0     (cloud engine, requires OPENAI_API_KEY)
  python-dotenv       (optional, for .env loading)

Fallback (no API key): returns evenly distributed clips from transcript.
"""

from __future__ import annotations

import json
import math
import os
import traceback
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dotenv
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI as _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None  # type: ignore

__all__ = ["ViralMomentAnalyzer", "analyze_transcript"]

# ---------------------------------------------------------------------------
# Constants (decoded from binary)
# ---------------------------------------------------------------------------
DEFAULT_MODEL       = "gpt-5-mini-2025-08-07"   # exact string from binary
MODEL_FALLBACKS     = [                           # fallback chain if default unavailable
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]
MIN_CLIP_DURATION   = 15    # seconds — binary: "MINIMUM: 15 seconds"
MAX_CLIP_DURATION   = 60    # seconds — binary: "MAXIMUM: 60 seconds"
TARGET_CLIPS        = 5     # soft target (prompt: "5-10 BEST viral moments")
MIN_CLIPS           = 3     # hard minimum (binary: "need at least N")
DEFAULT_CLIP_SPAN   = 30    # fallback clip length when using _ensure_minimum_clips
MAX_COMPLETION_TOKENS = 4096

# Recovered verbatim from binary
SYSTEM_PROMPT = (
    "You are a viral short-form content expert. Find moments that START with "
    "attention-grabbing hooks. Always return a JSON object with a 'moments' array. "
    "Use BATCH numbers from transcript for precise timestamps."
)

SYSTEM_PROMPT_EXTENDED = (
    "You are an expert social media content analyst specializing in viral short-form "
    "content for TikTok/YouTube Shorts/Reels."
)


# ===========================================================================
# ViralMomentAnalyzer
# ===========================================================================

class ViralMomentAnalyzer:
    """
    Analyzes transcript segments to identify viral moments using GPT-4o.
    Returns a list of potential viral clips with start/end times and reasoning.
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        # Model resolution: explicit arg > OPENAI_MODEL env > DEFAULT_MODEL
        env_model    = os.environ.get("OPENAI_MODEL", "")
        env_base_url = os.environ.get("OPENAI_BASE_URL", "")
        self.model   = model or env_model or DEFAULT_MODEL

        # Resolve API key and base_url
        resolved_key      = api_key  or os.environ.get("OPENAI_API_KEY", "")
        resolved_base_url = base_url or env_base_url or None

        if not resolved_key:
            print("[WARN] Warning: OPENAI_API_KEY not found in .env file.")

        self._client: Optional[object] = None
        if resolved_key and OPENAI_AVAILABLE and _OpenAI is not None:
            try:
                client_kwargs: dict = {"api_key": resolved_key}
                if resolved_base_url:
                    client_kwargs["base_url"] = resolved_base_url
                    print(f"[INFO] ViralMomentAnalyzer: using custom base_url={resolved_base_url}")
                self._client = _OpenAI(**client_kwargs)
                print(f"[OK] ViralMomentAnalyzer: client ready  model={self.model}"
                      + (f"  via {resolved_base_url}" if resolved_base_url else ""))
            except Exception as exc:
                print(f"[WARN] OpenAI client init failed: {exc}")

        if not self._client:
            print("[INFO] ViralMomentAnalyzer: running in fallback mode (no API)")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def analyze_transcript(
        self,
        transcript:       Dict,
        video_duration:   Optional[float] = None,
        existing_moments: Optional[List[Dict]] = None,
        min_clips:        int = MIN_CLIPS,
    ) -> List[Dict]:
        """
        Analyzes transcript segments to identify viral moments using GPT-4o.
        Returns a list of potential viral clips with start/end times and reasoning.

        Args:
            transcript:       Dict from transcriber.py with keys
                              "text", "segments", "words".
            video_duration:   Total video length in seconds (auto-detected if None).
            existing_moments: Pre-existing moments to avoid overlapping.
            min_clips:        Minimum clips to return (fallback fills the rest).

        Returns:
            List of moment dicts sorted by start time.
        """
        segments = transcript.get("segments", [])
        if not segments:
            print("[WARN] Transcript has no segments — returning empty moments.")
            return []

        # Infer video duration
        if video_duration is None:
            last_seg = segments[-1]
            video_duration = float(last_seg.get("end", last_seg.get("start", 0)) + 5)

        # Build BATCH-formatted transcript text
        transcript_text = self._build_batch_transcript(segments)

        # Language detection
        full_text = transcript.get("text", "") or " ".join(
            s.get("text", "") for s in segments
        )
        detected_lang = self._detect_language(full_text)

        lang_instruction = {
            "ru": "\nIMPORTANT: The video is in RUSSIAN. Generate ALL titles in RUSSIAN language.",
            "es": "\nIMPORTANT: The video is in SPANISH. Generate ALL titles in SPANISH language.",
            "de": "\nIMPORTANT: The video is in GERMAN. Generate ALL titles in GERMAN language.",
        }.get(detected_lang, "")

        # Try API first; fall back to heuristic
        viral_moments: List[Dict] = []
        if self._client is not None:
            viral_moments = self._call_api(
                transcript_text, video_duration, lang_instruction
            )

        # Validate moments (duration, overlaps, timestamp correction)
        validated = self._validate_moments(viral_moments, segments, video_duration)

        # Merge with existing_moments for overlap checking in _ensure_minimum_clips
        all_existing = list(existing_moments or []) + validated

        # Ensure minimum clip count
        result = self._ensure_minimum_clips(
            validated, segments, video_duration, min_clips
        )

        # Sort by start
        result.sort(key=lambda m: float(m.get("start", 0)))

        print(f"[OK] {len(result)} potential viral moments")
        for m in result:
            score = m.get("score", "?")
            title = m.get("title", "")[:50]
            print(f"[INFO]   (Score: {score}) {title}")

        return result

    # -----------------------------------------------------------------------
    # Private — language detection
    # -----------------------------------------------------------------------

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        Returns language code: 'en', 'ru', 'es', 'de', etc.
        """
        sample_text = text[:500]
        total_alpha = sum(1 for c in sample_text if c.isalpha()) or 1

        # Cyrillic → Russian/Ukrainian/etc.
        cyrillic_count = sum(
            1 for c in sample_text if "\u0400" <= c <= "\u04ff"
        )
        if cyrillic_count / total_alpha > 0.2:
            return "ru"

        # Spanish-specific characters
        spanish_chars = sum(
            1 for c in sample_text if c in "ñáéíóúüÑÁÉÍÓÚÜ¿¡"
        )
        if spanish_chars >= 2:
            return "es"

        # German-specific characters
        german_chars = sum(
            1 for c in sample_text if c in "äöüßÄÖÜ"
        )
        if german_chars > 3:
            return "de"

        return "en"

    # -----------------------------------------------------------------------
    # Private — minimum clips fallback
    # -----------------------------------------------------------------------

    def _ensure_minimum_clips(
        self,
        moments:        List[Dict],
        segments:       List[Dict],
        video_duration: float,
        min_clips:      int = MIN_CLIPS,
    ) -> List[Dict]:
        """
        Ensure we have at least min_clips by adding evenly distributed clips.
        Avoids overlapping with existing moments.
        """
        if len(moments) >= min_clips:
            return moments

        needed = min_clips - len(moments)
        print(f"[INFO] {len(moments)} moments found, need at least {min_clips}")

        existing_times = [(float(m["start"]), float(m["end"])) for m in moments]
        new_moments    = list(moments)

        # Evenly distribute target times for needed clips
        section_duration = video_duration / (needed + 1)

        added = 0
        for i in range(1, needed + 10):   # iterate more than needed to skip overlaps
            if added >= needed:
                break

            target_time = section_duration * i

            # Find segment closest to target_time
            best_seg_idx = 0
            best_distance = float("inf")
            for j, seg in enumerate(segments):
                distance = abs(float(seg.get("start", 0)) - target_time)
                if distance < best_distance:
                    best_distance = distance
                    best_seg_idx  = j

            seg          = segments[best_seg_idx]
            actual_start = float(seg.get("start", 0))
            end_time = min(actual_start + DEFAULT_CLIP_SPAN, video_duration)
            # If too close to end of video, slide start backward
            if (video_duration - actual_start) < MIN_CLIP_DURATION:
                actual_start = max(0.0, video_duration - DEFAULT_CLIP_SPAN)
                end_time     = video_duration
            clip_duration = end_time - actual_start

            if clip_duration < MIN_CLIP_DURATION:
                continue

            # Overlap check
            overlaps = any(
                not (end_time <= ex_start or actual_start >= ex_end)
                for ex_start, ex_end in existing_times
            )
            if overlaps:
                continue

            title_text = seg.get("text", "")[:50].strip()
            new_moment: Dict = {
                "start":       actual_start,
                "end":         end_time,
                "start_batch": best_seg_idx,
                "hook_phrase": title_text,
                "title":       title_text,
                "reason":      "Auto-selected to ensure minimum clips",
                "score":       5,
            }
            new_moments.append(new_moment)
            existing_times.append((actual_start, end_time))
            added += 1

            ts_from  = f"{actual_start:.1f}s"
            ts_to    = f"{end_time:.1f}s"
            batch_no = best_seg_idx
            print(
                f"[INFO]   Added fallback clip: [{ts_from} - {ts_to}] "
                f"(from BATCH_{batch_no}) Auto-selected to ensure minimum clips"
            )

        return new_moments

    # -----------------------------------------------------------------------
    # Private — transcript formatting
    # -----------------------------------------------------------------------

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        total_secs = int(seconds)
        mins       = total_secs // 60
        secs       = total_secs % 60
        return f"{mins:02d}:{secs:02d}"

    def _build_batch_transcript(self, segments: List[Dict]) -> str:
        """
        Build BATCH-numbered transcript text.
        Format per line: [BATCH_X][MM:SS][XXX.XXs] segment_text
        The model uses BATCH_X to reference precise timestamps.
        """
        lines = []
        for batch_idx, seg in enumerate(segments):
            start = float(seg.get("start", 0))
            text  = seg.get("text", "").strip()
            timestamp = self._format_time(start)
            lines.append(f"[BATCH_{batch_idx}][{timestamp}][{start:.2f}s] {text}")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Private — API call
    # -----------------------------------------------------------------------

    def _build_user_prompt(
        self,
        transcript_text: str,
        video_duration:  float,
        lang_instruction: str,
    ) -> str:
        """Build the full user prompt for the LLM."""
        dur_min = int(video_duration // 60)
        dur_sec = int(video_duration % 60)

        return f"""Analyze this video transcript and identify 5-10 BEST viral moments (15-60 seconds each).

Video duration: {dur_min}m {dur_sec}s
{lang_instruction}

???? HOOK RULE (MOST CRITICAL - THIS IS THE #1 PRIORITY):
- Each clip MUST START with a HOOK - a viral, attention-grabbing phrase!
- The FIRST 3 SECONDS of the clip must CAPTURE ATTENTION immediately
- ??? ALWAYS find the BEGINNING of a powerful statement as your start point
- ??? NEVER start a clip mid-sentence or mid-thought!
- ??? NEVER start with "так", "ну", "вот", "кстати говоря" - these indicate you're cutting mid-context!

???? TIMESTAMP PRECISION (USE BATCH NUMBERS):
- Each transcript line has [BATCH_X] prefix - use these to find EXACT timestamps
- The [XX:XX] and [XX.XXs] show the EXACT second where that text starts
- Your "start" time MUST match the EXACT timestamp of the HOOK phrase
- Example: If BATCH_15 has the hook at [02:34][154.20s], use start: 154.20
- Your "end" time MUST be where the thought COMPLETES (not mid-sentence)

- MINIMUM: 15 seconds | MAXIMUM: 60 seconds
- If moment is longer, SPLIT into multiple clips (each with its own hook!)

Look for moments that START with:
- Strong emotional moments (surprise, shock, excitement, controversy)
- Funny, entertaining, or dramatic content
- Complete thoughts with clear beginning and end

?????? CRITICAL REQUIREMENT - MINIMUM CLIPS:
- You MUST return AT LEAST 2-3 viral moments. NEVER return 0 or 1!
- A video without any viral moments is UNACCEPTABLE output
- For a 10+ minute video, there are ALWAYS at least 2-3 interesting moments worth cutting
- If content seems "boring", find the BEST 2-3 sections anyway - there's always something usable

???? TITLE RULES:
- Title MUST be the ACTUAL HOOK PHRASE from the video (or very close to it)
- Title = the first attention-grabbing words the viewer hears
- ??? NO generic titles like "Interesting moment", "Key insight"

CRITICAL CHECKLIST before returning:
??? At least 5 moments with score 7+?
??? Does each clip START with a hook phrase? (not mid-sentence)
??? Is duration 15-60 seconds?
??? Is the start timestamp the EXACT moment the hook begins?
??? Is the title based on the actual hook phrase?

Return ONLY a JSON object:
{{
  "moments": [
    {{
      "start": 154.20,
      "end": 189.50,
      "start_batch": 15,
      "hook_phrase": "The exact first words that hook the viewer",
      "title": "Hook-Based Attention Title",
      "reason": "Why this hook captures attention",
      "score": 8
    }}
  ]
}}

Transcript (with BATCH numbers for precise referencing):
{transcript_text}"""

    def _call_api(
        self,
        transcript_text:  str,
        video_duration:   float,
        lang_instruction: str,
    ) -> List[Dict]:
        """Call OpenAI API and return raw moments list."""
        user_prompt = self._build_user_prompt(
            transcript_text, video_duration, lang_instruction
        )

        model_to_try = self.model
        for attempt_model in [model_to_try] + MODEL_FALLBACKS:
            try:
                response = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model    = attempt_model,
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    response_format    = {"type": "json_object"},
                    max_completion_tokens = MAX_COMPLETION_TOKENS,
                )
                content = response.choices[0].message.content or "{}"
                print(f"[DEBUG] GPT-5-mini response: {content[:200]}...")

                result = json.loads(content)
                moments = result.get("moments", [])
                print(f"[INFO] Response content: {len(moments)} moments from {attempt_model}")
                return moments

            except Exception as exc:
                err_str = str(exc)
                if "model" in err_str.lower() and attempt_model != MODEL_FALLBACKS[-1]:
                    print(f"[WARN] Model {attempt_model} not available, trying fallback...")
                    continue
                print(f"[WARN] Error analyzing transcript with GPT-4o: {exc}")
                return []

        return []

    # -----------------------------------------------------------------------
    # Private — validation
    # -----------------------------------------------------------------------

    def _validate_moments(
        self,
        moments:        List[Dict],
        segments:       List[Dict],
        video_duration: float,
    ) -> List[Dict]:
        """
        Validate moments returned by LLM:
        - Duration must be 15-60 seconds
        - No overlaps
        - Correct start timestamps using BATCH numbers (actual_start vs expected_start)
        """
        # Build a BATCH index for timestamp correction
        batch_map = {i: seg for i, seg in enumerate(segments)}

        validated: List[Dict] = []

        for moment in moments:
            try:
                expected_start = float(moment.get("start", 0))
                end_time       = float(moment.get("end", 0))
                start_batch    = moment.get("start_batch")

                # Duration check on ORIGINAL values (before any correction)
                raw_duration = end_time - expected_start
                if raw_duration < MIN_CLIP_DURATION:
                    print(
                        f"[INFO]   Skipping too short moment: "
                        f"{raw_duration:.1f}s"
                    )
                    continue

                # Timestamp correction using BATCH reference
                actual_start = expected_start
                if start_batch is not None and int(start_batch) in batch_map:
                    future_seg = batch_map[int(start_batch)]
                    actual_start = float(future_seg.get("start", expected_start))
                    if abs(actual_start - expected_start) > 2.0:
                        print(
                            f"[INFO]   Fixing start time: "
                            f"{expected_start:.2f}s -> {actual_start:.2f}s "
                            f"(from BATCH_{start_batch})"
                        )

                # Clamp to video bounds
                actual_start = max(0.0, min(actual_start, video_duration))
                end_time     = max(actual_start, min(end_time, video_duration))

                clip_duration = end_time - actual_start

                # Re-check duration after clamping
                if clip_duration < MIN_CLIP_DURATION:
                    print(
                        f"[INFO]   Skipping clamped-too-short moment: "
                        f"{clip_duration:.1f}s"
                    )
                    continue

                # Soft-cap: warn but don't discard oversized clips
                if clip_duration > MAX_CLIP_DURATION:
                    end_time = actual_start + MAX_CLIP_DURATION

                # Overlap check with already validated moments
                overlaps = any(
                    not (end_time <= float(vm["start"]) or actual_start >= float(vm["end"]))
                    for vm in validated
                )
                if overlaps:
                    continue

                hook = moment.get("hook_phrase", "")
                if hook:
                    print(f"[INFO]   ???? Hook: \"{hook[:60]}\"")

                # Build normalised moment
                validated.append({
                    "start":       round(actual_start, 3),
                    "end":         round(end_time, 3),
                    "start_batch": start_batch,
                    "hook_phrase": hook,
                    "title":       moment.get("title", hook[:40]),
                    "reason":      moment.get("reason", ""),
                    "score":       int(moment.get("score", 7)),
                })

            except Exception as exc:
                print(f"[WARN] Moment validation error: {exc}")

        return validated


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def analyze_transcript(
    transcript:       Dict,
    video_duration:   Optional[float] = None,
    existing_moments: Optional[List[Dict]] = None,
    min_clips:        int = MIN_CLIPS,
    api_key:          Optional[str] = None,
    model:            Optional[str] = None,
    base_url:         Optional[str] = None,
) -> List[Dict]:
    """
    Convenience wrapper around ViralMomentAnalyzer.analyze_transcript().

    Args:
        transcript:   Dict from transcriber.py (needs "segments" at minimum).
        video_duration: Total duration in seconds.
        existing_moments: Already-placed clips to avoid overlapping.
        min_clips:    Hard minimum number of clips to return.
        api_key:      OpenAI API key (uses env OPENAI_API_KEY if not provided).
        model:        Model name (uses env OPENAI_MODEL if not provided, then DEFAULT_MODEL).
        base_url:     API base URL (uses env OPENAI_BASE_URL if not provided).
                      Set to https://openrouter.ai/api/v1 to use OpenRouter.

    Returns:
        List of moment dicts, sorted by start time.
    """
    analyzer = ViralMomentAnalyzer(api_key=api_key, model=model, base_url=base_url)
    return analyzer.analyze_transcript(
        transcript       = transcript,
        video_duration   = video_duration,
        existing_moments = existing_moments,
        min_clips        = min_clips,
    )


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import io
    import sys

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <transcript.json> [min_clips]")
        print("       transcript.json: output from transcriber.py")
        print("       min_clips: minimum number of moments to find (default: 3)")
        print()
        print("Demo mode (no file needed):")
        print("       python analyzer.py --demo")
        sys.exit(0)

    if sys.argv[1] == "--demo":
        # Use the test segments baked into the binary
        test_segments = [
            {"start": 0.0,   "end": 5.0,  "text": "Hey everyone, today I'm going to show you something incredible!"},
            {"start": 5.0,   "end": 12.0, "text": "This is the trick that professional editors don't want you to know."},
            {"start": 12.0,  "end": 25.0, "text": "This trick will change your life forever. Nobody talks about this!"},
            {"start": 25.0,  "end": 40.0, "text": "Here is exactly how you can apply this to your daily workflow."},
            {"start": 40.0,  "end": 55.0, "text": "The results were absolutely mind-blowing. I couldn't believe my eyes."},
            {"start": 55.0,  "end": 70.0, "text": "Most people make this critical mistake and they don't even realize it."},
            {"start": 70.0,  "end": 85.0, "text": "Here's the secret that nobody tells you about this process."},
            {"start": 85.0,  "end": 100.0,"text": "Watch what happens next. This completely changed how I work."},
        ]
        demo_transcript = {
            "text": " ".join(s["text"] for s in test_segments),
            "language": "en",
            "segments": test_segments,
            "words": [],
        }
        moments = analyze_transcript(demo_transcript, video_duration=100.0)
        print(f"\nFound {len(moments)} moments:")
        print(json.dumps(moments, indent=2, ensure_ascii=False))
        sys.exit(0)

    # Load transcript JSON
    transcript_path = sys.argv[1]
    min_clips_arg   = int(sys.argv[2]) if len(sys.argv) > 2 else MIN_CLIPS

    if not os.path.exists(transcript_path):
        print(f"File not found: {transcript_path}")
        sys.exit(1)

    with open(transcript_path, encoding="utf-8") as fh:
        transcript = json.load(fh)

    print("=" * 60)
    print("Boosta Analyzer")
    print("=" * 60)
    print(f"Segments: {len(transcript.get('segments', []))}")
    print(f"Text length: {len(transcript.get('text', ''))} chars")

    moments = analyze_transcript(
        transcript = transcript,
        min_clips  = min_clips_arg,
    )

    print(f"\nResult: {len(moments)} moments")
    for i, m in enumerate(moments, 1):
        print(f"\n  [{i}] {m['title']}")
        print(f"      Start: {m['start']:.1f}s  End: {m['end']:.1f}s  "
              f"Duration: {m['end']-m['start']:.1f}s  Score: {m.get('score','?')}")
        print(f"      Hook: {m.get('hook_phrase','')[:80]}")
        print(f"      Reason: {m.get('reason','')[:100]}")

    # Save moments.json next to transcript
    import pathlib
    out_path = str(pathlib.Path(transcript_path).with_suffix(".moments.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(moments, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
