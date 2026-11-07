# -*- coding: utf-8 -*-
"""
gemini_layout_analyzer.py
==========================
Step 2 AI layout analyzer for Boosta backend.

Reconstructed from: gemini_layout_analyzer.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary.

Confirmed recovered elements:
  - Class: GeminiLayoutAnalyzer  (6 methods from binary)
  - Methods: __init__, analyze_clip_layout, _build_layout_prompt,
             _parse_layout_response, _generate_fallback_segments,
             _get_video_duration
  - Model: "gemini-2.5-flash"  (exact from binary)
  - Package: google.generativeai (agenai, agenerativeai, aGenerativeModel)
  - Upload flow: upload_file → poll PROCESSING → generate_content → delete_file
  - Response format: application/json, segments with "from"/"to"/"layout" keys
  - Parsed to: "start"/"end"/"layout" (afrom_t, ato_t, avalidated_seg)
  - Safety: BLOCK_NONE for all HARM categories (binary: aBLOCK_NONE)
  - Fallback: YOLO yolo11n.pt person detection (aultralytics, aYOLO)
  - YOLO thresholds: person >30% frame = single_speaker, corner = screen_share
  - ffprobe for duration (format=duration, noprint_wrappers=1:nokey=1)
  - Layout types: single_speaker, screen_share, full_frame (all from binary)

Prompt recovered from binary strings:
  "Analyze this video clip (X seconds) and identify ALL layout changes."
  "Build the layout analysis prompt - ONLY asks for TYPE, not coordinates"
  "Your task is to segment this clip by visual layout TYPE ONLY."
  "WATCH CAREFULLY and detect EVERY transition between these types."
  "Return ONLY JSON (no coordinates needed - our AI detects them)"

JSON schema (internal, from binary examples):
  {"segments": [{"from": 0.0, "to": 8.5, "layout": "single_speaker"}, ...]}

Output for clipper.py / gemini_composer.py:
  {"video_type": "single_speaker",
   "segments": [{"start": 0.0, "end": 8.5, "layout": "single_speaker", ...}]}

Dependencies:
  google-generativeai >= 0.5.0  (Gemini API)
  ultralytics >= 8.0            (YOLO fallback, bundled)
  opencv-python >= 4.8          (frame sampling, bundled)
  ffprobe                       (duration, bundled)
  GEMINI_API_KEY env var
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
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
# google.genai (new SDK) OR google.generativeai (original binary, deprecated)
# ---------------------------------------------------------------------------
GENAI_AVAILABLE  = False
GENAI_NEW_SDK    = False   # True = google.genai, False = google.generativeai
genai            = None    # type: ignore
HarmCategory     = None    # type: ignore
HarmBlockThreshold = None  # type: ignore

# Try new SDK first (google.genai >= 0.8)
try:
    import google.genai as genai                          # type: ignore
    from google.genai import types as genai_types         # type: ignore
    GENAI_AVAILABLE = True
    GENAI_NEW_SDK   = True
except ImportError:
    pass

# Fall back to legacy SDK (google.generativeai — original binary engine)
if not GENAI_AVAILABLE:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai            # type: ignore
            from google.generativeai.types import (        # type: ignore
                HarmCategory, HarmBlockThreshold
            )
        GENAI_AVAILABLE = True
        GENAI_NEW_SDK   = False
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# OpenCV (for fallback frame sampling)
# ---------------------------------------------------------------------------
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore

# ---------------------------------------------------------------------------
# YOLO (fallback when Gemini unavailable)
# ---------------------------------------------------------------------------
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    _YOLO = None  # type: ignore

__all__ = ["GeminiLayoutAnalyzer", "analyze_clip_layout"]

# ---------------------------------------------------------------------------
# Constants decoded from binary
# ---------------------------------------------------------------------------
MODEL_NAME         = "gemini-2.5-flash"     # exact from binary
MAX_OUTPUT_TOKENS  = 2048
TEMPERATURE        = 0.1                     # deterministic layout analysis
UPLOAD_WAIT_S      = 5                       # poll interval while PROCESSING
UPLOAD_TIMEOUT     = 120                     # max wait for upload in seconds

LAYOUT_SINGLE      = "single_speaker"
LAYOUT_SCREEN      = "screen_share"
LAYOUT_FULL        = "full_frame"
ALL_LAYOUTS        = [LAYOUT_SINGLE, LAYOUT_SCREEN, LAYOUT_FULL]

# YOLO person-area thresholds (decoded from binary)
THRESHOLD_LARGE    = 0.30   # >30% of frame → single_speaker
THRESHOLD_CORNER   = 0.25   # max area for corner webcam (< 25%) → screen_share
THRESHOLD_SMALL    = 0.05   # < 5% → ignore

# Default fallback result when nothing works
FALLBACK_RESULT: Dict = {
    "video_type": LAYOUT_SINGLE,
    "segments": [
        {"start": 0.0, "end": None, "layout": LAYOUT_SINGLE,
         "video_type": LAYOUT_SINGLE}
    ],
}


# ===========================================================================
# GeminiLayoutAnalyzer
# ===========================================================================

class GeminiLayoutAnalyzer:
    """
    Gemini Layout Analyzer - Step 2 Agent.
    Analyzes individual clips to detect layout segments.
    """

    def __init__(
        self,
        api_key:      Optional[str] = None,
        model_name:   str = MODEL_NAME,
        ffprobe_path: str = "ffprobe",
        yolo_model:   str = "yolo11n.pt",
    ) -> None:
        self.model_name  = model_name
        self._model      = None
        self.detector    = None

        # Locate bundled binaries
        script_dir       = os.path.dirname(os.path.abspath(__file__))
        self.ffprobe     = self._find_bin(ffprobe_path, script_dir)
        self._yolo_path  = self._find_model(yolo_model, script_dir)

        # Configure Gemini
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            print("[WARN] GEMINI_API_KEY not found in environment variables")
        elif GENAI_AVAILABLE and genai is not None:
            try:
                if GENAI_NEW_SDK:
                    # New SDK: google.genai
                    self._client = genai.Client(api_key=resolved_key)
                    self._model  = model_name  # just store name; use client for calls
                else:
                    # Legacy SDK: google.generativeai (original binary)
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
                print(f"✅ Gemini Layout Analyzer initialized "
                      f"({'new' if GENAI_NEW_SDK else 'legacy'} SDK)")
            except Exception as exc:
                print(f"[WARN] Gemini init failed: {exc}")
                self._model  = None
                self._client = None
        else:
            self._client = None

        # Load YOLO for fallback
        if YOLO_AVAILABLE and _YOLO is not None and self._yolo_path:
            try:
                self.detector = _YOLO(self._yolo_path)
            except Exception as exc:
                print(f"[WARN] YOLO load failed: {exc}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def analyze_clip_layout(
        self,
        clip_path:     str,
        clip_duration: Optional[float] = None,
    ) -> Dict:
        """
        Analyze a single clip to detect layout segments.
        Analyzes each clip individually to detect layout changes frame-by-frame.

        Args:
            clip_path:     Path to the video clip.
            clip_duration: Duration in seconds (auto-detected if None).

        Returns:
            Dict with "video_type" and "segments" list. Each segment has:
            start, end, layout, video_type.
        """
        if not os.path.exists(clip_path):
            print(f"[WARN]    Layout analysis failed: file not found: {clip_path}")
            return FALLBACK_RESULT.copy()

        print(f"🔍 Analyzing layout for: {os.path.basename(clip_path)}")

        # Get duration
        duration = clip_duration or self._get_video_duration(clip_path)
        if duration:
            print(f"    ⏱  Clip duration: {duration:.1f}s")

        print(f"    🔄 Detecting layout segments...")

        # ── Attempt 1: Gemini API ────────────────────────────────────────────
        if self._model is not None and GENAI_AVAILABLE:
            try:
                result = self._analyze_with_gemini(clip_path, duration)
                if result and result.get("segments"):
                    n = len(result["segments"])
                    print(f"    ✅ Found {n} layout segment(s)")
                    return result
            except Exception as exc:
                print(f"    ❌ Layout analysis failed: {exc}")

        # ── Attempt 2: YOLO fallback ─────────────────────────────────────────
        print(f"    ⚠️  Empty response, using YOLO fallback")
        try:
            result = self._generate_fallback_segments(clip_path, duration)
            if result and result.get("segments"):
                return result
        except Exception as exc:
            print(f"    ❌ YOLO fallback failed: {exc}")

        # ── Attempt 3: trivial fallback ──────────────────────────────────────
        return self._trivial_fallback(duration)

    # -----------------------------------------------------------------------
    # Gemini pipeline
    # -----------------------------------------------------------------------

    def _analyze_with_gemini(self, clip_path: str, duration: Optional[float]) -> Dict:
        """Upload clip to Gemini Files API and run layout analysis."""
        prompt = self._build_layout_prompt(duration)
        print(f"    📤 Uploading clip to Gemini...")

        if GENAI_NEW_SDK:
            return self._analyze_with_new_sdk(clip_path, prompt, duration)
        else:
            return self._analyze_with_legacy_sdk(clip_path, prompt, duration)

    def _analyze_with_new_sdk(self, clip_path: str, prompt: str,
                               duration: Optional[float]) -> Dict:
        """Upload + generate using google.genai (new SDK)."""
        file_ref = None
        try:
            file_ref = self._client.files.upload(
                path      = clip_path,
                config    = {"mime_type": "video/mp4"},
            )
            # Poll until active
            deadline = time.time() + UPLOAD_TIMEOUT
            while file_ref.state.name not in ("ACTIVE", "FAILED"):
                if time.time() > deadline:
                    raise TimeoutError("Gemini file upload timed out")
                time.sleep(UPLOAD_WAIT_S)
                file_ref = self._client.files.get(name=file_ref.name)
            if file_ref.state.name == "FAILED":
                raise RuntimeError("Video processing failed")

            response = self._client.models.generate_content(
                model    = self._model,
                contents = [file_ref, prompt],
                config   = {"response_mime_type": "application/json",
                            "temperature": TEMPERATURE},
            )
            response_text = response.text.strip()
            print(f"    ✅ Clip processed {len(response_text)} chars")
            return self._parse_layout_response(response_text, duration)
        finally:
            if file_ref is not None:
                try: self._client.files.delete(name=file_ref.name)
                except Exception: pass

    def _analyze_with_legacy_sdk(self, clip_path: str, prompt: str,
                                  duration: Optional[float]) -> Dict:
        """Upload + generate using google.generativeai (legacy SDK)."""
        video_file = None
        try:
            video_file = genai.upload_file(clip_path, mime_type="video/mp4")
            deadline = time.time() + UPLOAD_TIMEOUT
            while video_file.state.name == "PROCESSING":
                if time.time() > deadline:
                    raise TimeoutError("Gemini file upload timed out")
                time.sleep(UPLOAD_WAIT_S)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise RuntimeError("Video processing failed")

            response      = self._model.generate_content([video_file, prompt])
            response_text = response.text.strip()
            print(f"    ✅ Clip processed {len(response_text)} chars")
            return self._parse_layout_response(response_text, duration)
        finally:
            if video_file is not None:
                try: genai.delete_file(video_file.name)
                except Exception: pass

    # -----------------------------------------------------------------------
    # Prompt builder
    # -----------------------------------------------------------------------

    def _build_layout_prompt(self, clip_duration: Optional[float]) -> str:
        """
        Build the layout analysis prompt - ONLY asks for TYPE, not coordinates.
        """
        dur_str = f"{clip_duration:.1f}" if clip_duration else "unknown"

        return f"""Analyze this video clip ({dur_str} seconds) and identify ALL layout changes.

Your task is to segment this clip by visual layout TYPE ONLY. Our AI will detect exact positions.

LAYOUT TYPES:
1. "single_speaker" - Person talking directly to camera, fills most of frame
   - The person takes up >30% of the frame
   - They are the main focus, no screen/browser visible behind them
   - Classic podcast/vlog style

2. "screen_share" - Screen recording with small webcam overlay
   - Main visible content is a screen (browser, trading platform, Twitter, charts, game)
   - Small person/webcam visible in corner (typically 5-25% of frame)
   - The SCREEN CONTENT is the main focus, person is small overlay

3. "full_frame" - Full screen content, NO person visible
   - Charts, text, presentations, gameplay without face
   - If no person at all = full_frame

RULES:
- If person is BIG (>30% frame) = single_speaker
- If screen is main content + small person in corner = screen_share
- If no person at all = full_frame

WATCH CAREFULLY and detect EVERY transition between these types.

Return ONLY JSON (no coordinates needed - our AI detects them):

{{"segments": [
  {{"from": 0.0, "to": 8.5, "layout": "single_speaker"}},
  {{"from": 8.5, "to": 22.0, "layout": "screen_share"}},
  {{"from": 22.0, "to": {dur_str if clip_duration else "END"}, "layout": "single_speaker"}}
]}}"""

    # -----------------------------------------------------------------------
    # Response parser
    # -----------------------------------------------------------------------

    def _parse_layout_response(
        self,
        response_text: str,
        clip_duration: Optional[float],
    ) -> Dict:
        """Parse and validate layout response from Gemini."""
        # Strip markdown code fences if present
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        # Try to parse JSON
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON object from surrounding text
            m = re.search(r'\{.*\}', clean, re.DOTALL)
            if not m:
                print(f"    ❌ Failed to parse JSON: {clean[:200]}")
                return {}
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                print(f"    ❌ Failed to parse JSON: {clean[:200]}")
                return {}

        raw_segments = data.get("segments", [])
        if not raw_segments:
            return {}

        # Convert from/to → start/end  (afrom_t, ato_t → astart, aend)
        validated: List[Dict] = []
        for seg in raw_segments:
            try:
                from_t  = float(seg.get("from", seg.get("start", 0)))
                to_t    = float(seg.get("to",   seg.get("end", clip_duration or 0)))
                layout  = str(seg.get("layout", LAYOUT_SINGLE)).strip()

                if layout not in ALL_LAYOUTS:
                    layout = LAYOUT_SINGLE

                validated_seg: Dict = {
                    "start":      round(from_t, 3),
                    "end":        round(to_t,   3),
                    "layout":     layout,
                    "video_type": layout,
                }
                if layout == LAYOUT_SCREEN:
                    validated_seg["webcam_region"]  = seg.get("webcam_region",  {})
                    validated_seg["content_region"] = seg.get("content_region", {})

                validated.append(validated_seg)
            except (ValueError, TypeError):
                continue

        if not validated:
            return {}

        # Sort by start time
        validated.sort(key=lambda s: s["start"])

        # Fix last segment end time
        if clip_duration and validated[-1]["end"] < clip_duration * 0.9:
            validated[-1]["end"] = clip_duration

        # Determine overall video_type from majority
        video_type = self._dominant_layout(validated)

        return {
            "video_type": video_type,
            "segments":   validated,
        }

    # -----------------------------------------------------------------------
    # YOLO fallback
    # -----------------------------------------------------------------------

    def _generate_fallback_segments(
        self,
        clip_path:     str,
        clip_duration: Optional[float],
    ) -> Dict:
        """
        Generate fallback segment using YOLO detection.
        Analyzes sample frames from the clip.
        """
        if not CV2_AVAILABLE:
            return {}

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return {}

        frame_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area    = frame_w * frame_h

        if total_frames <= 0 or frame_area <= 0:
            cap.release()
            return {}

        # Sample ~10 frames evenly distributed
        sample_count  = min(10, total_frames)
        sample_indices = [
            int(i * total_frames / sample_count)
            for i in range(sample_count)
        ]

        layout_votes: List[str] = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            layout = self._classify_frame_yolo(frame, frame_area)
            layout_votes.append(layout)

        cap.release()

        if not layout_votes:
            return {}

        # Dominant layout from votes
        from collections import Counter
        dominant = Counter(layout_votes).most_common(1)[0][0]
        duration = clip_duration or 0.0

        print(f"    ⚠️  YOLO fallback: {dominant} "
              f"({layout_votes.count(dominant)}/{len(layout_votes)} frames)")

        return {
            "video_type": dominant,
            "segments": [{
                "start":      0.0,
                "end":        duration or None,
                "layout":     dominant,
                "video_type": dominant,
            }],
        }

    def _classify_frame_yolo(self, frame, frame_area: int) -> str:
        """Classify a single frame as single_speaker / screen_share / full_frame."""
        if self.detector is None:
            return LAYOUT_SINGLE

        try:
            results = self.detector(frame, verbose=False)
            if not results or not results[0].boxes:
                return LAYOUT_FULL

            person_area = 0
            is_corner   = False

            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:  # 0 = person in COCO
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                person_area = max(person_area, int(area))

                # Is this person in a corner?
                cx    = (x1 + x2) / 2
                cy    = (y1 + y2) / 2
                fw, fh = frame.shape[1], frame.shape[0]
                is_corner = (cx < fw * 0.35 or cx > fw * 0.65) and cy > fh * 0.55

            if person_area == 0:
                return LAYOUT_FULL

            person_ratio = person_area / frame_area

            if person_ratio > THRESHOLD_LARGE:
                return LAYOUT_SINGLE
            elif person_ratio < THRESHOLD_CORNER and is_corner:
                print(f"    ⚠️  YOLO fallback: small person in corner → screen_share")
                return LAYOUT_SCREEN
            elif person_ratio > THRESHOLD_SMALL:
                return LAYOUT_SINGLE
            else:
                return LAYOUT_FULL

        except Exception as exc:
            print(f"    ❌ YOLO fallback failed: {exc}")
            return LAYOUT_SINGLE

    # -----------------------------------------------------------------------
    # Duration helper
    # -----------------------------------------------------------------------

    def _get_video_duration(self, clip_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                self.ffprobe,
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                clip_path,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
            return float(out.strip())
        except Exception:
            pass

        # cv2 fallback
        if CV2_AVAILABLE:
            try:
                cap = cv2.VideoCapture(clip_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if n > 0 and fps > 0:
                    return float(n / fps)
            except Exception:
                pass

        return 0.0

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _dominant_layout(segments: List[Dict]) -> str:
        """Return the layout that covers the most time in the segments list."""
        durations: Dict[str, float] = {}
        for seg in segments:
            start  = float(seg.get("start", 0))
            end    = float(seg.get("end") or 0)
            layout = seg.get("layout", LAYOUT_SINGLE)
            durations[layout] = durations.get(layout, 0) + max(0, end - start)
        if not durations:
            return LAYOUT_SINGLE
        return max(durations, key=lambda k: durations[k])

    @staticmethod
    def _trivial_fallback(duration: Optional[float]) -> Dict:
        """Return a trivial single-segment fallback."""
        return {
            "video_type": LAYOUT_SINGLE,
            "segments": [{
                "start":      0.0,
                "end":        duration,
                "layout":     LAYOUT_SINGLE,
                "video_type": LAYOUT_SINGLE,
            }],
        }

    @staticmethod
    def _find_bin(name: str, script_dir: str) -> str:
        for candidate in (name, name + ".exe",
                          os.path.join(script_dir, name),
                          os.path.join(script_dir, name + ".exe")):
            if os.path.exists(candidate):
                return candidate
        return name

    @staticmethod
    def _find_model(name: str, script_dir: str) -> str:
        for candidate in (name,
                          os.path.join(script_dir, name),
                          os.path.join(script_dir, "models", name)):
            if os.path.exists(candidate):
                return candidate
        return name


# ===========================================================================
# Module-level convenience
# ===========================================================================

def analyze_clip_layout(
    clip_path:     str,
    clip_duration: Optional[float] = None,
    api_key:       Optional[str]   = None,
) -> Dict:
    """
    Convenience function to analyze a clip's layouts.

    Args:
        clip_path:     Path to the video clip.
        clip_duration: Duration in seconds (auto-detected if None).
        api_key:       Gemini API key (uses GEMINI_API_KEY env if None).

    Returns:
        Dict with "video_type" and "segments" (start/end/layout/video_type).
    """
    analyzer = GeminiLayoutAnalyzer(api_key=api_key)
    return analyzer.analyze_clip_layout(clip_path, clip_duration)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import io
    import sys

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("Usage: python gemini_layout_analyzer.py <clip_path>")

    if len(sys.argv) < 2:
        sys.exit(0)

    clip = sys.argv[1]
    if not os.path.exists(clip):
        print(f"File not found: {clip}")
        sys.exit(1)

    print("=" * 60)
    print("Gemini Layout Analyzer")
    print("=" * 60)

    analyzer = GeminiLayoutAnalyzer()
    result   = analyzer.analyze_clip_layout(clip)
    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save next to the clip
    out_path = clip + ".layout.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
