# -*- coding: utf-8 -*-
"""
clipper.py
==========
Central composition orchestrator for Boosta backend.

Reconstructed from: clipper.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary (string + constant extraction).

Confirmed recovered elements:
  - Class VideoClipper  (all 30+ methods from binary)
  - 5 subtitle styles: karaoke / highlight / box-highlight / word-pair / word-reveal
  - Frame pipeline: _apply_frame_with_mask (PNG rounded mask), _apply_frame_simple
  - Logo overlay: _apply_logo  (position %, opacity, circle mask)
  - LUT color filter: _get_color_filter_ffmpeg  (12 .cube presets)
  - Layout pipeline: GeminiLayoutAnalyzer -> GeminiComposer -> VideoComposer fallback
  - Transcription: Transcriber (faster-whisper) with JSON cache
  - ASS header: PlayResX=1080 PlayResY=1920 ScriptType=v4.00+
  - Output suffixes: _vertical, _filtered, _framed, _logo, _subbed, _subs.ass, _mask.png
  - Feature flags: USE_GEMINI_COMPOSER, USE_TRANSCRIBER, CLOUD_MODE
  - Note: "SYNCED with VideoSubtitlePreview.jsx for identical output"

Dependencies:
    pip install opencv-python pillow numpy
    # Optional (graceful fallback if absent):
    # gemini_composer.py, video_composer.py, lip_sync_detector.py, cropper.py
    # transcriber.py (faster-whisper based)
    # gemini_layout_analyzer.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional deps — all fail-safe
# ---------------------------------------------------------------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None          # type: ignore
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    Image = ImageDraw = None    # type: ignore
    PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional Boosta modules
# ---------------------------------------------------------------------------
USE_GEMINI_COMPOSER = False
try:
    from gemini_composer import GeminiComposer, get_video_codec_params  # type: ignore
    USE_GEMINI_COMPOSER = True
except ImportError:
    GeminiComposer = None       # type: ignore
    def get_video_codec_params(path: str) -> Dict:
        return {"-vcodec": "libx264", "-preset": "fast", "-crf": "23"}

try:
    from video_composer import VideoComposer   # type: ignore
    VIDEO_COMPOSER_AVAILABLE = True
except ImportError:
    VideoComposer = None        # type: ignore
    VIDEO_COMPOSER_AVAILABLE = False

try:
    from gemini_layout_analyzer import GeminiLayoutAnalyzer  # type: ignore
    USE_LAYOUT_ANALYZER = os.environ.get("USE_LAYOUT_ANALYZER", "1") != "0"
except ImportError:
    GeminiLayoutAnalyzer = None  # type: ignore
    USE_LAYOUT_ANALYZER = False

USE_TRANSCRIBER = False
try:
    from transcriber import Transcriber         # type: ignore
    USE_TRANSCRIBER = True
except ImportError:
    Transcriber = None          # type: ignore

__all__ = ["VideoClipper"]

# ---------------------------------------------------------------------------
# Font map  (decoded from binary)
# ---------------------------------------------------------------------------
FONT_MAP: Dict[str, str] = {
    "Anton":          "Anton.ttf",
    "Archivo Black":  "ArchivoBlack.ttf",
    "Bangers":        "Bangers.ttf",
    "Bebas Neue":     "BebasNeue.ttf",
    "bebas neue":     "BebasNeue.ttf",
    "DM Sans":        "DMSans.ttf",
    "dm sans":        "DMSans.ttf",
    "Fredoka":        "Fredoka.ttf",
    "Inter":          "Inter.ttf",
    "Lato":           "Lato.ttf",
    "Lexend":         "Lexend.ttf",
    "Montserrat":     "Montserrat.ttf",
    "Noto Sans":      "NotoSans-Regular.ttf",
    "noto sans":      "NotoSans-Regular.ttf",
    "Nunito":         "Nunito.ttf",
    "Open Sans":      "OpenSans.ttf",
    "Oswald":         "Oswald.ttf",
    "Outfit":         "Outfit.ttf",
    "Pacifico":       "Pacifico.ttf",
    "Permanent Marker": "PermanentMarker.ttf",
    "Plus Jakarta Sans": "PlusJakartaSans.ttf",
    "Poppins":        "Poppins.ttf",
    "Quicksand":      "Quicksand.ttf",
    "Raleway":        "Raleway.ttf",
    "Righteous":      "Righteous.ttf",
    "Roboto":         "Roboto.ttf",
    "Rubik":          "Rubik.ttf",
    "Russo One":      "RussoOne.ttf",
    "Space Grotesk":  "SpaceGrotesk.ttf",
    "Titan One":      "TitanOne.ttf",
}

# Cyrillic-capable fonts
CYRILLIC_FONTS = {"Roboto", "Noto Sans", "Open Sans", "Montserrat",
                  "Nunito", "Lato", "Raleway", "Inter", "Lexend"}

# LUT color presets (decoded: uses .cube LUT files — same as WebGL frontend)
COLOR_PRESETS = {
    "none", "bw", "cinematic", "cold", "faded",
    "high-contrast", "matte", "sepia", "teal-orange",
    "vibrant", "vintage", "warm",
}

# Canvas
CANVAS_W = 1080
CANVAS_H = 1920


# ===========================================================================
# VideoClipper
# ===========================================================================

class VideoClipper:
    """
    Central composition orchestrator.

    Pipeline per clip:
        Step 1  cut_clip()              — raw cut from source video (copy codec)
        Step 2  analyze_clip_layout()   — GeminiLayoutAnalyzer or fallback
        Step 3  compose_clip()          — GeminiComposer / VideoComposer / simple crop
        Step 4  _apply_subtitles()      — burn styled ASS subtitles (optional)
        Step 5  _apply_logo()           — logo/banner overlay (optional)

    Also supports config-based styling (subtitles, filters, etc.)
    """

    def __init__(
        self,
        base_dir:   str = ".",
        fonts_dir:  Optional[str] = None,
        luts_dir:   Optional[str] = None,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        transcriber=None,
        layout_analyzer=None,
    ) -> None:
        print("[AI] VideoClipper initializing...")
        self.base_dir   = os.path.abspath(base_dir)
        script_dir      = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg     = self._find_bin(ffmpeg_path,  script_dir)
        self.ffprobe    = self._find_bin(ffprobe_path, script_dir)

        # Asset directories
        self.fonts_dir  = fonts_dir  or os.path.join(script_dir, "fonts")
        self.luts_dir   = luts_dir   or os.path.join(script_dir, "luts")

        # Build font path map
        self.font_map: Dict[str, str] = {}
        for name, fname in FONT_MAP.items():
            full = os.path.join(self.fonts_dir, fname)
            if os.path.exists(full):
                self.font_map[name] = full

        # Transcriber
        if transcriber is not None:
            self.transcriber = transcriber
        elif USE_TRANSCRIBER and Transcriber is not None:
            try:
                self.transcriber = Transcriber()
                print("  Transcriber loaded")
            except Exception as exc:
                print(f"  [WARN] Transcriber init failed: {exc}")
                self.transcriber = None
        else:
            self.transcriber = None
            if not USE_TRANSCRIBER:
                print("  [WARN] Whisper transcriber: NOT available")

        # Layout analyzer
        if layout_analyzer is not None:
            self.layout_analyzer = layout_analyzer
        elif USE_LAYOUT_ANALYZER and GeminiLayoutAnalyzer is not None:
            try:
                self.layout_analyzer = GeminiLayoutAnalyzer()
                print("  GeminiLayoutAnalyzer loaded")
            except Exception as exc:
                print(f"  [WARN] GeminiLayoutAnalyzer init failed: {exc}")
                self.layout_analyzer = None
        else:
            self.layout_analyzer = None

        # Gemini composer
        self.gemini_composer: Optional[object] = None
        if USE_GEMINI_COMPOSER and GeminiComposer is not None:
            try:
                self.gemini_composer = GeminiComposer()
                print("  GeminiComposer loaded")
            except Exception as exc:
                print(f"  [WARN] GeminiComposer init failed: {exc}")

        # Video composer (algorithmic fallback)
        self.video_composer: Optional[object] = None
        if VIDEO_COMPOSER_AVAILABLE and VideoComposer is not None:
            try:
                self.video_composer = VideoComposer()
                print("  VideoComposer loaded")
            except Exception as exc:
                print(f"  [WARN] VideoComposer init failed: {exc}")

        print("  VideoClipper ready")

    # -----------------------------------------------------------------------
    # Public API — clip cutting
    # -----------------------------------------------------------------------

    def cut_clip(
        self,
        input_path: str,
        start_time: float,
        end_time:   float,
        output_path: Optional[str] = None,
        vertical:   bool = True,
    ) -> Optional[str]:
        """
        Cut a single clip from video and optionally convert to vertical format.

        Uses ffmpeg with copy codec for instant processing.
        vertical: Convert to 9:16 vertical format (default: True)
        """
        if output_path is None:
            stem = Path(input_path).stem
            output_path = os.path.join(
                os.path.dirname(input_path),
                f"{stem}_clip_{int(start_time)}-{int(end_time)}.mp4",
            )

        print(f"[CLIP] Cutting clip: {start_time:.1f}s - {end_time:.1f}s -> {output_path}")
        duration = end_time - start_time

        raw_cut = output_path.replace(".mp4", "_raw.mp4")

        cmd = [
            self.ffmpeg, "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            raw_cut,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if not os.path.exists(raw_cut):
                print(f"  [FAIL] Raw clip not created")
                return None
            print(f"  [OK] Raw clip extracted: {raw_cut}")
        except Exception as exc:
            print(f"  [FAIL] Failed to cut clip: {exc}")
            return None

        if vertical:
            vert = self.convert_to_vertical(raw_cut, output_path)
            try:
                os.remove(raw_cut)
            except OSError:
                pass
            return vert or output_path
        else:
            shutil.move(raw_cut, output_path)
            return output_path

    def cut_all_moments(
        self,
        input_path: str,
        moments: List[Dict],
        output_dir: str,
        vertical: bool = True,
        config: Optional[Dict] = None,
    ) -> List[str]:
        """
        Cut all moments from a source video.

        moments: list of {"start": float, "end": float, "title": str, ...}
        Returns list of output paths.

        Step 1: Cut raw clips from source video
        Step 2: Analyze each clip's layout with Gemini (per-clip analysis)
        Step 3: Apply composition based on detected layout segments
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[str] = []

        if USE_LAYOUT_ANALYZER and self.layout_analyzer is not None:
            print(f"[INFO] Using GeminiLayoutAnalyzer for per-clip analysis (Step 2)")

        for idx, moment in enumerate(moments):
            start    = float(moment.get("start", 0))
            end      = float(moment.get("end", start + 60))
            title    = moment.get("title", f"moment_{idx}")
            safe     = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:50]

            output_path = os.path.join(output_dir, f"{idx+1:02d}_{safe}.mp4")
            print(f"\n[CLIP] Moment {idx+1}/{len(moments)}: {title}")

            if config:
                result = self.cut_clip_with_gemini(
                    input_path=input_path,
                    start_time=start,
                    end_time=end,
                    output_path=output_path,
                    config=config,
                    moment=moment,
                )
            else:
                result = self.cut_clip(
                    input_path=input_path,
                    start_time=start,
                    end_time=end,
                    output_path=output_path,
                    vertical=vertical,
                )

            if result and os.path.exists(result):
                results.append(result)
            else:
                results.append(output_path)

        print(f"\n[OK] {len(results)} clips created in {output_dir}")
        return results

    def cut_clip_with_gemini(
        self,
        input_path:  str,
        start_time:  float,
        end_time:    float,
        output_path: str,
        config:      Optional[Dict] = None,
        moment:      Optional[Dict] = None,
        recut:       bool = False,
    ) -> Optional[str]:
        """
        Full pipeline:
          Step 1: Cut raw clip
          Step 2: GeminiLayoutAnalyzer (or fallback)
          Step 3: Smart composition
          Step 4: Subtitles (if config)
          Step 5: Logo (if config)
        Uses cached transcription if available (for recut mode).
        """
        clip_dir  = os.path.dirname(output_path)
        clip_stem = Path(output_path).stem
        os.makedirs(clip_dir, exist_ok=True)

        # --- Step 1: Cut raw clip ---
        print(f"  [STEP1] Step 1: Cutting raw clip from source video")
        raw_clip = os.path.join(clip_dir, f"{clip_stem}_raw.mp4")
        duration = end_time - start_time
        cut_cmd  = [
            self.ffmpeg, "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            raw_clip,
        ]
        try:
            subprocess.run(cut_cmd, capture_output=True, check=True, timeout=120)
        except Exception as exc:
            print(f"  [FAIL] Failed to cut clip: {exc}")
            return None

        if not os.path.exists(raw_clip):
            print("  [FAIL] Raw clip not created")
            return None
        print(f"  [OK] Raw clip extracted: {raw_clip}")

        # --- Background transcription (always, for cache) ---
        print("  [INFO] Background transcription (for future use)...")
        # This runs for all clips regardless of subtitle config,
        # so recuts always have cached transcription.
        self._ensure_transcription(raw_clip, clip_dir, clip_stem)

        # --- Step 2: Layout analysis ---
        layout_segments = None
        video_type      = None

        if recut:
            loaded = self._load_saved_layouts(clip_dir)
            if loaded is not None:
                layout_segments, video_type = loaded
                print(f"  [OK] Step 2: Using saved layout ({len(layout_segments)} segments)")
            else:
                print("  [WARN] Step 2: Layout analyzer not available, using fallback")

        if layout_segments is None and self.layout_analyzer is not None:
            print("  [STEP2] Step 2: Sending raw clip to Gemini Layout Analyzer...")
            try:
                analysis = self.layout_analyzer.analyze_clip_layout(raw_clip)
                layout_segments = analysis.get("segments", [])
                video_type      = analysis.get("video_type", "single_speaker")
                self._save_layout_analysis(clip_dir, analysis)
                print(f"  [OK] Gemini detected {len(layout_segments)} layout segment(s)")
            except Exception as exc:
                print(f"  [WARN] Layout analysis failed: {exc}")
                print("  [WARN] Step 2: Layout analyzer not available, using fallback")

        # --- Step 3: Composition ---
        composed_path = os.path.join(clip_dir, f"{clip_stem}_vertical.mp4")

        if layout_segments is not None and self.gemini_composer is not None:
            print("  [STEP3] Step 3: Applying Gemini composition...")
            try:
                # Override movies mode
                if config and config.get("movies") or (moment and moment.get("type") == "movies"):
                    for seg in layout_segments:
                        seg["video_type"] = "letterbox_full_frame"
                    print("  [INFO] Movies mode: overriding saved layouts with letterbox_full_frame")

                # split_never: talking_heads -> wide_shot
                if config and config.get("split") == "never":
                    for seg in layout_segments:
                        if seg.get("video_type") == "talking_heads":
                            seg["video_type"] = "wide_shot"
                    print("  [INFO] split_never: converted talking_heads to wide_shot")

                result = self.gemini_composer.compose_with_segments(
                    raw_clip, composed_path, layout_segments
                )
                if result and os.path.exists(result):
                    print(f"  [OK] Gemini composition successful, got {len(layout_segments)} segment(s)")
                    composed_path = result
                else:
                    print("  [WARN] Gemini composition failed, falling back to algorithmic")
                    composed_path = self._fallback_composition(raw_clip, composed_path, video_type)
            except Exception as exc:
                print(f"  [WARN] Gemini composition failed, using fallback: {exc}")
                composed_path = self._fallback_composition(raw_clip, composed_path, video_type)
        else:
            print("  [STEP3] Step 3: Using algorithmic composition...")
            composed_path = self._fallback_composition(raw_clip, composed_path, video_type)

        # Clean raw clip
        try:
            os.remove(raw_clip)
        except OSError:
            pass

        current_path = composed_path
        if not current_path or not os.path.exists(current_path):
            print("  [FAIL] Composition failed completely")
            return None

        # --- Step 4: Subtitles + color filter + frame ---
        print("  [STEP4] Step 4: Applying effects...")
        current_path = self._apply_color_filter(current_path, clip_dir, clip_stem, config)

        if config and config.get("subtitles", {}).get("enabled", False):
            print("  [STEP4] Step 4: Adding subtitles...")
            subbed = self._apply_subtitles(
                current_path, clip_dir, clip_stem, config,
                transcript_dir=clip_dir,
            )
            if subbed and os.path.exists(subbed):
                try:
                    os.remove(current_path)
                except OSError:
                    pass
                current_path = subbed
                print("  [OK] Subtitles added successfully")
            else:
                print("  [WARN] Failed to add subtitles:")
        else:
            print("  [INFO] Subtitles: disabled")

        # --- Step 5: Logo ---
        if config and config.get("logo"):
            print("  [STEP5] Step 5: Adding logo...")
            logo_result = self._apply_logo(current_path, clip_dir, clip_stem, config)
            if logo_result and os.path.exists(logo_result):
                try:
                    os.remove(current_path)
                except OSError:
                    pass
                current_path = logo_result
                print(f"  [OK] Logo added")
            else:
                print("  [WARN] Failed to add logo:")

        # Move to final output path
        if current_path != output_path:
            shutil.move(current_path, output_path)

        print(f"  [OK] Gemini clip created: {output_path}")
        return output_path

    # -----------------------------------------------------------------------
    # Vertical conversion
    # -----------------------------------------------------------------------

    def convert_to_vertical(
        self,
        input_path:  str,
        output_path: Optional[str] = None,
        video_type:  str = "single_speaker",
    ) -> Optional[str]:
        """
        Convert horizontal clip to vertical 9:16 format (TikTok/Shorts).
        Path to vertical video
        """
        if output_path is None:
            output_path = input_path.replace(".mp4", "_vertical.mp4")

        print(f"[CONV] Converting to vertical format: {input_path}")

        # Use VideoComposer if available
        if self.video_composer is not None:
            try:
                result = self.video_composer.create_smart_composition(
                    input_path, output_path
                )
                if result and os.path.exists(result):
                    print(f"  [OK] Vertical video created: {result}")
                    return result
            except Exception as exc:
                print(f"  [WARN] Smart composition failed, falling back to simple crop: {exc}")

        # Simple center crop 9:16
        return self._simple_crop_916(input_path, output_path)

    def _simple_crop_916(
        self, input_path: str, output_path: str
    ) -> Optional[str]:
        """Simple center 9:16 crop — fast fallback."""
        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-vf", "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos,setsar=1",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            if os.path.exists(output_path):
                print(f"  [OK] Simple crop created: {output_path}")
                return output_path
        except Exception as exc:
            print(f"  [FAIL] Simple crop failed: {exc}")
        return None

    def _fallback_composition(
        self,
        input_path:   str,
        output_path:  str,
        video_type:   Optional[str] = None,
    ) -> str:
        """Fallback composition using GeminiComposer with default single_speaker layout."""
        if self.gemini_composer is not None:
            try:
                print("  [INFO] Using GeminiComposer for AI-guided layouts")
                # Use single speaker layout by default
                seg = [{"video_type": video_type or "single_speaker",
                        "start": 0, "end": None}]
                result = self.gemini_composer.compose_with_segments(
                    input_path, output_path, seg
                )
                if result and os.path.exists(result):
                    print(f"  [OK] Fallback clip created: {result}")
                    return result
            except Exception as exc:
                print(f"  [WARN] GeminiComposer fallback failed: {exc}")

        if self.video_composer is not None:
            try:
                print("  [INFO] Using local AI composer (YOLO+MediaPipe)...")
                result = self.video_composer.create_smart_composition(
                    input_path, output_path
                )
                if result and os.path.exists(result):
                    return result
            except Exception as exc:
                print(f"  [WARN] VideoComposer fallback failed: {exc}")

        # Last resort: simple crop
        result = self._simple_crop_916(input_path, output_path)
        return result or output_path

    # -----------------------------------------------------------------------
    # Layout analysis
    # -----------------------------------------------------------------------

    def analyze_clip_layout(
        self, clip_path: str, save: bool = True
    ) -> Dict:
        """
        Analyze clip layout using GeminiLayoutAnalyzer.
        Returns analysis dict with 'segments' and 'video_type'.
        """
        if self.layout_analyzer is not None:
            try:
                analysis = self.layout_analyzer.analyze_clip_layout(clip_path)
                if save:
                    clip_dir = os.path.dirname(clip_path)
                    self._save_layout_analysis(clip_dir, analysis)
                    print(f"  [INFO] Layout analysis saved to: layout_analysis.json")
                return analysis
            except Exception as exc:
                print(f"  [WARN] Layout analysis failed: {exc}")

        return {"segments": [], "video_type": "single_speaker"}

    def _save_layout_analysis(self, clip_dir: str, analysis: Dict) -> None:
        """Save layout analysis results to JSON file for debugging and analysis"""
        analysis_file = os.path.join(clip_dir, "layout_analysis.json")
        try:
            data = {
                "analyzed_at": datetime.now().isoformat(),
                "video_type":  analysis.get("video_type", "unknown"),
                "segments":    analysis.get("segments", []),
            }
            with open(analysis_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False,
                          default=str)
        except Exception as exc:
            print(f"  [WARN] Could not save layout_analysis.json: {exc}")

    def _load_saved_layouts(
        self, clip_dir: str
    ) -> Optional[Tuple[List[Dict], Optional[str]]]:
        """Load previously saved layout analysis for recut mode. Also loads video_type."""
        analysis_file = os.path.join(clip_dir, "layout_analysis.json")
        if not os.path.exists(analysis_file):
            print("  [WARN] No saved layout_analysis.json found")
            return None
        try:
            with open(analysis_file, encoding="utf-8") as fh:
                data = json.load(fh)
            segments   = data.get("segments", [])
            video_type = data.get("video_type")
            print(f"  [OK] {len(segments)} saved layouts from layout_analysis.json")
            return segments, video_type
        except Exception as exc:
            print(f"  [WARN] Error loading layout_analysis.json: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Transcription
    # -----------------------------------------------------------------------

    def _ensure_transcription(
        self,
        clip_path:  str,
        clip_dir:   str,
        clip_stem:  str,
    ) -> Optional[Dict]:
        """
        Transcribe clip and burn subtitles with config styling.
        This runs for all clips regardless of subtitle config,
        so recuts always have cached transcription.
        """
        cached = self._load_cached_transcription(clip_dir, clip_stem)
        if cached is not None:
            size = len(json.dumps(cached))
            print(f"  [OK] Using cached transcription ({size} bytes)")
            return cached

        if self.transcriber is None:
            print("  [WARN] Transcriber not available")
            return None

        print("  [INFO] Transcribing with Whisper...")
        try:
            result = self.transcriber.transcribe(clip_path)
            if result:
                self._save_transcription(clip_dir, clip_stem, result)
                words = len(result.get("words", []))
                print(f"  [OK] Transcription cached ({words} words)")
                return result
        except Exception as exc:
            print(f"  [WARN] Transcription failed: {exc}")

        return None

    def _load_cached_transcription(
        self, clip_dir: str, clip_stem: str
    ) -> Optional[Dict]:
        """Load cached transcription for a clip"""
        transcripts_file = os.path.join(clip_dir, "transcripts.json")
        if not os.path.exists(transcripts_file):
            return None
        try:
            with open(transcripts_file, encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get(clip_stem)
        except Exception:
            return None

    def _save_transcription(
        self, clip_dir: str, clip_stem: str, transcript: Dict
    ) -> None:
        """Save transcription to cache file"""
        transcripts_file = os.path.join(clip_dir, "transcripts.json")
        existing: Dict = {}
        if os.path.exists(transcripts_file):
            try:
                with open(transcripts_file, encoding="utf-8") as fh:
                    existing = json.load(fh)
            except Exception:
                existing = {}

        existing[clip_stem] = {
            **transcript,
            "transcribed_at": datetime.now().isoformat(),
        }
        with open(transcripts_file, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False, default=str)

    # -----------------------------------------------------------------------
    # Color filter
    # -----------------------------------------------------------------------

    def _get_color_filter_ffmpeg(self, color_preset: str) -> str:
        """
        Get FFmpeg filter string for color preset.
        Uses .cube LUT files — SAME files as WebGL frontend for 100% match!
        """
        if not color_preset or color_preset == "none":
            return ""

        lut_file = os.path.join(self.luts_dir, f"{color_preset}.cube")
        if not os.path.exists(lut_file):
            return ""

        # lut3d filter — path must use forward slashes and escape colons
        lut_path_slash = lut_file.replace("\\", "/").replace(":", "\\:")
        return f"lut3d='{lut_path_slash}'"

    def _apply_color_filter(
        self,
        input_path: str,
        clip_dir:   str,
        clip_stem:  str,
        config:     Optional[Dict],
    ) -> str:
        """Apply LUT color filter. Returns output path (may be same as input if no filter)."""
        if not config:
            return input_path

        color_preset = (
            config.get("colorFilter")
            or config.get("color_filter")
            or config.get("color_preset")
            or "none"
        )

        if not color_preset or color_preset == "none":
            return input_path

        color_ff = self._get_color_filter_ffmpeg(color_preset)
        if not color_ff:
            return input_path

        print(f"  [INFO] Applying color filter: {color_preset}")
        output_path = os.path.join(clip_dir, f"{clip_stem}_filtered.mp4")

        codec = get_video_codec_params(input_path)
        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-vf", color_ff,
            "-c:v", codec.get("-vcodec", "libx264"),
            "-crf", codec.get("-crf", "23"),
            "-preset", codec.get("-preset", "fast"),
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            if os.path.exists(output_path):
                return output_path
        except subprocess.CalledProcessError as exc:
            print(f"  [WARN] Color filter failed: {exc.stderr.decode(errors='replace')[:200]}")

        return input_path

    def _apply_color_filter_only(
        self,
        input_path:  str,
        output_path: str,
        config:      Dict,
    ) -> str:
        """
        Apply color filter and/or frame without subtitles.
        Returns output_path on success, input_path on failure.
        """
        clip_dir  = os.path.dirname(output_path)
        clip_stem = Path(output_path).stem
        result = self._apply_color_filter(input_path, clip_dir, clip_stem, config)
        if result != input_path:
            shutil.move(result, output_path)
            return output_path
        return input_path

    # -----------------------------------------------------------------------
    # Frame filter
    # -----------------------------------------------------------------------

    def _get_frame_params(self, frame_config: Dict, vid_w: int, vid_h: int) -> Dict:
        """Get frame parameters for FFmpeg."""
        x_pct = float(frame_config.get("posX", 0))
        y_pct = float(frame_config.get("posY", 0))
        w_pct = float(frame_config.get("width", 100))
        h_pct = float(frame_config.get("height", 100))
        return {
            "x":      int(vid_w * x_pct / 100),
            "y":      int(vid_h * y_pct / 100),
            "width":  int(vid_w * w_pct / 100),
            "height": int(vid_h * h_pct / 100),
        }

    def _get_frame_filter_ffmpeg(self, frame_config: Dict) -> str:
        """
        Get simple frame filter (no rounded corners -
        use _apply_frame_with_mask for that).
        """
        if not frame_config or not frame_config.get("enabled"):
            return ""

        x_pct   = float(frame_config.get("posX", 0))
        y_pct   = float(frame_config.get("posY", 0))
        w_pct   = float(frame_config.get("width", 100))
        h_pct   = float(frame_config.get("height", 100))
        bg_color = frame_config.get("bgColor", "#000000").lstrip("#")

        visible_x = int(CANVAS_W * x_pct / 100)
        visible_y = int(CANVAS_H * y_pct / 100)
        visible_w = int(CANVAS_W * w_pct / 100)
        visible_h = int(CANVAS_H * h_pct / 100)

        # Scale video to visible area then pad to canvas
        scale = f"scale={visible_w}:{visible_h}"
        pad   = (
            f"pad={CANVAS_W}:{CANVAS_H}:"
            f"(ow-iw)/2:(oh-ih)/2:black,"
            f"crop={visible_w}:{visible_h}:{visible_x}:{visible_y}"
        )
        return f"{scale},{pad}"

    def _create_rounded_mask(
        self,
        mask_path: str,
        width:     int,
        height:    int,
        radius:    int,
    ) -> bool:
        """Create a grayscale PNG mask with rounded corners (white=visible, black=transparent)."""
        if not PIL_AVAILABLE:
            return False
        try:
            img  = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([0, 0, width - 1, height - 1],
                                   radius=radius, fill=255)
            img.save(mask_path)
            return True
        except Exception as exc:
            print(f"  [WARN] Mask creation error: {exc}")
            return False

    def _create_circle_mask(
        self, mask_path: str, size: int
    ) -> bool:
        """Create a grayscale circle mask (white=visible, black=transparent)."""
        if not PIL_AVAILABLE:
            return False
        try:
            img  = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(img)
            draw.ellipse([0, 0, size - 1, size - 1], fill=255)
            img.save(mask_path)
            return True
        except Exception as exc:
            print(f"  [WARN] Circle mask creation error: {exc}")
            return False

    def _apply_frame_with_mask(
        self,
        input_path:   str,
        output_path:  str,
        frame_config: Dict,
    ) -> Optional[str]:
        """Apply frame with rounded corners using PNG mask overlay."""
        if not frame_config or not frame_config.get("enabled"):
            return None

        x_pct  = float(frame_config.get("posX", 0))
        y_pct  = float(frame_config.get("posY", 0))
        w_pct  = float(frame_config.get("width", 100))
        h_pct  = float(frame_config.get("height", 100))
        radius = int(frame_config.get("borderRadius", 0))

        visible_x = int(CANVAS_W * x_pct / 100)
        visible_y = int(CANVAS_H * y_pct / 100)
        visible_w = int(CANVAS_W * w_pct / 100)
        visible_h = int(CANVAS_H * h_pct / 100)
        needs_rounded = radius > 0

        print(f"  [INFO] Applying frame: x={visible_x} y={visible_y} w={visible_w} h={visible_h}")

        if needs_rounded and PIL_AVAILABLE:
            mask_path = output_path.replace(".mp4", "_mask.png")
            ok = self._create_rounded_mask(mask_path, visible_w, visible_h, radius)
            if ok:
                print(f"  [OK] Mask created: {mask_path}")
                # Frame pipeline:
                # 1. Video scales to visible_w x visible_h
                # 2. Frame mask is applied at x,y position with size w,h
                filter_complex = (
                    f"[0:v]scale={visible_w}:{visible_h},"
                    f"format=rgba[vid];"
                    f"[1:v]format=gray[mask];"
                    f"[vid][mask]alphamerge[masked];"
                    f"[bg][masked]overlay={visible_x}:{visible_y}:shortest=1"
                )
                bg_cmd = (
                    f"color=c=0x000000:s={CANVAS_W}x{CANVAS_H}"
                    f":r=30[bg];"
                )
                full_fc = bg_cmd + filter_complex
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", input_path,
                    "-i", mask_path,
                    "-filter_complex", full_fc,
                    "-map", "[bg]",
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-c:a", "copy",
                    "-pix_fmt", "yuv420p",
                    output_path,
                ]
                try:
                    subprocess.run(cmd, capture_output=True, check=True, timeout=300)
                    try:
                        os.remove(mask_path)
                    except OSError:
                        pass
                    if os.path.exists(output_path):
                        return output_path
                except Exception as exc:
                    print(f"  [WARN] Mask overlay error: {exc}")
                    try:
                        os.remove(mask_path)
                    except OSError:
                        pass

        # Fallback: simple rectangular frame
        return self._apply_frame_simple(input_path, output_path, frame_config)

    def _apply_frame_simple(
        self,
        input_path:   str,
        output_path:  str,
        frame_config: Dict,
    ) -> Optional[str]:
        """Apply simple rectangular frame (fast fallback, no rounded corners)."""
        x_pct = float(frame_config.get("posX", 0))
        y_pct = float(frame_config.get("posY", 0))
        w_pct = float(frame_config.get("width", 100))
        h_pct = float(frame_config.get("height", 100))

        visible_x = int(CANVAS_W * x_pct / 100)
        visible_y = int(CANVAS_H * y_pct / 100)
        visible_w = int(CANVAS_W * w_pct / 100)
        visible_h = int(CANVAS_H * h_pct / 100)

        vf = (
            f"scale={visible_w}:{visible_h},"
            f"pad={CANVAS_W}:{CANVAS_H}:{visible_x}:{visible_y}:black"
        )
        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            if os.path.exists(output_path):
                return output_path
        except Exception as exc:
            print(f"  [WARN] Frame failed, continuing without: {exc}")
        return None

    # -----------------------------------------------------------------------
    # Logo overlay
    # -----------------------------------------------------------------------

    def _apply_logo(
        self,
        input_path: str,
        clip_dir:   str,
        clip_stem:  str,
        config:     Dict,
    ) -> Optional[str]:
        """
        Apply logo/banner overlay to video.

        logo_config: Logo settings from config with keys:
          - savedPath: path to logo file
          - posX: horizontal position (0-100%)
          - posY: vertical position (0-100%)
          - opacity: opacity (0-100%)
        """
        logo_config = config.get("logo", {})
        if not logo_config:
            return None

        logo_path = logo_config.get("savedPath", "")
        if not logo_path or not os.path.exists(logo_path):
            print(f"  [WARN] Logo file not found: {logo_path}")
            return None

        pos_x    = float(logo_config.get("posX", 50))
        pos_y    = float(logo_config.get("posY", 5))
        opacity  = float(logo_config.get("opacity", 100)) / 100.0
        logo_w   = int(logo_config.get("logoWidth",
                       logo_config.get("width", CANVAS_W // 4)))
        is_circle = logo_config.get("circle", False)

        print(f"  [INFO] Logo config: posX={pos_x:.0f}% posY={pos_y:.0f}%")
        print(f"  [INFO] Logo file: {logo_path}")

        overlay_x = int(CANVAS_W * pos_x / 100 - logo_w / 2)
        overlay_y = int(CANVAS_H * pos_y / 100)
        output_path = os.path.join(clip_dir, f"{clip_stem}_logo.mp4")

        print(f"  [INFO] Logo pos: x={overlay_x} y={overlay_y}")
        print("  [INFO] Running FFmpeg overlay...")

        logo_filters: List[str] = []

        if is_circle and PIL_AVAILABLE:
            mask_path = output_path.replace(".mp4", "_mask.png")
            ok = self._create_circle_mask(mask_path, logo_w)
            if ok:
                print(f"  [OK] Mask created: {mask_path}")
                logo_filters.append(
                    f"[1:v]scale={logo_w}:-1,format=rgba[logo_raw];"
                    f"[2:v]format=gray[circ];"
                    f"[logo_raw][circ]alphamerge[logo]"
                )
            else:
                logo_filters.append(f"[1:v]scale={logo_w}:-1[logo]")
                mask_path = None
        else:
            logo_filters.append(f"[1:v]scale={logo_w}:-1[logo]")
            mask_path = None

        alpha_hex = format(int(opacity * 255), "02x")
        overlay_filter = (
            f"[0:v][logo]overlay=x={overlay_x}:y={overlay_y}"
            f"-(overlay_h/2):shortest=1[out]"
        )
        logo_filters.append(overlay_filter)
        logo_chain = ";".join(logo_filters)

        inputs = [self.ffmpeg, "-y", "-i", input_path, "-i", logo_path]
        if mask_path and os.path.exists(mask_path if mask_path else ""):
            inputs += ["-i", mask_path]

        cmd = inputs + [
            "-filter_complex", logo_chain,
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            if mask_path:
                try:
                    os.remove(mask_path)
                except OSError:
                    pass
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  [OK] Logo applied ({size_mb:.1f} MB)")
                return output_path
            print("  [WARN] Logo output file not created:")
        except subprocess.CalledProcessError as exc:
            print(f"  [WARN] Logo overlay error: {exc.stderr.decode(errors='replace')[:200]}")
        return None

    # -----------------------------------------------------------------------
    # Subtitle pipeline
    # -----------------------------------------------------------------------

    def _apply_subtitles(
        self,
        input_path:     str,
        clip_dir:       str,
        clip_stem:      str,
        config:         Dict,
        transcript_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Transcribe clip and burn subtitles with config styling.
        subtitles_config: Subtitle settings from config
        Path to subtitled video or None
        SYNCED with VideoSubtitlePreview.jsx for identical output.
        """
        subtitles_config = config.get("subtitles", {})
        if not subtitles_config or not subtitles_config.get("enabled", False):
            print("  [INFO] Subtitles: disabled")
            return None

        print("  [INFO] Subtitles: ENABLED (will add to clips)")

        # Get transcript
        tdir      = transcript_dir or clip_dir
        transcript = self._load_cached_transcription(tdir, clip_stem)
        if transcript is None:
            transcript = self._ensure_transcription(input_path, tdir, clip_stem)

        if not transcript:
            print("  [WARN] No speech detected")
            return None

        words = transcript.get("words", [])
        if not words:
            words = _words_from_segments(transcript.get("segments", []))
        if not words:
            print("  [WARN] No speech detected in clip")
            return None

        # Generate ASS file
        ass_path = os.path.join(clip_dir, f"{clip_stem}_subs.ass")
        style    = subtitles_config.get("animation_type",
                   subtitles_config.get("animation", "karaoke"))

        print(f"  [INFO] Burning subtitles... (style={style})")

        try:
            ass_content = self._generate_styled_ass(words, subtitles_config, style)
            with open(ass_path, "w", encoding="utf-8") as fh:
                fh.write(ass_content)
            print(f"  [OK] ASS file created with '{style}' animation")
        except Exception as exc:
            print(f"  [WARN] Subtitle error: {exc}")
            return None

        # Burn ASS into video
        output_path = os.path.join(clip_dir, f"{clip_stem}_subbed.mp4")
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")

        # Font and margin
        font_name = subtitles_config.get("font", "Montserrat")
        font_path = self.font_map.get(font_name, "")
        fonts_dir_slash = self.fonts_dir.replace("\\", "/")

        # Check cyrillic
        all_text = " ".join(w.get("word", "") for w in words)
        has_cyrillic = any(ord(c) > 0x400 for c in all_text)
        if has_cyrillic and font_name not in CYRILLIC_FONTS:
            font_name = "Noto Sans"
            print(f"  [WARN] Cyrillic text detected, '{font_name}' fallback used")

        margin_v  = int(subtitles_config.get("marginV",
                        subtitles_config.get("margin_v", 80)))
        ass_align = int(subtitles_config.get("ass_alignment", 2))

        vf_parts = [
            f"ass='{ass_escaped}'"
            f":fontsdir='{fonts_dir_slash}'"
            f":force_style='Alignment={ass_align},"
            f"MarginV={margin_v}'"
        ]
        vf_string = ",".join(vf_parts)

        codec = get_video_codec_params(input_path)
        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-vf", vf_string,
            "-c:v", codec.get("-vcodec", "libx264"),
            "-crf", codec.get("-crf", "23"),
            "-preset", codec.get("-preset", "fast"),
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            stderr = result.stderr.decode(errors="replace")
            if result.returncode != 0:
                print(f"  [WARN] FFmpeg warning: {stderr[:200]}")
            if os.path.exists(output_path):
                print("  [OK] Subtitles burned successfully")
                try:
                    os.remove(ass_path)
                except OSError:
                    pass
                return output_path
        except Exception as exc:
            print(f"  [WARN] Subtitle error: {exc}")

        return None

    # -----------------------------------------------------------------------
    # ASS subtitle generation
    # -----------------------------------------------------------------------

    def _generate_styled_ass(
        self,
        words:             List[Dict],
        subtitles_config:  Dict,
        style:             str = "karaoke",
    ) -> str:
        """
        Generate ASS subtitle file with config styling.
        Supports: karaoke / highlight / box-highlight / word-pair / word-reveal
        """
        # ── Helpers ─────────────────────────────────────────────────────────
        def hex_to_ass(hex_color: str, alpha: int = 0) -> str:
            """Convert #RRGGBB to ASS &HAABBGGRR format."""
            h = hex_color.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            r, g, b = h[0:2], h[2:4], h[4:6]
            a_hex = format(255 - alpha, "02X")
            return f"&H{a_hex}{b}{g}{r}"

        def format_text(text: str, case: str) -> str:
            if case == "uppercase":
                return text.upper()
            if case == "lowercase":
                return text.lower()
            return text

        # ── Style params from config ─────────────────────────────────────
        font_name    = subtitles_config.get("font", "Montserrat")
        font_size    = int(subtitles_config.get("size", 72))
        text_color   = subtitles_config.get("text_color",
                       subtitles_config.get("textColor", "#FFFFFF"))
        outline_color = subtitles_config.get("outline_color",
                        subtitles_config.get("outlineColor",
                        subtitles_config.get("strokeColor", "#000000")))
        stroke_width = float(subtitles_config.get("stroke_width",
                             subtitles_config.get("strokeWidth", 2.0)))
        shadow       = float(subtitles_config.get("shadow", 0))
        bold         = int(subtitles_config.get("bold", -1))
        italic       = int(subtitles_config.get("italic", 0))
        underline    = int(subtitles_config.get("underline", 0))
        ass_align    = int(subtitles_config.get("ass_alignment", 2))
        margin_v     = int(subtitles_config.get("marginV",
                           subtitles_config.get("margin_v", 80)))
        margin_side  = int(subtitles_config.get("margin_side", 60))
        text_case    = subtitles_config.get("textCase",
                       subtitles_config.get("text_case", "normal"))
        words_per_line = int(subtitles_config.get("wordsPerLine",
                             subtitles_config.get("words_per_line",
                             subtitles_config.get("words_per_group", 3))))

        # Highlight / glow config
        highlight_color = subtitles_config.get("highlightColor",
                          subtitles_config.get("highlight_color", "#FFE600"))
        glow_color      = subtitles_config.get("glowColor",
                          subtitles_config.get("glow_color", "#FFE600"))
        glow_intensity  = float(subtitles_config.get("glowIntensity",
                                subtitles_config.get("glow_intensity", 0)))
        box_color       = subtitles_config.get("bgColor",
                          subtitles_config.get("box_color", "#000000"))
        box_padding     = int(subtitles_config.get("boxPadding",
                              subtitles_config.get("box_padding", 10)))

        # ASS colours
        primary_ass  = hex_to_ass(text_color)
        outline_ass  = hex_to_ass(outline_color)
        back_ass     = hex_to_ass("#000000", alpha=200)

        # Bold print
        print(f"  [INFO] Bold: {bold}, Italic: {italic}")
        print(f"  [INFO] Animation: {style}")

        # ── ASS header ───────────────────────────────────────────────────
        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {CANVAS_W}\n"
            f"PlayResY: {CANVAS_H}\n"
            "ScaledBorderAndShadow: yes\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font_name},{font_size},"
            f"{primary_ass},&H00FFFFFF,{outline_ass},{back_ass},"
            f"{bold},{italic},{underline},0,"
            f"100,100,0,0,1,{stroke_width:.0f},{shadow:.0f},"
            f"{ass_align},{margin_side},{margin_side},{margin_v},1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        # ── Dispatch to style writer ─────────────────────────────────────
        if style == "karaoke":
            events = self._write_karaoke_subtitles(
                words, words_per_line, text_case, hex_to_ass, format_text,
                highlight_color, glow_intensity, glow_color,
            )
        elif style == "highlight":
            events = self._write_highlight_subtitles(
                words, words_per_line, text_case, hex_to_ass, format_text,
                highlight_color, primary_ass,
            )
        elif style == "box-highlight":
            events = self._write_box_highlight_subtitles(
                words, words_per_line, text_case, hex_to_ass, format_text,
                highlight_color, box_color, box_padding,
            )
        elif style in ("word-pair", "pair"):
            events = self._write_pair_subtitles(
                words, words_per_line, text_case, format_text,
            )
        elif style in ("word-reveal", "reveal"):
            events = self._write_reveal_subtitles(
                words, words_per_line, text_case, format_text,
            )
        else:
            # Default: karaoke
            events = self._write_karaoke_subtitles(
                words, words_per_line, text_case, hex_to_ass, format_text,
                highlight_color, glow_intensity, glow_color,
            )

        return header + events

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        """Format seconds to ASS timestamp H:MM:SS.cc"""
        total_cs = int(round(seconds * 100))
        cs       = total_cs % 100
        total_s  = total_cs // 100
        secs     = total_s % 60
        total_m  = total_s  // 60
        mins     = total_m  % 60
        hours    = total_m  // 60
        return f"{hours}:{mins:02d}:{secs:02d}.{cs:02d}"

    @staticmethod
    def _group_words_into_phrases(
        words: List[Dict],
        n:     int = 3,
    ) -> List[Dict]:
        """Group individual words into phrases of N words"""
        if not words:
            return []
        phrases: List[Dict] = []
        for i in range(0, len(words), n):
            group = words[i:i + n]
            phrases.append({
                "start":       group[0].get("start", 0.0),
                "end":         group[-1].get("end", group[0].get("start", 0.0) + 1.0),
                "text":        " ".join(w.get("word", w.get("text", "")) for w in group),
                "words":       group,
            })
        return phrases

    # ── Subtitle style writers ────────────────────────────────────────────

    def _write_karaoke_subtitles(
        self,
        words:           List[Dict],
        words_per_group: int,
        text_case:       str,
        hex_to_ass,
        format_text,
        highlight_color: str,
        glow_intensity:  float,
        glow_color:      str,
    ) -> str:
        """
        Karaoke style: word-by-word color highlight using ASS karaoke tags.
        Uses ASS karaoke tags for smooth animation.
        """
        phrases  = self._group_words_into_phrases(words, words_per_group)
        hl_ass   = hex_to_ass(highlight_color)
        lines: List[str] = []

        for phrase in phrases:
            p_start = phrase["start"]
            p_end   = phrase["end"]
            t_start = self._format_ass_time(p_start)
            t_end   = self._format_ass_time(p_end)

            # Build karaoke text: {\\kDuration}Word
            parts: List[str] = []
            phrase_words = phrase.get("words", [])
            for w in phrase_words:
                word_start = w.get("start", p_start)
                word_end   = w.get("end", word_start + 0.3)
                duration_cs = max(1, int((word_end - word_start) * 100))
                text = format_text(w.get("word", w.get("text", "")), text_case)
                parts.append(f"{{\\k{duration_cs}}}{text}")

            # Glow prefix
            glow_prefix = ""
            if glow_intensity > 0:
                glow_bl   = max(1, int(glow_intensity * 10))
                glow_col  = hex_to_ass(glow_color)
                glow_prefix = f"{{\\blur{glow_bl}\\c{glow_col}}}"

            karaoke_text = glow_prefix + " ".join(parts)

            line = (
                f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
                f"{{\\K0\\c&HFFFFFF&}}{karaoke_text}\n"
            )
            lines.append(line)

        return "".join(lines)

    def _write_highlight_subtitles(
        self,
        words:           List[Dict],
        words_per_group: int,
        text_case:       str,
        hex_to_ass,
        format_text,
        highlight_color: str,
        normal_color_ass: str,
    ) -> str:
        """
        Highlight style: one word at a time is shown highlighted,
        others are normal color.
        """
        phrases  = self._group_words_into_phrases(words, words_per_group)
        hl_ass   = hex_to_ass(highlight_color)
        lines: List[str] = []

        for phrase in phrases:
            p_words = phrase.get("words", [])
            p_start = phrase["start"]
            p_end   = phrase["end"]
            all_texts = [format_text(w.get("word", w.get("text", "")), text_case)
                         for w in p_words]

            for hi, w in enumerate(p_words):
                w_start   = w.get("start", p_start)
                w_end     = w.get("end", w_start + 0.3)
                t_start   = self._format_ass_time(w_start)
                t_end     = self._format_ass_time(w_end)

                parts = []
                for j, txt in enumerate(all_texts):
                    if j == hi:
                        parts.append(f"{{\\c{hl_ass}}}{txt}{{\\c{normal_color_ass}}}")
                    else:
                        parts.append(txt)
                highlight_text = " ".join(parts)

                lines.append(
                    f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
                    f"{highlight_text}\n"
                )

        return "".join(lines)

    def _write_box_highlight_subtitles(
        self,
        words:           List[Dict],
        words_per_group: int,
        text_case:       str,
        hex_to_ass,
        format_text,
        highlight_color: str,
        box_color:       str,
        box_padding:     int,
    ) -> str:
        """
        Box-highlight style: highlighted word has a colored background box.
        """
        phrases  = self._group_words_into_phrases(words, words_per_group)
        hl_ass   = hex_to_ass(highlight_color)
        box_ass  = hex_to_ass(box_color)
        lines: List[str] = []

        for phrase in phrases:
            p_words = phrase.get("words", [])
            p_start = phrase["start"]

            for w in p_words:
                w_start = w.get("start", p_start)
                w_end   = w.get("end", w_start + 0.3)
                t_start = self._format_ass_time(w_start)
                t_end   = self._format_ass_time(w_end)
                text    = format_text(w.get("word", w.get("text", "")), text_case)

                box_text = (
                    f"{{\\c{hl_ass}\\3c{box_ass}\\bord{box_padding}}}"
                    f"{text}"
                    f"{{\\bord2}}"
                )
                lines.append(
                    f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
                    f"{box_text}\n"
                )

        return "".join(lines)

    def _write_pair_subtitles(
        self,
        words:           List[Dict],
        words_per_group: int,
        text_case:       str,
        format_text,
    ) -> str:
        """
        Word-pair style: groups of N words shown simultaneously,
        each group timed to its first-to-last word span.
        """
        phrases = self._group_words_into_phrases(words, words_per_group)
        lines: List[str] = []

        for phrase in phrases:
            t_start = self._format_ass_time(phrase["start"])
            t_end   = self._format_ass_time(phrase["end"])
            text    = format_text(phrase["text"], text_case)

            lines.append(
                f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
                f"{text}\n"
            )

        return "".join(lines)

    def _write_reveal_subtitles(
        self,
        words:           List[Dict],
        words_per_group: int,
        text_case:       str,
        format_text,
    ) -> str:
        """
        Word-reveal style: each word appears at its timestamp and stays
        until the phrase ends (accumulating reveal).
        """
        phrases = self._group_words_into_phrases(words, words_per_group)
        lines: List[str] = []

        for phrase in phrases:
            p_words = phrase.get("words", [])
            p_end   = phrase["end"]
            accumulated = ""

            for i, w in enumerate(p_words):
                w_start = w.get("start", phrase["start"])
                w_end   = p_end  # reveal: word stays until phrase end
                text    = format_text(w.get("word", w.get("text", "")), text_case)
                accumulated = (accumulated + " " + text).strip()

                t_start = self._format_ass_time(w_start)
                t_end   = self._format_ass_time(w_end)

                lines.append(
                    f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
                    f"{accumulated}\n"
                )

        return "".join(lines)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_bin(name: str, script_dir: str) -> str:
        """
        Resolve a binary name to an absolute path.
        Priority: local script_dir/<name> -> local script_dir/<name>.exe -> PATH.
        On Linux the .exe variant won't exist, so it falls through to PATH.
        """
        import shutil
        local = os.path.join(script_dir, name)
        if os.path.isfile(local):
            return local
        local_exe = local + ".exe"
        if os.path.isfile(local_exe):
            return local_exe
        # Last resort: rely on system PATH (works on Linux out of the box)
        on_path = shutil.which(name)
        return on_path if on_path else name

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                self.ffprobe, "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json", video_path,
            ]
            out = subprocess.check_output(cmd, timeout=30, stderr=subprocess.DEVNULL)
            data = json.loads(out)
            return float(data["format"]["duration"])
        except Exception:
            if CV2_AVAILABLE and cv2 is not None:
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    return count / fps
                except Exception:
                    pass
        return 0.0

    def compose_clip(
        self,
        clip_path:   str,
        output_path: Optional[str] = None,
        config:      Optional[Dict] = None,
        recut:       bool = False,
    ) -> Optional[str]:
        """
        Apply full composition pipeline to an already-cut clip.
        Useful for re-processing existing clips with new config.
        """
        if output_path is None:
            output_path = clip_path.replace(".mp4", "_composed.mp4")

        clip_dir  = os.path.dirname(clip_path)
        clip_stem = Path(clip_path).stem

        current = clip_path

        # Color filter
        if config:
            current = self._apply_color_filter(current, clip_dir, clip_stem, config)

        # Subtitles
        if config and config.get("subtitles", {}).get("enabled"):
            subbed = self._apply_subtitles(current, clip_dir, clip_stem, config)
            if subbed and os.path.exists(subbed):
                if current != clip_path:
                    try:
                        os.remove(current)
                    except OSError:
                        pass
                current = subbed

        # Logo
        if config and config.get("logo"):
            logo_r = self._apply_logo(current, clip_dir, clip_stem, config)
            if logo_r and os.path.exists(logo_r):
                if current != clip_path:
                    try:
                        os.remove(current)
                    except OSError:
                        pass
                current = logo_r

        if current != output_path:
            shutil.move(current, output_path)
        return output_path

    def compose_clip_auto(
        self,
        clip_path:   str,
        output_path: Optional[str] = None,
        config:      Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Auto-compose: layout analysis + composition + effects in one call.
        """
        if output_path is None:
            output_path = clip_path.replace(".mp4", "_auto.mp4")

        clip_dir  = os.path.dirname(clip_path)
        clip_stem = Path(clip_path).stem

        # Ensure vertical
        vert_path = clip_path.replace(".mp4", "_vertical.mp4")
        if not os.path.exists(vert_path):
            vert_path = self.convert_to_vertical(clip_path, vert_path) or clip_path

        return self.compose_clip(vert_path, output_path, config)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _words_from_segments(segments: List[Dict]) -> List[Dict]:
    """Flatten segment-level words into a flat word list."""
    words: List[Dict] = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            words.extend(seg_words)
        else:
            # Create a fake single-word entry from segment text
            words.append({
                "word":  seg.get("text", "").strip(),
                "start": float(seg.get("start", 0)),
                "end":   float(seg.get("end", 0)),
            })
    return words


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python clipper.py <input.mp4> <start> <end> [output.mp4]")
        print("       python clipper.py <input.mp4> moments.json <output_dir>")
        sys.exit(1)

    inp   = sys.argv[1]
    start = sys.argv[2]
    end   = sys.argv[3]
    out   = sys.argv[4] if len(sys.argv) > 4 else None

    clipper = VideoClipper()

    if start.endswith(".json"):
        # moments mode
        with open(start, encoding="utf-8") as fh:
            moments = json.load(fh)
        results = clipper.cut_all_moments(inp, moments, end or "output_clips")
        for r in results:
            print(f"  {r}")
    else:
        result = clipper.cut_clip(inp, float(start), float(end), out)
        print(f"Done: {result}")
