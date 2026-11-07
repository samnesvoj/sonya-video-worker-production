"""
sonya_enhancer.py
=================
Central enrichment pipeline for all SONYA production modes.
All sub-modules are optional — missing keys/models/deps → warning + fallback.

Return dict keys:
    gemini_viral_moments   – list of viral moments from GeminiVideoAnalyzer
    layout_segments        – list of layout segments from GeminiLayoutAnalyzer
    crop_hints             – dict of crop region suggestions
    yolo_detections        – YOLO object detections per frame
    pose_detections        – YOLO pose keypoints per frame
    webcam_layout          – webcam/person bounding box layout info
    active_speaker_segments – speaking segments from lip_sync_detector
    word_timestamps        – word-level transcript from Transcriber
    warnings               – list of non-fatal warnings collected during enrichment
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from scripts.shared.model_paths import model_file
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def enrich_video_for_mode(
    input_video_path: str,
    output_dir: str,
    mode: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Run available enrichment sub-modules and return merged analysis dict.
    Every sub-module is optional: if unavailable → warning added, pipeline continues.
    """
    if params is None:
        params = {}

    result: Dict[str, Any] = {
        "gemini_viral_moments":   [],
        "layout_segments":        [],
        "crop_hints":             {},
        "yolo_detections":        [],
        "pose_detections":        [],
        "webcam_layout":          {},
        "active_speaker_segments": [],
        "word_timestamps":        [],
        "warnings":               [],
    }

    if not _flag("SONYA_ENHANCERS_ENABLED"):
        logger.info("[enhancer] SONYA_ENHANCERS_ENABLED=false — skipping all enrichment")
        result["warnings"].append("SONYA_ENHANCERS_ENABLED=false")
        return result

    def _step(name: str, progress: float, fn: Callable) -> None:
        try:
            value = fn()
            if value is not None:
                result[name] = value
            if progress_callback:
                progress_callback(name, progress)
        except Exception as exc:
            msg = f"{name} failed: {exc}"
            logger.warning("[enhancer] %s", msg)
            result["warnings"].append(msg)

    # ── Gemini Video Analyzer (viral moments) ─────────────────────────────────
    if _flag("GEMINI_DIRECT_VIDEO_ENABLED", "false"):
        def _gemini_viral():
            from scripts.shared.gemini.gemini_analyzer import GeminiVideoAnalyzer
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                result["warnings"].append("GEMINI_API_KEY not set — gemini_viral_moments skipped")
                return None
            analyzer = GeminiVideoAnalyzer(api_key=api_key)
            # Returns list of moment dicts: [{start, end, duration, relevance, ...}]
            moments = analyzer.analyze_video(
                video_path=input_video_path,
                num_clips=params.get("max_clips", 5),
            )
            return moments if isinstance(moments, list) else []
        _step("gemini_viral_moments", 0.10, _gemini_viral)

    # ── Gemini Layout Analyzer ────────────────────────────────────────────────
    if _flag("GEMINI_LAYOUT_ENABLED"):
        def _gemini_layout():
            from scripts.shared.gemini.gemini_layout_analyzer import GeminiLayoutAnalyzer
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                result["warnings"].append("GEMINI_API_KEY not set — layout_segments skipped")
                return None
            analyzer = GeminiLayoutAnalyzer(api_key=api_key)
            # Returns dict: {"video_type": str, "segments": [{start, end, layout, video_type}]}
            layout_result = analyzer.analyze_clip_layout(clip_path=input_video_path)
            return layout_result.get("segments", []) if isinstance(layout_result, dict) else []
        _step("layout_segments", 0.20, _gemini_layout)

    # ── YOLO common object detection ──────────────────────────────────────────
    if _flag("YOLO_COMMON_ENABLED"):
        def _yolo_common():
            from scripts.shared.vision.yolo_common import detect_common
            model_path = model_file("common", "yolo11n.pt")
            if not model_path.exists():
                result["warnings"].append("yolo11n.pt not found — yolo_detections skipped")
                return None
            return detect_common(input_video_path, str(model_path))
        _step("yolo_detections", 0.35, _yolo_common)

    # ── YOLO pose detection ───────────────────────────────────────────────────
    if _flag("YOLO_POSE_ENABLED"):
        def _yolo_pose():
            from scripts.shared.vision.yolo_common import detect_pose
            model_path = model_file("common", "yolo11n-pose.pt")
            if not model_path.exists():
                result["warnings"].append("yolo11n-pose.pt not found — pose_detections skipped")
                return None
            return detect_pose(input_video_path, str(model_path))
        _step("pose_detections", 0.50, _yolo_pose)

    # ── Webcam / person detector ──────────────────────────────────────────────
    if _flag("WEBCAM_DETECTOR_ENABLED"):
        def _webcam():
            from scripts.shared.vision.webcam_detector import WebcamDetector
            model_path = model_file("common", "webcam_detector.pt")
            if not model_path.exists():
                result["warnings"].append("webcam_detector.pt not found — webcam_layout skipped")
                return None
            det = WebcamDetector(str(model_path))
            boxes = det.detect(input_video_path)
            # Wrap in a layout dict for consistent interface
            return {"boxes": boxes, "method": "webcam_detector", "count": len(boxes)}
        _step("webcam_layout", 0.60, _webcam)

    # ── Transcriber (word-level) ──────────────────────────────────────────────
    # A caller that already ran ASR (SONYA Desktop's ASR seam) passes the
    # artifact path; its words are used as-is instead of transcribing again.
    precomputed = params.get("word_timestamps_path")
    if precomputed:
        def _load_words():
            import json
            with open(precomputed, encoding="utf-8") as fh:
                return json.load(fh).get("words", [])
        _step("word_timestamps", 0.72, _load_words)
    elif _flag("WORD_LEVEL_SUBTITLES_ENABLED"):
        def _transcribe():
            from scripts.shared.transcription.transcriber import Transcriber
            t = Transcriber()
            # Returns: {"text", "language", "segments", "words": [{word, start, end}]}
            transcript = t.transcribe(audio_file=input_video_path)
            # Return word-level list directly
            return transcript.get("words", [])
        _step("word_timestamps", 0.72, _transcribe)

    # ── Lip sync / active speaker detection ───────────────────────────────────
    if _flag("LIP_SYNC_DETECTION_ENABLED", "false"):
        def _lip_sync():
            from scripts.shared.speaker.lip_sync_detector import analyze_speakers_in_clip
            # Pipeline-safe: returns empty dict if MediaPipe unavailable
            lip_result = analyze_speakers_in_clip(video_path=input_video_path)
            # Extract speaking_segments for active_speaker_segments
            return lip_result.get("speaking_segments", [])
        _step("active_speaker_segments", 0.85, _lip_sync)

    # ── Crop hints (always runs, based on whatever was collected) ─────────────
    def _crop_hints():
        from scripts.shared.crop.smart_crop_adapter import get_crop_hints
        return get_crop_hints(
            input_video_path,
            yolo_detections=result.get("yolo_detections"),
            webcam_layout=result.get("webcam_layout"),
            pose_detections=result.get("pose_detections"),
            layout_segments=result.get("layout_segments"),
        )
    _step("crop_hints", 0.95, _crop_hints)

    if progress_callback:
        progress_callback("done", 1.0)

    logger.info(
        "[enhancer] mode=%s completed. keys_collected=%s warnings=%d",
        mode,
        [k for k, v in result.items() if v and k != "warnings"],
        len(result["warnings"]),
    )
    return result
