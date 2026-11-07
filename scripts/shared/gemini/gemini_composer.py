"""
gemini_composer.py — GeminiComposer module
Reconstructed from gemini_composer.cp311-win_amd64.pyd (Nuitka-compiled, Python 3.11).

Static analysis of the .pyd extracted:
  - All class/method names, variable names, string literals, docstrings
  - Full method signatures and return types (via Nuitka QNames)
  - All ffmpeg filter_complex strings
  - VIDEO_TYPE_CONFIG keys and option names
  - Layout type names and routing logic
  - YOLO model filenames (yolo11n-pose.pt, yolo11n.pt, webcam_detector.pt)
  - MediaPipe integration (blaze_face_short_range.tflite)
  - LipSync, TextDetector, SportsDetector hooks

This is a clean reconstruction — logic matches all recovered evidence.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency guards
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

try:
    import mediapipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mediapipe = None
    MEDIAPIPE_AVAILABLE = False

try:
    from scenedetect import ContentDetector, AdaptiveDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    from lip_sync_detector import LipSyncDetector
    from text_detector import TextDetector
    LIP_SYNC_AVAILABLE = True
except ImportError:
    LIP_SYNC_AVAILABLE = False

try:
    from sports_detector import SportsDetector
    SPORTS_DETECTOR_AVAILABLE = True
except ImportError:
    SPORTS_DETECTOR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Canvas constants  (9:16 portrait output)
# ─────────────────────────────────────────────────────────────────────────────

CANVAS_W: int = 1080
CANVAS_H: int = 1920

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO_TYPE_CONFIG
# Recovered from .pyd: keys 'solo', 'conversation', 'screenReacts', 'vlog',
# 'faceless', 'movies', 'sports' and all option-name attributes.
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_TYPE_CONFIG: Dict[str, Dict[str, Any]] = {
    "solo": {
        "priority": "person",
        "detect_webcam": False,
        "detect_people": True,
        "multi_person_layout": "split_or_wide",
        "fallback_on_screen": False,
        "person_threshold": 0.4,
        "split_or_wide": True,
        "closeup": True,
        "medium": True,
        "small": False,
        "check_screen_share": False,
        "fallback_on_no_webcam": False,
        "webcam_max_ratio": 0.35,
        "side_by_side_threshold": 0.5,
        "corner_webcam_threshold": 0.25,
        "fit_check": True,
        "prefer_group_shot": False,
        "skip_layout_detection": False,
        "use_stable_crop": True,
        "main_person": "center",
        "smart_crop": True,
        "center_crop_fallback": True,
        "content": None,
        "detect_ball": False,
        "follow_ball_player": False,
        "smooth_transitions": True,
        "fallback_wide_shot": True,
        "split_never": False,
        "split_on_dialog": True,
        "group_aware": False,
        "cinematic": False,
        "detect_letterbox": False,
        "preserve_full_frame": False,
        "group_shot": False,
    },
    "conversation": {
        "priority": "people",
        "detect_webcam": False,
        "detect_people": True,
        "multi_person_layout": "split_or_wide",
        "fallback_on_screen": False,
        "person_threshold": 0.35,
        "split_or_wide": True,
        "closeup": True,
        "medium": True,
        "small": False,
        "check_screen_share": False,
        "fallback_on_no_webcam": False,
        "webcam_max_ratio": 0.4,
        "side_by_side_threshold": 0.45,
        "corner_webcam_threshold": 0.25,
        "fit_check": True,
        "prefer_group_shot": False,
        "skip_layout_detection": False,
        "use_stable_crop": False,
        "main_person": None,
        "smart_crop": True,
        "center_crop_fallback": True,
        "content": None,
        "detect_ball": False,
        "follow_ball_player": False,
        "smooth_transitions": True,
        "fallback_wide_shot": True,
        "split_never": False,
        "split_on_dialog": True,
        "group_aware": False,
        "cinematic": False,
        "detect_letterbox": False,
        "preserve_full_frame": False,
        "group_shot": False,
    },
    "screenReacts": {
        "priority": "person",
        "detect_webcam": True,
        "detect_people": True,
        "multi_person_layout": "split_or_wide",
        "fallback_on_screen": True,
        "person_threshold": 0.3,
        "split_or_wide": False,
        "closeup": False,
        "medium": True,
        "small": True,
        "check_screen_share": True,
        "fallback_on_no_webcam": True,
        "webcam_max_ratio": 0.4,
        "side_by_side_threshold": 0.5,
        "corner_webcam_threshold": 0.2,
        "fit_check": False,
        "prefer_group_shot": False,
        "skip_layout_detection": False,
        "use_stable_crop": True,
        "main_person": None,
        "smart_crop": True,
        "center_crop_fallback": False,
        "content": "screen",
        "detect_ball": False,
        "follow_ball_player": False,
        "smooth_transitions": True,
        "fallback_wide_shot": False,
        "split_never": True,
        "split_on_dialog": False,
        "group_aware": False,
        "cinematic": False,
        "detect_letterbox": False,
        "preserve_full_frame": False,
        "group_shot": False,
    },
    "vlog": {
        "priority": "person",
        "detect_webcam": False,
        "detect_people": True,
        "multi_person_layout": "split_or_wide",
        "fallback_on_screen": False,
        "person_threshold": 0.3,
        "split_or_wide": False,
        "closeup": True,
        "medium": True,
        "small": True,
        "check_screen_share": False,
        "fallback_on_no_webcam": False,
        "webcam_max_ratio": 0.4,
        "side_by_side_threshold": 0.5,
        "corner_webcam_threshold": 0.25,
        "fit_check": True,
        "prefer_group_shot": True,
        "skip_layout_detection": False,
        "use_stable_crop": False,
        "main_person": "center",
        "smart_crop": True,
        "center_crop_fallback": True,
        "content": None,
        "detect_ball": False,
        "follow_ball_player": False,
        "smooth_transitions": True,
        "fallback_wide_shot": True,
        "split_never": True,
        "split_on_dialog": False,
        "group_aware": True,
        "cinematic": False,
        "detect_letterbox": False,
        "preserve_full_frame": False,
        "group_shot": True,
    },
    "faceless": {
        "priority": "content",
        "detect_webcam": False,
        "detect_people": False,
        "multi_person_layout": "wide",
        "fallback_on_screen": False,
        "person_threshold": 0.5,
        "split_or_wide": False,
        "closeup": False,
        "medium": False,
        "small": False,
        "check_screen_share": False,
        "fallback_on_no_webcam": False,
        "webcam_max_ratio": 0.3,
        "side_by_side_threshold": 0.5,
        "corner_webcam_threshold": 0.25,
        "fit_check": False,
        "prefer_group_shot": False,
        "skip_layout_detection": True,
        "use_stable_crop": False,
        "main_person": None,
        "smart_crop": False,
        "center_crop_fallback": True,
        "content": None,
        "detect_ball": False,
        "follow_ball_player": False,
        "smooth_transitions": True,
        "fallback_wide_shot": True,
        "split_never": True,
        "split_on_dialog": False,
        "group_aware": False,
        "cinematic": True,
        "detect_letterbox": False,
        "preserve_full_frame": False,
        "group_shot": False,
    },
    "movies": {
        "priority": "content",
        "detect_webcam": False,
        "detect_people": False,
        "multi_person_layout": "wide",
        "fallback_on_screen": False,
        "person_threshold": 0.5,
        "split_or_wide": False,
        "closeup": False,
        "medium": False,
        "small": False,
        "check_screen_share": False,
        "fallback_on_no_webcam": False,
        "webcam_max_ratio": 0.3,
        "side_by_side_threshold": 0.5,
        "corner_webcam_threshold": 0.25,
        "fit_check": False,
        "prefer_group_shot": False,
        "skip_layout_detection": True,
        "use_stable_crop": False,
        "main_person": None,
        "smart_crop": False,
        "center_crop_fallback": True,
        "content": None,
        "detect_ball": False,
        "follow_ball_player": False,
        "smooth_transitions": False,
        "fallback_wide_shot": True,
        "split_never": True,
        "split_on_dialog": False,
        "group_aware": False,
        "cinematic": False,
        "detect_letterbox": True,
        "preserve_full_frame": True,
        "group_shot": False,
    },
    "sports": {
        "priority": "ball",
        "detect_webcam": False,
        "detect_people": True,
        "multi_person_layout": "wide",
        "fallback_on_screen": False,
        "person_threshold": 0.3,
        "split_or_wide": False,
        "closeup": False,
        "medium": True,
        "small": True,
        "check_screen_share": False,
        "fallback_on_no_webcam": False,
        "webcam_max_ratio": 0.3,
        "side_by_side_threshold": 0.5,
        "corner_webcam_threshold": 0.25,
        "fit_check": False,
        "prefer_group_shot": False,
        "skip_layout_detection": False,
        "use_stable_crop": False,
        "main_person": None,
        "smart_crop": True,
        "center_crop_fallback": True,
        "content": None,
        "detect_ball": True,
        "follow_ball_player": True,
        "smooth_transitions": True,
        "fallback_wide_shot": True,
        "split_never": True,
        "split_on_dialog": False,
        "group_aware": False,
        "cinematic": False,
        "detect_letterbox": False,
        "preserve_full_frame": False,
        "group_shot": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────

def get_video_codec_params(video_path: str) -> dict:
    """
    Returns FFmpeg video codec parameters based on actual GPU availability.
    Auto-detects if NVIDIA GPU with NVENC is available.

    Recovered from .pyd: torch.cuda.is_available check, ffmpeg -hide_banner
    -encoders probe for h264_nvenc, fallback to libx264 -preset fast -crf.
    """
    # Optional fast-path via torch
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        gpu_available = False

    # Definitive check: probe ffmpeg encoder list
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "h264_nvenc" in result.stdout:
            print(" [FFmpeg] GPU detected  Using h264_nvenc (NVENC)")
            return {"-c:v": "h264_nvenc", "-preset": "p4", "-b:v": "5M"}
    except Exception:
        pass

    print(" [FFmpeg] No GPU/NVENC  Using libx264 (CPU)")
    return {"-c:v": "libx264", "-preset": "fast", "-crf": "23"}


# ─────────────────────────────────────────────────────────────────────────────
# GeminiComposer
# ─────────────────────────────────────────────────────────────────────────────

class GeminiComposer:
    """
    Composes vertical videos based on Gemini's layout instructions.
    Uses YOLO for precise webcam/person detection when needed.
    Uses MediaPipe for lip sync detection (active speaker detection).

    Recovered from .pyd: class docstring, all method QNames, all variable names.
    """

    # ── init ────────────────────────────────────────────────────────────────

    def __init__(self, output_dir: str = ".", video_type: str | None = None) -> None:
        """
        Recovered from .pyd: model loading order, print messages, model paths.
        """
        self.output_dir = output_dir
        self.video_type = video_type

        os.makedirs(output_dir, exist_ok=True)

        # ── YOLO-Pose (skeleton / nose / shoulders) ──────────────────────
        self.yolo_pose = None
        self.YOLO_AVAILABLE = YOLO_AVAILABLE
        if YOLO_AVAILABLE:
            try:
                self.yolo_pose = YOLO("yolo11n-pose.pt")
                print(" YOLO-Pose loaded for skeleton detection")
            except Exception as e:
                print(f" YOLO-Pose not available: {e}")

            # ── YOLO plain (person bounding box) ─────────────────────────
            self.yolo = None
            try:
                self.yolo = YOLO("yolo11n.pt")
                print(" YOLO loaded for person detection")
            except Exception as e:
                print(f" YOLO not available: {e}")
        else:
            self.yolo = None

        # ── Webcam detector (custom trained model) ────────────────────────
        self.yolo_webcam = None
        script_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        webcam_model_path = os.path.join(script_dir, "models", "webcam_detector.pt")
        self.webcam_model_path = webcam_model_path

        if YOLO_AVAILABLE:
            if os.path.exists(webcam_model_path):
                try:
                    self.yolo_webcam = YOLO(webcam_model_path)
                    print(" Webcam detector model loaded")
                except Exception as e:
                    print(f" Webcam detector not loaded: {e}")
            else:
                print(" Webcam detector model not found")

        # ── MediaPipe Face Detector (BlazeFace) ───────────────────────────
        self.mp_face_detector = None
        model_path = os.path.join(script_dir, "blaze_face_short_range.tflite")
        self.model_path = model_path

        if MEDIAPIPE_AVAILABLE:
            try:
                import mediapipe.tasks.python as mp_tasks
                import mediapipe.tasks.python.vision as mp_vision

                options = mp_vision.FaceDetectorOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                if os.path.exists(model_path):
                    self.mp_face_detector = mp_vision.FaceDetector.create_from_options(options)
                    print(" MediaPipe Face Detection loaded")
                else:
                    print(f" MediaPipe Face Detection model not found: {model_path}")
            except Exception as e:
                print(f" MediaPipe Face Detection not loaded: {e}")

        # ── Lip sync + Text detector ──────────────────────────────────────
        self.lip_sync = None
        self.text_detector = None
        self.LIP_SYNC_AVAILABLE = LIP_SYNC_AVAILABLE
        if LIP_SYNC_AVAILABLE:
            try:
                self.lip_sync = LipSyncDetector()
                self.text_detector = TextDetector()
                print(" MediaPipe loaded for lip sync detection")
            except Exception as e:
                print(f" Lip sync detector not loaded: {e}")

        # ── Sports detector ───────────────────────────────────────────────
        self.sports_detector = None
        self.SPORTS_DETECTOR_AVAILABLE = SPORTS_DETECTOR_AVAILABLE
        if SPORTS_DETECTOR_AVAILABLE:
            try:
                self.sports_detector = SportsDetector()
                print(" Sports detector loaded for ball tracking")
            except Exception as e:
                print(f" Sports detector not loaded: {e}")

        print(f" GeminiComposer ready (output: {output_dir})")

    # ── Config ──────────────────────────────────────────────────────────────

    def _get_video_type_config(self) -> dict:
        """
        Get configuration for current video_type.
        Returns default config if video_type not set or unknown.
        """
        default_config = VIDEO_TYPE_CONFIG.get("solo", {})
        if not self.video_type:
            return default_config
        return VIDEO_TYPE_CONFIG.get(self.video_type, default_config)

    # ─────────────────────────────────────────────────────────────────────────
    # Content / letterbox analysis
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_letterbox(
        self, frame: np.ndarray, frame_h: int
    ) -> tuple | None:
        """
        Detect letterbox (black bars) in movie content.
        Returns (content_y_start, content_height) or None if no letterbox.
        Common aspect ratios with letterbox:
        - 2.35:1 (Cinemascope) - significant black bars
        - 2.39:1 (Anamorphic) - significant black bars
        - 1.85:1 - slight bars on 16:9 source
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_threshold = 16

        row_mean = np.mean(gray, axis=1)

        # Find top content row
        top_content = 0
        for i, v in enumerate(row_mean):
            if v > black_threshold:
                top_content = i
                break

        # Find bottom content row
        bottom_content = frame_h - 1
        for i in range(frame_h - 1, -1, -1):
            if row_mean[i] > black_threshold:
                bottom_content = i
                break

        top_bar = top_content
        bottom_bar = frame_h - 1 - bottom_content
        content_height = bottom_content - top_content + 1

        min_bar = int(frame_h * 0.05)
        if top_bar < min_bar and bottom_bar < min_bar:
            return None

        print(
            f" Letterbox detected: top={top_bar / frame_h:.1%}"
            f", bottom={bottom_bar / frame_h:.1%}"
        )
        return top_content, content_height

    def _classify_content_type(self, video_path: str) -> str:
        """
        Classify frame content type: 'screen_share' or 'natural_scene'
        Uses edge analysis to detect UI elements vs natural scenes:
        - Screen share: lots of edges (windows, buttons, text, graphs)
        - Natural scene: fewer edges (walls, sky, furniture, nature)
        Returns: 'screen_share' or 'natural_scene'
        NOTE: We no longer override layouts globally here.
        Each segment is validated IN ITS OWN METHOD based on actual person detection
        for that specific segment's frames.
        This function now just passes through segments unchanged.
        The real validation happens in _layout_single_speaker and _layout_screen_share
        which detect person in their own segment's video.
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.333))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "natural_scene"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=50, maxLineGap=10)
        num_lines = len(lines) if lines is not None else 0
        line_density = num_lines / (h * w / 10000)

        v_lines = h_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1))
                if angle < 0.1 or angle > np.pi - 0.1:
                    h_lines += 1
                elif abs(angle - np.pi / 2) < 0.1:
                    v_lines += 1
        hv_ratio = (h_lines + v_lines) / max(num_lines, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_std = float(np.std(hsv[:, :, 0]))
        s_std = float(np.std(hsv[:, :, 1]))
        color_variance = (h_std + s_std) / 2

        v_mean = float(np.mean(hsv[:, :, 2]))

        is_screen_share = (edge_density > 0.08 and line_density > 2.0 and hv_ratio > 0.5)
        is_text_overlay = edge_density > 0.12
        is_definitely_natural = color_variance > 30 and edge_density < 0.06

        text_overlay = is_text_overlay
        screen_share = is_screen_share and not is_definitely_natural
        natural_scene = not screen_share

        print(
            f" Content classification: edges={edge_density:.0%}"
            f", lines={num_lines}, hv={hv_ratio:.0%}"
        )

        if screen_share:
            return "screen_share"
        return "natural_scene"

    def _validate_layouts(self, segments: list) -> list:
        """
        NOTE: We no longer override layouts globally here.
        Each segment is validated IN ITS OWN METHOD based on actual person detection
        for that specific segment's frames.
        This function now just passes through segments unchanged.
        The real validation happens in _layout_single_speaker and _layout_screen_share
        which detect person in their own segment's video.
        """
        return segments

    # ─────────────────────────────────────────────────────────────────────────
    # Scene detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_scenes(self, video_path: str) -> list:
        """
        Detect scene changes using PySceneDetect (algorithmic, not AI).
        Returns list of (start_time, end_time) tuples.
        """
        if not SCENEDETECT_AVAILABLE:
            print(" PySceneDetect not available, using single segment")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = total / fps
            return [(0.0, duration)]

        try:
            from scenedetect import open_video, SceneManager
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(min_scene_len=15))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            scenes = []
            for scene in scene_list:
                start_t = scene[0].get_seconds()
                end_t = scene[1].get_seconds()
                scenes.append((start_t, end_t))

            print(f" PySceneDetect found {len(scenes)} scene(s)")
            return scenes if scenes else [(0.0, self._get_duration(video_path))]
        except Exception as e:
            print(f" PySceneDetect error: {e}")
            return [(0.0, self._get_duration(video_path))]

    def _get_duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total / fps

    def classify_scene(self, video_path: str, start: float, fps: float) -> str:
        """
        Classify a scene segment using YOLO-Pose with MULTI-SAMPLE.
        Checks START, MIDDLE, and END of scene to avoid empty frames.
        Returns layout type: 'single_person', 'talking_heads', 'wide_shot', etc.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(int(start * fps + total_frames), total_frames)
        start_frame = int(start * fps)
        duration_frames = end_frame - start_frame

        sample_offsets = [0.1, 0.5, 0.9]
        all_counts: list = []
        all_positions: list = []

        for pct in sample_offsets:
            frame_pos = start_frame + int(pct * duration_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]
            people_count = 0
            people_positions: list = []

            if self.yolo_pose and YOLO_AVAILABLE:
                results = self.yolo_pose(frame, classes=[0], verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf)
                            if conf < 0.3:
                                continue
                            xyxy = box.xyxy[0].tolist()
                            cx = (xyxy[0] + xyxy[2]) / 2
                            people_positions.append(cx)
                            people_count += 1

            all_counts.append(people_count)
            all_positions.extend(people_positions)

        cap.release()

        avg_count = sum(all_counts) / max(len(all_counts), 1)

        if not all_counts or max(all_counts) == 0:
            return "wide_shot"

        if any(c != all_counts[0] for c in all_counts):
            print(f" People count varies ({min(all_counts)}-{max(all_counts)})")

        if avg_count < 0.5:
            print(" Some frames have no people ( wide_shot)")
            return "wide_shot"
        elif avg_count < 1.5:
            return "single_person"
        elif avg_count < 2.5:
            return "talking_heads"
        else:
            return "wide_shot"

    # ─────────────────────────────────────────────────────────────────────────
    # Person / people detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_person_region(
        self, video_path: str, verbose: bool = False
    ) -> tuple | None:
        """
        Detect person region in video using YOLO.
        Samples multiple frames for stability and returns median position.
        Returns (x, y, w, h) of the person bounding box, or None if not found.
        """
        if not self.yolo or not YOLO_AVAILABLE:
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        all_detections: list = []

        for pct in sample_points:
            frame_num = int(pct * total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.yolo(frame, classes=[0], verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf)
                    if conf < 0.4:
                        continue
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    all_detections.append((x1, y1, x2 - x1, y2 - y1, conf))

        cap.release()

        if not all_detections:
            return None

        all_detections.sort(key=lambda d: d[4], reverse=True)
        top_n = 5
        top_dets = all_detections[:top_n]
        median_x = float(np.median([d[0] for d in top_dets]))
        median_y = float(np.median([d[1] for d in top_dets]))
        median_w = float(np.median([d[2] for d in top_dets]))
        median_h = float(np.median([d[3] for d in top_dets]))
        return (median_x, median_y, median_w, median_h)

    def _detect_people_at_time(
        self, input_path: str, time_sec: float
    ) -> list:
        """
        Detect people at a specific time. Returns list of (center_x, shoulder_width).
        Uses shoulder center for stable positioning - nose can be at edge when turned.
        shoulder_width is used to filter out background people (small = far away).
        """
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        people: list = []

        # Try YOLO-Pose first (shoulder-based)
        if self.yolo_pose and YOLO_AVAILABLE:
            try:
                results = self.yolo_pose(frame, classes=[0], verbose=False)
                for r in results:
                    if r.keypoints is None:
                        continue
                    kps = r.keypoints.cpu().numpy() if hasattr(r.keypoints, "cpu") else r.keypoints
                    for person_kps in kps:
                        pts = person_kps.data if hasattr(person_kps, "data") else person_kps
                        if len(pts) < 7:
                            continue
                        left_sh_x = float(pts[5][0]) if pts[5][2] > 0.3 else None
                        right_sh_x = float(pts[6][0]) if pts[6][2] > 0.3 else None

                        if left_sh_x is not None and right_sh_x is not None:
                            shoulder_x = (left_sh_x + right_sh_x) / 2
                            shoulder_width = abs(right_sh_x - left_sh_x)
                        elif left_sh_x is not None:
                            shoulder_x = left_sh_x
                            shoulder_width = frame_w * 0.15
                        elif right_sh_x is not None:
                            shoulder_x = right_sh_x
                            shoulder_width = frame_w * 0.15
                        else:
                            continue

                        min_shoulder_width = frame_w * 0.05
                        if shoulder_width < min_shoulder_width:
                            continue

                        people.append((shoulder_x, shoulder_width))
            except Exception as e:
                print(f" YOLO detection error @ {time_sec}s: {e}")

        return people

    def _detect_all_people(
        self, video_path: str, multi_sample: bool = True
    ) -> list:
        """
        Detect all people using YOLO with MULTI-FRAME SAMPLING.
        Samples multiple frames and aggregates results for reliability.
        Filters out false positives (hands, small objects).
        """
        if not self.yolo and not self.yolo_pose:
            print(" YOLO not initialized!")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f" Cannot open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sample_frames = [0.1, 0.3, 0.5, 0.7, 0.9] if multi_sample else [0.5]
        all_detections: list = []

        yolo_model = self.yolo_pose or self.yolo

        for pct in sample_frames:
            frame_pos = int(pct * total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                results = yolo_model(frame, classes=[0], verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        conf = float(box.conf)
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = xyxy
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        min_area = frame_w * frame_h * 0.01
                        min_height = frame_h * 0.1
                        if area < min_area or h < min_height:
                            continue
                        cx = (x1 + x2) / 2
                        all_detections.append((cx, w, conf))
            except Exception as e:
                print(f" YOLO error: {e}")

        cap.release()

        if not all_detections:
            print(" YOLO: no people after clustering")
            return []

        print(f" YOLO raw detections: {len(all_detections)} across {len(sample_frames)} frames")

        # Cluster nearby detections
        band_width = frame_w * 0.15
        clusters: list = []
        for det in all_detections:
            cx, sw, conf = det
            merged = False
            for i, cluster in enumerate(clusters):
                ccx, csw, ccount = cluster
                if abs(cx - ccx) < band_width:
                    new_cx = (ccx * ccount + cx) / (ccount + 1)
                    new_sw = (csw * ccount + sw) / (ccount + 1)
                    clusters[i] = (new_cx, new_sw, ccount + 1)
                    merged = True
                    break
            if not merged:
                clusters.append((cx, sw, 1))

        unique_positions = [(c[0], c[1]) for c in clusters]
        print(f" YOLO found {len(unique_positions)} unique people")
        return unique_positions

    def _detect_all_people_legacy(self, video_path: str) -> list:
        """Legacy single-frame detection (backup)"""
        return self._detect_all_people(video_path, multi_sample=False)

    def _detect_with_yolo_pose(self, video_path: str, num_samples: int = 5) -> list:
        """
        Detect people using YOLO-Pose - returns SKELETON points (nose, shoulders).
        Much better than regular YOLO: no false positives on mics/furniture.
        Returns list of (nose_x, confidence) for each person.
        """
        if not self.yolo_pose:
            return []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frames_detections: list = []
        sample_positions = [i / (num_samples + 1) for i in range(1, num_samples + 1)]

        for pct in sample_positions:
            frame_pos = int(pct * total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                results = self.yolo_pose(frame, classes=[0], verbose=False)
                frame_people: list = []
                for r in results:
                    if r.keypoints is None:
                        continue
                    kps = r.keypoints.cpu().numpy() if hasattr(r.keypoints, "cpu") else r.keypoints
                    for person_kps in kps:
                        pts = person_kps.data if hasattr(person_kps, "data") else person_kps
                        if len(pts) < 6:
                            continue
                        nose_x = float(pts[0][0])
                        nose_conf = float(pts[0][2]) if len(pts[0]) > 2 else 0.5
                        if nose_conf < 0.15:
                            continue
                        # Filter out background people (by shoulder width)
                        left_sh_x = float(pts[5][0]) if pts[5][2] > 0.3 else None
                        right_sh_x = float(pts[6][0]) if pts[6][2] > 0.3 else None
                        if left_sh_x and right_sh_x:
                            sw = abs(right_sh_x - left_sh_x)
                            if sw < frame_w * 0.05:
                                continue
                        frame_people.append((nose_x, nose_conf))
                frames_detections.append(frame_people)
            except Exception as e:
                print(f" YOLO-Pose failed: {e}")

        cap.release()

        if not frames_detections:
            return []

        max_ppl = max(len(fp) for fp in frames_detections)
        print(f" YOLO-Pose: found {max_ppl} simultaneous people (max across samples)")

        # Return the frame with the most people
        best_frame_people = max(frames_detections, key=lambda fp: len(fp))
        filtered_people = [p for p in best_frame_people if p[1] > 0.15]
        return filtered_people

    def _detect_person_pose(self, video_path: str) -> list:
        """
        Detect people using MediaPipe Pose - finds SKELETON (nose, shoulders).
        Much better than bounding box: centers on FACE, ignores raised arms.
        Returns list of (nose_x, shoulder_center_x) for each person.
        """
        if not MEDIAPIPE_AVAILABLE:
            return []

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.5))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return []

        try:
            mp_pose = mediapipe.solutions.pose
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
            ) as pose:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = pose.process(frame_rgb)

                if not mp_results.pose_landmarks:
                    return []

                landmarks = mp_results.pose_landmarks.landmark
                PL = mp_pose.PoseLandmark

                nose = landmarks[PL.NOSE]
                left_sh = landmarks[PL.LEFT_SHOULDER]
                right_sh = landmarks[PL.RIGHT_SHOULDER]

                h, w = frame.shape[:2]
                nose_x = nose.x * w
                left_sh_x = left_sh.x * w
                right_sh_x = right_sh.x * w
                shoulder_x = (left_sh_x + right_sh_x) / 2
                shoulder_width = abs(right_sh_x - left_sh_x)

                print(f" MediaPipe Pose: found person, shoulder_w={shoulder_width:.0f}px")
                return [(nose_x, shoulder_x)]
        except Exception as e:
            print(f" MediaPipe Pose failed: {e}")
            return []

    def _check_people_fit(
        self, positions: list, frame_w: int
    ) -> tuple:
        """
        Fit Check: Can all people fit in a single 9:16 vertical crop?
        Args:
            positions: List of person center_x positions
            frame_w: Frame width in pixels
        Returns:
            (fits: bool, center_x: float, spread: float)
            - fits: True if all people can fit in one 9:16 crop
            - center_x: Optimal crop center (midpoint between leftmost and rightmost)
            - spread: Distance between leftmost and rightmost as fraction of frame width
        """
        if not positions:
            return False, frame_w / 2, 0.0

        target_w = int(frame_w * 9 / 16) if frame_w > frame_w * 9 / 16 else frame_w
        MAX_FIT_SPREAD = target_w * 0.8

        leftmost = min(positions)
        rightmost = max(positions)
        spread = rightmost - leftmost
        center_x = (leftmost + rightmost) / 2
        spread_ratio = spread / frame_w

        fits = spread <= MAX_FIT_SPREAD
        return fits, center_x, spread_ratio

    def _get_person_size_at_time(self, video_path: str, time_sec: float) -> float | None:
        """
        Get the relative size of the largest person in frame at given time.
        Returns: float between 0 and 1 (person area / frame area), or None if no person.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
        ret, frame = cap.read()
        cap.release()

        if not ret or not self.yolo:
            return None

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h
        max_area = 0.0

        results = self.yolo(frame, classes=[0], verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                person_area = (x2 - x1) * (y2 - y1)
                max_area = max(max_area, person_area)

        if max_area == 0:
            return None
        return max_area / frame_area

    # ─────────────────────────────────────────────────────────────────────────
    # Screen content / MSER detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_screen_content(
        self, frame: np.ndarray
    ) -> tuple:
        """
        Semantic Content Detection (MSER).
        Detects if there is a 'Screen' or 'Slide' with text in the frame.
        Used to distinguish Side-by-Side Expert from Wide Solo Shot.
        Returns (has_content: bool, area_pct: float, density: float)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)

            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            hulls: list = []
            for region in regions:
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                hulls.append(hull)
                cv2.fillPoly(mask, [hull], 255)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_dilated = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            max_area = 0
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                max_area = max(max_area, area)

            frame_area = h * w
            area_pct = max_area / frame_area

            roi_mask = mask_dilated
            density = float(cv2.countNonZero(roi_mask)) / (frame_area)

            has_content = area_pct > 0.05 and density > 0.01

            if has_content:
                print(f" Found Screen Content: {area_pct:.0%} area, {density:.2f} density)")
            return has_content, area_pct, density

        except Exception as e:
            print(f" Screen detection failed: {e}")
            return False, 0.0, 0.0

    def _has_important_text(self, video_path: str) -> bool:
        """Check if video has important text that shouldn't be cropped"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.5))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        has_content, area_pct, density = self._detect_screen_content(frame)
        if has_content:
            print(" Important text detected - using letterbox")
        return has_content

    # ─────────────────────────────────────────────────────────────────────────
    # Webcam detection
    # ─────────────────────────────────────────────────────────────────────────

    def _try_webcam_model_detection(
        self, frame: np.ndarray, frame_w: int, frame_h: int
    ) -> dict | None:
        """
        Try trained webcam model to detect webcam boundaries.
        Returns webcam info dict if found, None otherwise.
        """
        if not self.yolo_webcam:
            return None

        try:
            results = self.yolo_webcam(frame, imgsz=640, verbose=False)
            all_boxes: list = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf)
                    xyxy = box.xyxy[0].tolist()
                    all_boxes.append((conf, xyxy))

            all_boxes.sort(key=lambda b: b[0], reverse=True)

            valid_candidates: list = []
            for conf, xyxy in all_boxes:
                if conf < 0.333:
                    continue
                x1, y1, x2, y2 = xyxy
                webcam_w = x2 - x1
                webcam_h = y2 - y1

                cx_pct = (x1 + x2) / 2 / frame_w
                is_centered = 0.3 < cx_pct < 0.7
                is_large = webcam_w / frame_w > 0.6

                if is_centered and is_large:
                    has_person, _, _ = self._detect_screen_content(
                        frame[int(y1):int(y2), int(x1):int(x2)]
                    )
                    if not has_person:
                        print(f" Webcam model rejected: centered ({cx_pct:.2f}) without content")
                        continue

                valid_candidates.append((conf, xyxy))

            if not valid_candidates:
                return None

            best_conf, best_xyxy = valid_candidates[0]
            x1, y1, x2, y2 = best_xyxy
            webcam_w = x2 - x1
            webcam_h = y2 - y1

            # Classify corner
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            is_left = cx < frame_w * 0.4
            is_right = cx > frame_w * 0.6
            is_top = cy < frame_h * 0.4
            is_bottom = cy > frame_h * 0.6

            if is_left and is_top:
                corner = "top_left"
            elif is_right and is_top:
                corner = "top_right"
            elif is_left and is_bottom:
                corner = "bottom_left"
            elif is_right and is_bottom:
                corner = "bottom_right"
            else:
                corner = "center"

            # Check for person inside webcam area
            webcam_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            pose_results = self._detect_person_pose.__func__(
                self, ""
            ) if False else []

            print(
                f" Webcam Model: Found {best_conf:.2f} @ ({int(x1)},{int(y1)})"
                f", conf={best_conf:.2f}"
            )

            return {
                "has_webcam": True,
                "webcam_bbox": (int(x1), int(y1), int(webcam_w), int(webcam_h)),
                "webcam_corner": corner,
                "score": best_conf,
                "is_edge": corner != "center",
                "person_center": (cx, cy),
            }

        except Exception as e:
            print(f" Webcam model error: {e}")
            return None

    def _detect_webcam_bounds_pose(
        self, video_path: str, corner: str, sample_pos: float = 0.1
    ) -> dict | None:
        """
        Detect precise webcam boundaries using YOLO-Pose skeleton.
        Finds the person in the webcam corner and calculates exact bounds.
        sample_pos: position in video to sample (default 0.1 = 10%, same as webcam detection)
        """
        if not self.yolo_pose:
            return None

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_pos * total))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        try:
            results = self.yolo_pose(frame, classes=[0], verbose=False)
            for r in results:
                if r.keypoints is None:
                    continue
                kps = r.keypoints.cpu().numpy() if hasattr(r.keypoints, "cpu") else r.keypoints
                for person_kps in kps:
                    pts = person_kps.data if hasattr(person_kps, "data") else person_kps
                    if len(pts) < 7:
                        continue
                    nose_x = float(pts[0][0])
                    nose_y = float(pts[0][1])
                    left_sh = pts[5]
                    right_sh = pts[6]
                    left_sh_x = float(left_sh[0]) if float(left_sh[2]) > 0.3 else None
                    right_sh_x = float(right_sh[0]) if float(right_sh[2]) > 0.3 else None

                    if left_sh_x is None or right_sh_x is None:
                        continue

                    shoulder_w = abs(right_sh_x - left_sh_x)
                    center_x = (left_sh_x + right_sh_x) / 2

                    # Check if this person is in the expected corner
                    is_left = center_x < frame_w * 0.4
                    is_right = center_x > frame_w * 0.6
                    is_top_region = nose_y < frame_h * 0.5

                    if "left" in corner and not is_left:
                        continue
                    if "right" in corner and not is_right:
                        continue

                    # Build webcam bounds from shoulder + face
                    margin_x = shoulder_w * 0.3
                    margin_top = shoulder_w * 0.8
                    head_top = nose_y - margin_top

                    webcam_x = int(max(0, center_x - shoulder_w / 2 - margin_x))
                    webcam_y = int(max(0, head_top))
                    webcam_w = int(min(shoulder_w + 2 * margin_x, frame_w - webcam_x))
                    webcam_h = int(min(frame_h - webcam_y, webcam_w * 4 / 3))

                    print(f" Pose-based webcam: person @ ({int(center_x)},{int(nose_y)})"
                          f", shoulders={int(shoulder_w)}px")
                    print(f" Webcam area: {webcam_x},{webcam_y},{webcam_w},{webcam_h}")

                    return {
                        "has_webcam": True,
                        "webcam_bbox": (webcam_x, webcam_y, webcam_w, webcam_h),
                        "webcam_corner": corner,
                        "shoulder_w": shoulder_w,
                        "center_x": center_x,
                    }
        except Exception as e:
            print(f" Detect webcam bounds pose failed: {e}")

        return None

    def _detect_webcam_bounds_from_face(
        self, face_bbox: tuple, frame_w: int, frame_h: int
    ) -> dict:
        """
        Get webcam bounds based on face position.
        Creates a region around the face that captures the person well.
        """
        face_x, face_y, face_w, face_h = face_bbox

        margin_x = face_w * 0.5
        margin_top = face_h * 0.3
        margin_bottom = face_h * 2.0

        webcam_x = int(max(0, face_x - margin_x))
        webcam_y = int(max(0, face_y - margin_top))
        webcam_w = int(min(face_w + 2 * margin_x, frame_w - webcam_x))
        webcam_h = int(min(face_h + margin_top + margin_bottom, frame_h - webcam_y))

        cx = face_x + face_w / 2
        cy = face_y + face_h / 2
        is_left = cx < frame_w * 0.4
        is_right = cx > frame_w * 0.6
        is_bottom = cy > frame_h * 0.6

        if is_left and is_bottom:
            corner = "bottom_left"
        elif is_right and is_bottom:
            corner = "bottom_right"
        elif is_left:
            corner = "top_left"
        elif is_right:
            corner = "top_right"
        else:
            corner = "bottom_left"

        return {
            "has_webcam": True,
            "webcam_bbox": (webcam_x, webcam_y, webcam_w, webcam_h),
            "webcam_corner": corner,
            "face_w": face_w,
            "face_h": face_h,
        }

    def _detect_webcam_at_time(
        self,
        input_path: str,
        sample_pos: float = 0.1,
    ) -> dict:
        """
        WEBCAM DETECTION - Uses trained model + YOLO Pose verification.
        Algorithm:
        1. Use trained webcam detector model to find webcam boundaries
        2. Verify with YOLO Pose that there's a person inside
        3. Fallback to pose-based detection if model fails
        Returns precise webcam rectangle.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"has_webcam": False}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_pos * total_frames))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"has_webcam": False}

        # Try trained model first
        model_result = self._try_webcam_model_detection(frame, frame_w, frame_h)
        if model_result:
            x, y, w, h = model_result["webcam_bbox"]
            person_center_x = x + w / 2
            person_center_y = y + h / 2
            shoulder_pct = w / frame_w

            is_large = shoulder_pct > 0.6
            is_medium = 0.3 < shoulder_pct <= 0.6
            is_small = shoulder_pct <= 0.3

            cx_pct = (x + w / 2) / frame_w
            is_centered = 0.3 < cx_pct < 0.7
            is_side_aligned = not is_centered

            # Classify: full-screen person or side-by-side or corner
            if is_large and is_centered:
                has_content, area_pct, density = self._detect_screen_content(frame)
                if has_content:
                    print(f" Side-by-Side Expert detected ({model_result['webcam_corner']},"
                          f" shoulders={int(w)}px)")
                    model_result["is_full_screen"] = False
                    model_result["content_type"] = "side_by_side_expert"
                    return model_result
                else:
                    print(f" Full-Screen Person detected (shoulders={int(w)}px, {shoulder_pct:.0%} width)")
                    model_result["is_full_screen"] = True
                    model_result["content_type"] = "full_screen"
                    return model_result

            print(f" Person verified inside webcam area")
            model_result["is_full_screen"] = False
            model_result["content_type"] = "webcam_corner"
            return model_result

        # Fallback: pose-based detection
        pose_result = self._detect_webcam_bounds_pose(input_path, "bottom_right", sample_pos)
        if pose_result:
            return pose_result

        return {"has_webcam": False}

    def _detect_webcam_overlay(self, input_path: str) -> dict:
        """
        Detect if video has webcam overlay (small face in corner).
        Samples MULTIPLE frames to catch webcam in any scene.
        Returns: {'has_webcam': bool, 'webcam_bbox': (x,y,w,h), 'webcam_corner': str}
        """
        sample_percentages = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
        webcam_results: list = []

        for pct in sample_percentages:
            result = self._detect_webcam_at_time(input_path, sample_pos=pct)
            if result.get("has_webcam"):
                webcam_results.append(result)

        if not webcam_results:
            return {"has_webcam": False}

        # Merge overlapping results
        merged = webcam_results[0].copy()
        print(f" Merged {len(webcam_results)} overlapping webcam detections")

        webcam_bbox = merged.get("webcam_bbox", (0, 0, 0, 0))
        corner = merged.get("webcam_corner", "unknown")
        print(f" Webcam detected @ {sample_percentages[0]:.0%}: {corner}, area={webcam_bbox}")

        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Content ROI / smart crop
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_content_roi(
        self,
        frame: np.ndarray,
        exclude_region: tuple | None = None,
        sample_frames: int = 3,
    ) -> tuple:
        """
        Smart content crop - detect region of interest (ROI) in content area.
        Uses SALIENCY detection + ACTIVITY analysis to find important content.
        Returns the best crop region for content (excluding webcam area).
        """
        h, w = frame.shape[:2]

        # Saliency detection
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(frame)
            sal = (saliency_map * 255).astype(np.uint8) if success else None
        except Exception as e:
            print(f" Saliency detection failed: {e}")
            sal = None

        # Activity map (edge-based)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 50, 150)

        # Activity map (motion approximation via pixel std)
        ones = np.ones_like(gray, dtype=np.float32)
        activity_map = edge_map.astype(np.float32) / 255.0

        # Combine maps
        if sal is not None:
            combined_map = (sal.astype(np.float32) / 255.0) * 0.5 + activity_map * 0.5
        else:
            combined_map = activity_map

        # Exclude webcam region
        if exclude_region:
            ex, ey, ew, eh = exclude_region
            combined_map[ey:ey + eh, ex:ex + ew] = 0

        # Find weighted center
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        total_weight = combined_map.sum()
        if total_weight < 1e-6:
            roi_cx, roi_cy = w // 2, h // 2
        else:
            roi_cx = int((combined_map * x_coords).sum() / total_weight)
            roi_cy = int((combined_map * y_coords).sum() / total_weight)

        print(f" ROI center (saliency+activity+edges): ({roi_cx}, {roi_cy})")
        return roi_cx, roi_cy

    def _find_best_crop_x(
        self,
        frame: np.ndarray,
        x_start: int,
        x_end: int,
    ) -> float:
        """
        Smart Crop: Find the "most interesting" horizontal crop within a range.
        Uses Sobel gradient magnitude to detect text/edges.
        """
        h, w = frame.shape[:2]
        crop_w = x_end - x_start
        roi_gray = frame[0:h, x_start:x_end]
        if len(roi_gray.shape) == 3:
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

        roi_gray = roi_gray.astype(np.float32)
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        target_crop_w = int(h * 9 / 16)
        crop_w_small = min(target_crop_w, crop_w)
        roi_w_small = crop_w

        best_score = -1.0
        best_x_small = 0

        step = max(1, crop_w_small // 20)
        for x in range(0, roi_w_small - crop_w_small + 1, step):
            window = mag[:, x:x + crop_w_small]
            # Bias toward center
            bias = 1.0 - abs(x + crop_w_small / 2 - roi_w_small / 2) / roi_w_small
            biased_score = float(window.mean()) * (1 + 0.2 * bias)
            if biased_score > best_score:
                best_score = biased_score
                best_x_small = x

        best_x = x_start + best_x_small + crop_w_small / 2
        return float(best_x)

    # ─────────────────────────────────────────────────────────────────────────
    # Core crop / apply helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_crop(
        self,
        input_path: str,
        output_path: str,
        crop_x: int,
        crop_y: int = 0,
        target_w: int = CANVAS_W,
        target_h: int = CANVAS_H,
        ppp: bool = False,
    ) -> str:
        """Apply crop and scale to target size"""
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            x_off = max(0, min(crop_x - target_w // 2, src_w - target_w))
            crop_h = min(src_h - crop_y, target_h if target_h <= src_h else src_h)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            temp_silent = output_path.replace(".mp4", ".silent.mp4")
            out = cv2.VideoWriter(temp_silent, fourcc, fps, (target_w, target_h))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cropped = frame[crop_y:crop_y + crop_h, x_off:x_off + target_w]
                if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
                    cropped = cv2.resize(cropped, (target_w, target_h))
                out.write(cropped)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f" Crop: {frame_count} frames...")

            cap.release()
            out.release()

            # Merge audio
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_silent,
                "-i", input_path,
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            os.remove(temp_silent)
            print(f" Cropped: {output_path}")
            return output_path

        except Exception as e:
            print(f" Crop error: {e}")
            traceback.print_exc()
            return input_path

    def _crop_centered(
        self,
        input_path: str,
        output_path: str,
        crop_x: float,
        target_w: int = CANVAS_W,
        target_h: int = CANVAS_H,
    ) -> str:
        """
        Crop 9:16 centered on a specific person.
        Ensures person is IN CENTER of crop, not at edge!
        """
        print(f" Crop centered @ x={crop_x:.0f}")
        print(f" crop_x={int(crop_x)}")
        return self._apply_crop(
            input_path, output_path, int(crop_x), 0, target_w, target_h
        )

    def _crop_on_nose(
        self,
        input_path: str,
        output_path: str,
        nose_x: float,
        target_w: int = CANVAS_W,
        target_h: int = CANVAS_H,
    ) -> str:
        """Legacy - redirects to _crop_centered."""
        return self._crop_centered(input_path, output_path, nose_x, target_w, target_h)

    def _crop_on_person(
        self,
        input_path: str,
        output_path: str,
        person_bbox: tuple,
        target_w: int = CANVAS_W,
        target_h: int = CANVAS_H,
    ) -> str:
        """Crop 9:16 centered on a specific person."""
        x, y, w, h = person_bbox
        cx = x + w / 2
        print(f" Cropping on person @ x={cx:.0f}")
        return self._crop_centered(input_path, output_path, cx, target_w, target_h)

    # ─────────────────────────────────────────────────────────────────────────
    # ffmpeg-based layout helpers (filter_complex)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_ffmpeg_filter(
        self,
        input_path: str,
        output_path: str,
        vf: str,
        extra_args: list | None = None,
    ) -> str:
        """Run ffmpeg with a video filter string."""
        codec = get_video_codec_params(input_path)
        vcodec = codec.get("-c:v", "libx264")
        preset = codec.get("-preset", "fast")
        crf = codec.get("-crf", "23")

        cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", vf]
        if vcodec == "libx264":
            cmd += ["-c:v", vcodec, "-preset", preset, "-crf", str(crf)]
        else:
            cmd += ["-c:v", vcodec, "-preset", preset, "-b:v", codec.get("-b:v", "5M")]
        cmd += ["-c:a", "aac", "-b:a", "192k", output_path]
        if extra_args:
            cmd = cmd[:-1] + extra_args + [cmd[-1]]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def _run_ffmpeg_complex(
        self,
        input_path: str,
        output_path: str,
        filter_complex: str,
        map_v: str = "[v]",
        map_a: str = "0:a?",
        extra_args: list | None = None,
    ) -> str:
        """Run ffmpeg with filter_complex."""
        codec = get_video_codec_params(input_path)
        vcodec = codec.get("-c:v", "libx264")

        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-filter_complex", filter_complex,
               "-map", map_v, "-map", map_a,
               "-c:v", vcodec, "-c:a", "aac",
               output_path]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Layout methods
    # ─────────────────────────────────────────────────────────────────────────

    def _layout_wide_shot(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        wide_shot: full frame with letterbox
        Wide shot - scale to FILL WIDTH (1080), add black bars top/bottom only.
        For 16:9 source (1280x720) scales to 1080x607, centered with bars top/bottom.
        This is the correct "full screen" behavior - fills width, preserves aspect.
        """
        vf = (
            f"scale={CANVAS_W}:-1:force_original_aspect_ratio=decrease,"
            f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )
        try:
            print(" wide_shot: full frame with letterbox")
            return self._run_ffmpeg_filter(input_path, output_path, vf)
        except Exception as e:
            print(f" Letterbox error: {e}")
            return input_path

    def _layout_letterbox_full_frame(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        MOVIES MODE: Full original frame with letterbox.
        Shows the ENTIRE source frame (no cropping!) centered in 9:16 with black bars.
        Preserves director's original composition - perfect for movies/cinematic content.
        For 16:9 source  fills width, black bars top/bottom
        For 2.35:1 source  fills width, larger black bars top/bottom
        For 4:3 source  fills width, smaller black bars top/bottom
        """
        vf = (
            f"scale={CANVAS_W}:-1:force_original_aspect_ratio=decrease,"
            f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )
        try:
            print(" letterbox: full frame preserved (no cropping)")
            return self._run_ffmpeg_filter(input_path, output_path, vf)
        except Exception as e:
            print(f" Letterbox error: {e}")
            return input_path

    def _layout_full_frame(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Full frame layout - scale to FILL WIDTH, add bars top/bottom.
        Good for charts, text, presentations.
        """
        print(" full_frame: scaled to fit")
        vf = (
            f"scale={CANVAS_W}:-1:force_original_aspect_ratio=decrease,"
            f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )
        try:
            return self._run_ffmpeg_filter(input_path, output_path, vf)
        except Exception as e:
            return input_path

    def _layout_center_crop(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """Simple center crop when no people detected."""
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return self._crop_centered(input_path, output_path, frame_w / 2)

    def _layout_screen_only(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Smart crop for screen content only (no webcam) - screenReacts mode.
        Uses saliency detection to find the most interesting part of the screen.
        Crops a 9:16 vertical slice centered on the point of interest.
        """
        print(" screen_only: smart crop on content")

        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sample_frames: list = []
        for pct in [0.2, 0.5, 0.8]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pct * total))
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        cap.release()

        if not sample_frames:
            print(" No frames, falling back to center crop")
            return self._layout_center_crop(input_path, output_path)

        # Saliency-based center detection
        roi_xs: list = []
        for frame in sample_frames:
            try:
                roi_cx, roi_cy = self._detect_content_roi(frame)
                roi_xs.append(roi_cx)
            except Exception as e:
                print(f" Saliency failed ({e}), using center")
                roi_xs.append(frame_w // 2)

        if not roi_xs:
            roi_xs = [frame_w // 2]

        target_x = int(np.median(roi_xs))
        target_w = min(int(frame_h * 9 / 16), frame_w)

        x_start = max(0, target_x - target_w // 2)
        x_start = min(x_start, frame_w - target_w)

        edges = [f"x={x_start}", f"w={target_w}"]
        print(f" Crop: {target_w}px @ x={x_start}")

        vf = (
            f"crop={target_w}:{frame_h}:{x_start}:0,"
            f"scale={CANVAS_W}:{CANVAS_H}:flags=lanczos,setsar=1"
        )
        try:
            result = self._run_ffmpeg_filter(input_path, output_path, vf)
            print(" screen_only: smart crop completed")
            return result
        except Exception as e:
            print(f" Screen content failed: {e}")
            return self._layout_center_crop(input_path, output_path)

    def _layout_screen_content(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Screen content layout - scale to FILL WIDTH in vertical frame.
        Used for screenshares, presentations, gameplay without webcam.
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        scale = CANVAS_W / frame_w
        new_h = int(frame_h * scale)

        print(f" Screen content  scale to fill width")
        vf = (
            f"scale={CANVAS_W}:-1,"
            f"pad={CANVAS_W}:{CANVAS_H}:0:(oh-ih)/2:black,setsar=1"
        )
        try:
            return self._run_ffmpeg_filter(input_path, output_path, vf)
        except Exception as e:
            print(f" Screen content failed: {e}")
            return input_path

    def _layout_split_by_centers(
        self,
        input_path: str,
        output_path: str,
        centers: list,
        **kwargs,
    ) -> str:
        """
        Split screen based on person CENTERS.
        Each half is 1080x960 (9:8 aspect ratio).
        Crops 9:8 from source centered on each person, preserving proportions.
        """
        if len(centers) < 2:
            cx = centers[0] if centers else 0
            return self._crop_centered(input_path, output_path, cx)

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        half_h = CANVAS_H // 2
        half_aspect = CANVAS_W / half_h

        sorted_centers = sorted(centers)
        left_center = sorted_centers[0]
        right_center = sorted_centers[-1]

        crop_w = int(frame_h * half_aspect)
        crop_w = min(crop_w, frame_w)

        # Head position estimate
        head_y_estimate = int(frame_h * 0.15)
        target_head_position = int(half_h * 0.25)

        left_crop_x = int(max(0, min(left_center - crop_w // 2, frame_w - crop_w)))
        right_crop_x = int(max(0, min(right_center - crop_w // 2, frame_w - crop_w)))

        left_pct = (left_center - left_crop_x) / crop_w * 100
        right_pct = (right_center - right_crop_x) / crop_w * 100

        print(f" Split: L crop @ x={left_crop_x}, R crop @ x={right_crop_x}")
        print(f" Left person @ {left_pct:.0f}% in crop)")
        print(f" Right person @ {right_pct:.0f}%)")
        print(f" Left: {left_center:.0f}px | Right: {right_center:.0f}px (crop_w={crop_w})")

        filter_str = (
            f"[0:v]crop={crop_w}:{frame_h}:{left_crop_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[top];"
            f"[0:v]crop={crop_w}:{frame_h}:{right_crop_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2[v]"
        )

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(" Split screen created (aspect preserved)")
            return output_path
        except Exception as e:
            print(f" Split failed: {e}, using single crop")
            return self._crop_centered(input_path, output_path, (left_center + right_center) / 2)

    def _layout_split_by_nose(
        self,
        input_path: str,
        output_path: str,
        nose_positions: list,
        **kwargs,
    ) -> str:
        """
        Split screen based on detected nose positions.
        Each half is 9:8 aspect ratio.
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        sorted_noses = sorted(nose_positions)
        if len(sorted_noses) < 2:
            return self._crop_centered(input_path, output_path, sorted_noses[0] if sorted_noses else frame_w / 2)

        half_h = CANVAS_H // 2
        half_aspect = CANVAS_W / half_h
        crop_w = int(frame_h * half_aspect)
        crop_w = min(crop_w, frame_w)

        left_nose = sorted_noses[0]
        right_nose = sorted_noses[-1]

        left_crop_x = int(max(0, min(left_nose - crop_w // 2, frame_w - crop_w)))
        right_crop_x = int(max(0, min(right_nose - crop_w // 2, frame_w - crop_w)))

        print(f" Split: L crop @ x={left_crop_x}, R crop @ x={right_crop_x}")
        print(f" u, y=0, size={crop_w} (9:8)")

        filter_str = (
            f"[0:v]crop={crop_w}:{frame_h}:{left_crop_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[top];"
            f"[0:v]crop={crop_w}:{frame_h}:{right_crop_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2[v]"
        )

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "0:a?",
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(" Split screen created")
            return output_path
        except Exception as e:
            print(f" Split failed: {e}")
            return self._crop_centered(input_path, output_path, (left_nose + right_nose) / 2)

    def _layout_split_screen(
        self,
        input_path: str,
        output_path: str,
        left_person: tuple | None = None,
        right_person: tuple | None = None,
        **kwargs,
    ) -> str:
        """
        Split screen for 2+ people spread across frame.
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not left_person or not right_person:
            people = self._detect_all_people(input_path)
            if len(people) < 2:
                print(" People not on both sides - using wide shot")
                return self._layout_wide_shot(input_path, output_path)
            sorted_p = sorted(people, key=lambda p: p[0])
            left_cx = sorted_p[0][0]
            right_cx = sorted_p[-1][0]
        else:
            left_cx = left_person[0] + left_person[2] / 2
            right_cx = right_person[0] + right_person[2] / 2

        half_h = CANVAS_H // 2
        crop_w = int(frame_h * CANVAS_W / half_h)
        crop_w = min(crop_w, frame_w // 2 + 100)

        left_x = int(max(0, left_cx - crop_w // 2))
        right_x = int(max(0, min(right_cx - crop_w // 2, frame_w - crop_w)))

        print(f" Split: left @ {left_x}, right @ {right_x} (crop_w={crop_w})")

        filter_str = (
            f"[0:v]split=2[left][right];"
            f"[left]crop={crop_w}:{frame_h}:{left_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[left_out];"
            f"[right]crop={crop_w}:{frame_h}:{right_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[right_out];"
            f"[left_out][right_out]vstack=inputs=2"
        )

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(" Split screen created")
            return output_path
        except Exception as e:
            print(f" Split failed: {e}, using single crop")
            return self._crop_centered(input_path, output_path, (left_cx + right_cx) / 2)

    def _render_dynamic_switching(
        self,
        input_path: str,
        output_path: str,
        known_noses: list,
        **kwargs,
    ) -> str:
        """
        Generates a dynamic "Zoom-style" video that switches to the active speaker.
        Used for 3+ people or when people are too close for split screen.
        """
        if not self.lip_sync or not LIP_SYNC_AVAILABLE:
            print(" Lip sync not available, falling back to center crop")
            cx = sum(n[0] for n in known_noses) / len(known_noses) if known_noses else 0
            return self._crop_centered(input_path, output_path, cx)

        try:
            analysis = self.lip_sync.analyze_video_segment(input_path)
            speaking_segments = analysis.get("speaking_segments", [])

            if not speaking_segments:
                print(" No speech detected -> Defaulting to Wide Shot")
                return self._layout_wide_shot(input_path, output_path)

            # Build cuts
            cuts: list = []
            current_time = 0.0

            for seg in speaking_segments:
                speaker_id = seg.get("speaker_id", 0)
                start = seg.get("start", current_time)
                end = seg.get("end", start + 1.0)
                dur = end - start

                # Map speaker to person
                if speaker_id < len(known_noses):
                    nose_x = known_noses[speaker_id][0]
                else:
                    nose_x = known_noses[0][0]

                cuts.append({
                    "type": "crop",
                    "start": start,
                    "end": end,
                    "target_x": nose_x,
                })

            print(f" Generated {len(cuts)} switching cuts")

            # Extract and concatenate cuts
            temp_dir = tempfile.mkdtemp()
            cut_files: list = []

            for i, cut in enumerate(cuts):
                cut_in = os.path.join(temp_dir, f"cut_{i:02d}_in.mp4")
                cut_out = os.path.join(temp_dir, f"cut_{i:02d}_out.mp4")
                dur = cut["end"] - cut["start"]

                # Extract segment
                cmd_extract = [
                    "ffmpeg", "-y",
                    "-ss", f"{cut['start']:.3f}",
                    "-i", input_path,
                    "-t", f"{dur:.3f}",
                    "-c", "copy",
                    cut_in,
                ]
                subprocess.run(cmd_extract, capture_output=True, check=True)

                # Crop this segment
                self._crop_centered(cut_in, cut_out, cut["target_x"])
                if os.path.exists(cut_out):
                    cut_files.append(cut_out)

            if not cut_files:
                return self._layout_wide_shot(input_path, output_path)

            # Concat
            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, "w") as cf:
                for f in cut_files:
                    cf.write(f"file '{f}'\n")

            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                output_path,
            ]
            subprocess.run(cmd_concat, capture_output=True, check=True)
            shutil.rmtree(temp_dir, ignore_errors=True)

            print(" Dynamic switching video created")
            return output_path

        except Exception as e:
            print(f" Dynamic rendering failed: {e}")
            cx = sum(n[0] for n in known_noses) / len(known_noses) if known_noses else 0
            return self._crop_centered(input_path, output_path, cx)

    def _layout_talking_heads(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Smart layout for 2+ people (podcast/interview).
        Uses YOLO-Pose for DIRECT nose coordinates.
        Checks for simultaneous people availability.
        """
        pose_people = self._detect_with_yolo_pose(input_path)

        if not pose_people:
            print(" No people detected -> wide shot")
            return self._layout_wide_shot(input_path, output_path)

        if len(pose_people) == 1:
            print(f" Single person @ x={pose_people[0][0]:.0f}")
            return self._crop_centered(input_path, output_path, pose_people[0][0])

        # Multiple people - get nose positions
        sorted_noses = sorted([p[0] for p in pose_people])

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        spread = sorted_noses[-1] - sorted_noses[0]
        min_split_distance = frame_w * 0.25

        if spread < min_split_distance:
            print(f" People too close ({spread:.0f}px) -> Single Crop (Center)")
            center = sum(sorted_noses) / len(sorted_noses)
            return self._crop_centered(input_path, output_path, center)

        # Check if 2 specific people → split
        if len(pose_people) == 2:
            print(f" Split on noses @ {sorted_noses[0]:.0f} | {sorted_noses[1]:.0f}")
            return self._layout_split_by_nose(input_path, output_path, sorted_noses)

        # 3+ people → dynamic switching with LipSync
        print(f" Multi-person ({len(pose_people)}) -> Dynamic Switching with LipSync")
        return self._render_dynamic_switching(input_path, output_path, pose_people)

    def _layout_single_speaker(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Single speaker layout - crop centered on person's FACE (not body).
        Priority: YOLO-Pose (best)  MediaPipe Pose  YOLO (fallback)
        Uses skeleton detection - ignores raised arms, no false positives.
        """
        # 1. YOLO-Pose
        pose_people = self._detect_with_yolo_pose(input_path, num_samples=3)

        if pose_people:
            cap = cv2.VideoCapture(input_path)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()

            positions = [p[0] for p in pose_people]
            spread = max(positions) - min(positions) if len(positions) > 1 else 0

            if spread > frame_w * 0.3:
                print(f" Camera switching detected (spread={spread:.0f}) wide shot")
                return self._layout_wide_shot(input_path, output_path)

            follow_largest_person = positions[0]
            print(f" YOLO-Pose: centering on skeleton @ x={follow_largest_person:.0f}")
            return self._crop_centered(input_path, output_path, follow_largest_person)

        # 2. MediaPipe
        mp_poses = self._detect_person_pose(input_path)
        if mp_poses:
            nose_x, shoulder_x = mp_poses[0]
            print(f" MediaPipe Pose: centering on face @ x={nose_x:.0f}")
            return self._crop_centered(input_path, output_path, nose_x)

        print(" MediaPipe found no faces, trying YOLO...")

        # 3. YOLO fallback
        people = self._detect_all_people(input_path)
        if people:
            cap = cv2.VideoCapture(input_path)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()

            def distance_from_center(cx_sw):
                return abs(cx_sw[0] - frame_w / 2)

            edge_person = min(people, key=distance_from_center)
            print(f" YOLO fallback: {len(people)} people, using edge @ x={edge_person[0]:.0f}")
            return self._crop_centered(input_path, output_path, edge_person[0])

        print(" No faces/people detected - using wide shot")
        return self._layout_wide_shot(input_path, output_path)

    def _layout_screen_share(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Screen share layout - SPLIT: content on top, webcam on bottom.
        Uses YOLO for PRECISE webcam detection (not Gemini coordinates).
        Gemini only tells us it's screen_share, YOLO finds exact person bbox.
        Layout:
         CONTENT     60% height (screen excluding webcam area)
         WEBCAM      40% height (person, precisely cropped)
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Detect person region
        person = self._detect_person_region(input_path)

        if person:
            px, py, pw, ph = person
            px = int(px)
            person_ratio = pw / frame_w
            print(f" YOLO person: {int(pw)}x{int(ph)}) = {person_ratio:.0%} of frame")

            if person_ratio > 0.5:
                print(f" Person too big for screen_share ({person_ratio:.0%}), using single_speaker")
                return self._layout_single_speaker(input_path, output_path)

            person_center_x = px + pw / 2
            is_left = person_center_x < frame_w * 0.35
            is_right = person_center_x > frame_w * 0.65
            print(f" Position: {int(person_center_x)}, corner={'left' if is_left else 'right'}")

            left_x = int(px) if is_left else 0
            right_x = int(px) if is_right else frame_w

            if is_left:
                print(f" Left: YOLO found at x={left_x}")
            else:
                print(f" Left: using default position (0)")

            if is_right:
                print(f" Right: YOLO found at x={right_x}")
            else:
                print(f" Right: using default position ({frame_w})")
        else:
            left_x = 0
            right_x = frame_w

        content_h = int(CANVAS_H * 0.6)
        webcam_h = CANVAS_H - content_h

        content_h_out = content_h
        webcam_h_out = webcam_h

        target_content_aspect = CANVAS_W / content_h_out
        target_webcam_aspect = CANVAS_W / webcam_h_out

        content_crop_w = int(frame_h * target_content_aspect)
        content_crop_w = min(content_crop_w, frame_w)
        content_x = (frame_w - content_crop_w) // 2

        webcam_crop_w = int(frame_h * target_webcam_aspect)
        webcam_crop_w = min(webcam_crop_w, frame_w)
        webcam_x = max(0, int((person[0] if person else frame_w / 2) - webcam_crop_w / 2)) if person else (frame_w - webcam_crop_w) // 2

        content_scale = f"scale={CANVAS_W}:{content_h_out}:flags=lanczos"
        webcam_scale = f"scale={CANVAS_W}:{webcam_h_out}:flags=lanczos"

        filter_str = (
            f"[0:v]split=2[content_src][webcam_src];"
            f"[content_src]crop={content_crop_w}:{frame_h}:{content_x}:0,"
            f"{content_scale},setsar=1[content_out];"
            f"[webcam_src]crop={webcam_crop_w}:{frame_h}:{webcam_x}:0,"
            f"unsharp=5:5:0.8:5:5:0.0,{webcam_scale},setsar=1[webcam_out];"
            f"[content_out][webcam_out]vstack=inputs=2"
        )

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-map", "0:v?", "-map", "0:a?",
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(" screen_share: content top (60%) + webcam bottom (40%)")
            return output_path
        except Exception as e:
            print(f" Screen share failed: {e}")
            return self._layout_single_speaker(input_path, output_path)

    def _layout_interview(
        self,
        input_path: str,
        output_path: str,
        webcam_position: str | None = None,
        **kwargs,
    ) -> str:
        """
        Interview/Podcast layout - two people, split top/bottom.
        TRUSTS GEMINI 100%: Gemini analyzed the whole video and said this is podcast.
        YOLO only REFINES positions, never changes the layout decision.
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Detect 2 people
        people = self._detect_all_people(input_path)

        left_x = 0
        right_x = frame_w

        if people and len(people) >= 2:
            sorted_p = sorted(people, key=lambda p: p[0])
            left_cx = sorted_p[0][0]
            right_cx = sorted_p[-1][0]
            left_x = int(max(0, left_cx - frame_w * 0.25))
            right_x = int(min(frame_w, right_cx + frame_w * 0.25))
            print(f" Left: YOLO found at x={left_x}")
            print(f" Right: YOLO found at x={right_x}")
        else:
            print(f" Left: using default position (0)")
            print(f" Right: using default position ({frame_w})")

        half_h = CANVAS_H // 2
        crop_w = int(frame_h * CANVAS_W / half_h)
        crop_w = min(crop_w, frame_w)

        left_crop_x = max(0, min(left_x, frame_w - crop_w))
        right_crop_x = max(0, min(right_x - crop_w // 2, frame_w - crop_w))

        filter_str = (
            f"[0:v]crop={crop_w}:{frame_h}:{left_crop_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[top];"
            f"[0:v]crop={crop_w}:{frame_h}:{right_crop_x}:0,"
            f"scale={CANVAS_W}:{half_h}:flags=lanczos,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2[v]"
        )

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "0:a?",
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(" interview: 2 people stacked (YOLO verified)")
            return output_path
        except Exception as e:
            print(f" Interview failed: {e}")
            return self._layout_wide_shot(input_path, output_path)

    def _layout_gameplay(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Gameplay layout - webcam on top, game on bottom.
        Similar to screen_share but webcam is on TOP.
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Webcam position (top portion of frame)
        webcam_info = self._detect_webcam_at_time(input_path, sample_pos=0.2)

        if webcam_info.get("has_webcam"):
            wc_x, wc_y, wc_w, wc_h = webcam_info["webcam_bbox"]
        else:
            # Default top-right corner webcam
            wc_w = int(frame_w * 0.25)
            wc_h = int(wc_w * 3 / 4)
            wc_x = frame_w - wc_w
            wc_y = 0

        wc_ratio = CANVAS_H * 0.3
        game_h = int(CANVAS_H * 0.7)
        wc_new_h = int(CANVAS_H * 0.3)
        wc_new_w = CANVAS_W
        wc_scale = f"{CANVAS_W}:{wc_new_h}"

        game_scale_w = CANVAS_W
        game_filter = (
            f"scale={game_scale_w}:-1:flags=lanczos,"
            f"pad={CANVAS_W}:{game_h}:0:(oh-ih)/2:black,setsar=1"
        )

        filter_str = (
            f"[0:v]split=2[game][webcam];"
            f"[webcam]crop={wc_w}:{wc_h}:{wc_x}:{wc_y},"
            f"scale={wc_scale}:flags=lanczos,setsar=1[webcam_out];"
            f"[game]{game_filter}[game_out];"
            f"[webcam_out][game_out]vstack=inputs=2"
        )

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(f" gameplay: webcam top + game bottom")
            return output_path
        except Exception as e:
            print(f" Gameplay failed: {e}")
            return self._layout_screen_content(input_path, output_path)

    def _layout_webcam_content(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Webcam+Content layout
        Layout: webcam (small, corner) + content area
        """
        print(" Webcam+Content layout")

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret_tmp, tmp_frame = cap.read()
        cap.release()

        webcam_info = self._detect_webcam_at_time(input_path, sample_pos=0.1)

        if not webcam_info.get("has_webcam"):
            print(" No webcam detected, falling back to screen content")
            return self._layout_screen_content(input_path, output_path)

        wc_x, wc_y, wc_w, wc_h = webcam_info["webcam_bbox"]

        # Content area = entire frame minus webcam area
        content_h_out = int(CANVAS_H * 0.6)
        webcam_h_out = CANVAS_H - content_h_out

        content_aspect = CANVAS_W / content_h_out
        webcam_aspect = CANVAS_W / webcam_h_out

        content_crop_w = min(int(frame_h * content_aspect), frame_w)
        content_x = max(0, (frame_w - content_crop_w) // 2)

        webcam_crop_w = min(int(wc_h * webcam_aspect), frame_w)
        webcam_crop_x = max(0, wc_x - (webcam_crop_w - wc_w) // 2)
        webcam_crop_x = min(webcam_crop_x, frame_w - webcam_crop_w)

        content_scale = f"scale={CANVAS_W}:{content_h_out}:flags=lanczos"
        webcam_scale = f"scale={CANVAS_W}:{webcam_h_out}:flags=lanczos"

        filter_str = (
            f"[0:v]split=2[content_src][webcam_src];"
            f"[content_src]crop={content_crop_w}:{frame_h}:{content_x}:0,"
            f"{content_scale},setsar=1[content_out];"
            f"[webcam_src]crop={webcam_crop_w}:{wc_h}:{webcam_crop_x}:{wc_y},"
            f"unsharp=5:5:0.8:5:5:0.0,{webcam_scale},setsar=1[webcam_out];"
            f"[content_out][webcam_out]vstack=inputs=2"
        )

        try:
            temp_silent = output_path.replace(".mp4", ".silent.mp4")
            cmd_video = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_str,
                "-c:v", "libx264", "-crf", "23",
                temp_silent,
            ]
            subprocess.run(cmd_video, capture_output=True, check=True)

            # Merge audio
            codec = get_video_codec_params(input_path)
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", temp_silent, "-i", input_path,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "copy", "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd_audio, capture_output=True, check=True)
            if os.path.exists(temp_silent):
                os.remove(temp_silent)

            print(" Video render complete (Silent). Merging audio...")
            print(" Audio merged successfully.")
            return output_path

        except Exception as e:
            print(f" Webcam+Content failed: {e}")
            return self._layout_screen_content(input_path, output_path)

    def _layout_cinematic(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Cinematic layout - FILL mode (no black bars).
        Scale video to fill the entire 9:16 frame, cropping edges if needed.
        SMART: Uses YOLO to center crop on people, not just frame center.
        ENHANCED: Detects and removes letterbox (black bars) for movie content.
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sample frame for letterbox detection
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
        ret, sample_frame = cap.read()
        cap.release()

        # Detect letterbox
        letterbox_bounds = None
        if ret and sample_frame is not None:
            letterbox_bounds = self._detect_letterbox(sample_frame, frame_h)

        effective_h = frame_h
        letterbox_crop_y = 0

        if letterbox_bounds:
            letterbox_crop_y, effective_h = letterbox_bounds
            print(f" Letterbox: cropping y={letterbox_crop_y} h={effective_h}")

        # Detect people for smart center
        people = self._detect_all_people(input_path)

        target_w = int(effective_h * 9 / 16)
        padded_w = min(target_w, frame_w)

        if people:
            all_centers_x = [p[0] for p in people]
            group_center_x = sum(all_centers_x) / len(all_centers_x)
            crop_x = int(max(0, min(group_center_x - padded_w // 2, frame_w - padded_w)))
            print(f" Cinematic: {len(people)} people, centering on x={group_center_x:.0f}")
        else:
            crop_x = (frame_w - padded_w) // 2
            print(f" Cinematic: no people detected, using center crop")

        # Cinematic padding
        print(f" Cinematic padding:  crop_w={padded_w}")

        vf = (
            f"crop={padded_w}:{effective_h}:{crop_x}:{letterbox_crop_y},"
            f"scale={CANVAS_W}:{CANVAS_H}:flags=lanczos,setsar=1"
        )

        try:
            result = self._run_ffmpeg_filter(input_path, output_path, vf)
            print(f" cinematic: smart fill (cropped at x={crop_x})")
            return result
        except Exception as e:
            print(f" Cinematic failed: {e}")
            return self._layout_wide_shot(input_path, output_path)

    def _layout_group_scene(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """
        Group scene layout - multiple people together (vlog, walking, eating).
        Crop to fit all people in vertical frame WITHOUT splitting.
        Uses YOLO to find bounding box of all people and crops around them.
        Falls back to cinematic (fill) if no people found.
        """
        people = self._detect_all_people(input_path)

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        target_w = int(frame_h * 9 / 16)
        target_w = min(target_w, frame_w)

        if not people:
            print(" Group scene: no people detected, using fill mode")
            return self._layout_cinematic(input_path, output_path)

        all_centers_x = [p[0] for p in people]
        min_x = min(p[0] - p[1] / 2 for p in people)
        max_x = max(p[0] + p[1] / 2 for p in people)
        group_width = max_x - min_x

        if group_width > target_w * 1.2:
            print(f" Group too wide ({group_width:.0f}px), using fill mode")
            return self._layout_cinematic(input_path, output_path)

        group_center_x = (min_x + max_x) / 2
        crop_x = int(max(0, min(group_center_x - target_w // 2, frame_w - target_w)))

        print(f" Group of {len(people)} people, center @ x={group_center_x:.0f}")

        vf = (
            f"crop={target_w}:{frame_h}:{crop_x}:0,"
            f"scale={CANVAS_W}:{CANVAS_H}:flags=lanczos,setsar=1"
        )
        try:
            result = self._run_ffmpeg_filter(input_path, output_path, vf)
            print(f" group_scene: cropped to fit all people")
            return result
        except Exception as e:
            return self._layout_cinematic(input_path, output_path)

    def _layout_auto_detect(
        self, input_path: str, output_path: str, **kwargs
    ) -> str:
        """Auto-detect best layout based on YOLO."""
        people = self._detect_all_people(input_path)

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        if not people:
            return self._layout_wide_shot(input_path, output_path)

        if len(people) == 1:
            return self._crop_centered(input_path, output_path, people[0][0])

        fits, center_x, spread = self._check_people_fit([p[0] for p in people], frame_w)
        if fits:
            return self._crop_centered(input_path, output_path, center_x)
        return self._layout_talking_heads(input_path, output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # _apply_layout dispatcher
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_layout(
        self,
        input_path: str,
        output_path: str,
        layout: str,
        **kwargs,
    ) -> str:
        """
        Apply layout based on SCENE TYPE from Gemini.
        YOLO handles actual person detection and positioning.
        """
        layout_map = {
            "talking_heads":       self._layout_talking_heads,
            "screen_content":      self._layout_screen_content,
            "single_speaker":      self._layout_single_speaker,
            "podcast":             self._layout_interview,
            "interview":           self._layout_interview,
            "speaker_left":        self._layout_interview,
            "speaker_right":       self._layout_interview,
            "split_both":          self._layout_interview,
            "group_shot":          self._layout_group_scene,
            "group_scene":         self._layout_cinematic,
            "gameplay":            self._layout_gameplay,
            "screen_share":        self._layout_screen_share,
            "wide_shot":           self._layout_wide_shot,
            "webcam_corner":       self._layout_auto_detect,
            "webcam_content":      self._layout_webcam_content,
            "full_screen_webcam":  self._layout_single_speaker,
            "screen_only":         self._layout_screen_only,
            "letterbox_full_frame": self._layout_letterbox_full_frame,
            "full_frame":          self._layout_full_frame,
            "cinematic":           self._layout_cinematic,
            "ab_roll":             self._layout_wide_shot,
            "faceless":            self._layout_cinematic,
        }

        fn = layout_map.get(layout)
        if fn:
            return fn(input_path, output_path, **kwargs)

        print(f" Unknown layout '{layout}', detecting people...")
        return self._layout_auto_detect(input_path, output_path, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Sports video composition
    # ─────────────────────────────────────────────────────────────────────────

    def _compose_sports_video(
        self,
        input_path: str,
        output_path: str,
        fps: float,
        config: dict,
    ) -> str:
        """
        SPORTS MODE: Frame-by-frame ball/action tracking.
        Unlike other modes that use scene-based layouts, sports mode:
        1. Tracks ball position every frame (or every N frames for speed)
        2. Finds player closest to ball
        3. Crops around that player
        4. Smoothly interpolates between positions
        """
        print(" Starting sports composition...")

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or fps

        crop_padding = config.get("crop_padding", 0.1)
        transition_speed = config.get("transition_speed", 0.333)
        detection_interval = max(1, int(fps // 3))

        crop_w = int(frame_h * 9 / 16)
        crop_w = min(crop_w, frame_w)
        crop_h = frame_h

        print(f" Crop size: {crop_w}x{crop_h}, detection every {detection_interval} frames")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            temp_video = tf.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (CANVAS_W, CANVAS_H))

        last_crop_x = frame_w // 2
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.sports_detector and frame_idx % detection_interval == 0:
                try:
                    action_x, action_y, confidence = self.sports_detector.get_action_crop_center(
                        frame
                    )
                    if action_x is not None:
                        target_crop_x = int(action_x)
                    else:
                        target_crop_x = last_crop_x
                except Exception:
                    target_crop_x = last_crop_x
            else:
                target_crop_x = last_crop_x

            # Smooth transition
            last_crop_x = int(
                transition_speed * target_crop_x + (1.0 - transition_speed) * last_crop_x
            )
            x_start = max(0, min(last_crop_x - crop_w // 2, frame_w - crop_w))

            crop_region = frame[:crop_h, x_start:x_start + crop_w]
            output_frame = cv2.resize(crop_region, (CANVAS_W, CANVAS_H))
            out.write(output_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                progress = frame_idx / total_frames * 100
                print(f" Progress: {progress:.1f}%   ")

        print(f" Processed {frame_idx} frames")
        cap.release()
        out.release()

        # Merge audio
        print(" Adding original audio...")
        codec = get_video_codec_params(input_path)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", codec.get("-c:v", "libx264"),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            os.remove(temp_video)
            print(f" Sports video saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f" Audio muxing failed, using video only: {e}")
            shutil.move(temp_video, output_path)

        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Segment processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_segments_with_centers(
        self,
        input_path: str,
        output_path: str,
        segments: list,
    ) -> str:
        """
        Process segments using SAVED person centers (no re-detection!)
        """
        if not segments:
            print(" No segments processed")
            return input_path

        temp_dir = tempfile.mkdtemp()
        temp_files: list = []

        try:
            fps_cap = cv2.VideoCapture(input_path)
            fps = fps_cap.get(cv2.CAP_PROP_FPS) or 25.0
            fps_cap.release()

            for i, seg in enumerate(segments):
                seg_start = seg.get("from", seg.get("start", 0.0))
                seg_end = seg.get("to", seg.get("end", seg_start + 1.0))
                seg_duration = seg_end - seg_start
                layout = seg.get("layout", "wide_shot")
                centers = seg.get("person_centers", [])

                temp_input = os.path.join(temp_dir, f"seg_{i:02d}_input.mp4")
                temp_output = os.path.join(temp_dir, f"seg_{i:02d}_output.mp4")

                print(f" Segment {i + 1}/{len(segments)} ({seg_duration:.1f}s, centers={centers})")

                # Extract segment
                cmd_extract = [
                    "ffmpeg", "-y",
                    "-loglevel", "error",
                    "-ss", f"{seg_start:.3f}",
                    "-i", input_path,
                    "-t", f"{seg_duration:.3f}",
                    "-c:a", "aac", "-b:a", "128k",
                    temp_input,
                ]
                subprocess.run(cmd_extract, capture_output=True, check=True)

                # Apply layout using saved centers
                if centers:
                    best_center = float(np.median(centers))
                    self._crop_centered(temp_input, temp_output, best_center)
                else:
                    self._apply_layout(temp_input, temp_output, layout)

                if os.path.exists(temp_output):
                    temp_files.append(temp_output)

            if not temp_files:
                print(" No segments processed successfully")
                return input_path

            # Concat
            concat_file = os.path.join(temp_dir, "concat.txt")
            with open(concat_file, "w") as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path,
            ]
            subprocess.run(concat_cmd, capture_output=True, check=True)
            print(f" Composed {len(temp_files)} segments")
            return output_path

        except Exception as e:
            print(f" Error: {e}")
            traceback.print_exc()
            return input_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _apply_layout_with_centers(
        self,
        segment: dict,
        input_path: str,
        output_path: str,
    ) -> str:
        """
        Apply layout - MULTI-SAMPLE detection across segment!
        Camera might switch between 1 and 2 people within segment.
        """
        seg_start = segment.get("from", segment.get("start", 0.0))
        seg_end = segment.get("to", segment.get("end", seg_start + 5.0))
        layout = segment.get("layout", "wide_shot")
        saved_centers = segment.get("person_centers", [])

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        duration = seg_end - seg_start
        sample_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
        all_samples: list = []
        empty_count = 0
        single_person_count = 0
        two_people_count = 0
        all_person_positions: list = []

        for pct in sample_percentages:
            t = seg_start + pct * duration
            detections = self._detect_people_at_time(input_path, t)
            if not detections:
                empty_count += 1
                all_samples.append([])
            elif len(detections) == 1:
                single_person_count += 1
                all_samples.append(detections)
                all_person_positions.extend([d[0] for d in detections])
            else:
                two_people_count += 1
                all_samples.append(detections)
                all_person_positions.extend([d[0] for d in detections])

        total_valid = len(sample_percentages) - empty_count
        total_samples = len(sample_percentages)

        # Use saved centers if majority is empty but centers exist
        if empty_count > total_samples // 2 and saved_centers:
            best_center = float(np.median(saved_centers))
            print(f" Majority empty ({empty_count}) BUT saved centers exist ({len(saved_centers)}). Using saved.")
            return self._crop_centered(input_path, output_path, best_center)

        if empty_count > total_samples // 2:
            print(f" wide_shot (camera switching)")
            return self._apply_layout(input_path, output_path, "wide_shot")

        # Determine dominant situation
        if two_people_count >= total_samples // 2:
            # Majority 2-person
            all_two_person_detections = [s for s in all_samples if len(s) >= 2]
            if all_two_person_detections:
                best_sample = max(all_two_person_detections, key=len)
                centers = sorted([d[0] for d in best_sample])
                spread = centers[-1] - centers[0] if len(centers) > 1 else 0
                if len(centers) >= 2 and spread > frame_w * 0.2:
                    print(f" All samples show 2 people  SPLIT (L={centers[0]:.0f}, R={centers[-1]:.0f})")
                    return self._layout_split_by_centers(input_path, output_path, centers)
                else:
                    best_center = float(np.median(centers))
                    print(f" 2 people close ({spread:.0f}px)  crop @ {best_center:.0f}")
                    return self._crop_centered(input_path, output_path, best_center)
            print(" 2 people but positions unclear ")
            return self._apply_layout(input_path, output_path, layout)

        # Majority single person
        if all_person_positions:
            best_center = float(np.median(all_person_positions))
            print(f" Majority single ({single_person_count}) @ {best_center:.0f} crop")
            return self._crop_centered(input_path, output_path, best_center)

        print(f" u      DEBUG: Majority empty decision returned  wide_shot")
        return self._apply_layout(input_path, output_path, "wide_shot")

    def _process_segments(
        self,
        input_path: str,
        output_path: str,
        segments: list,
    ) -> str:
        """Process segments and create final video (legacy - uses re-detection)."""
        return self._process_segments_with_centers(input_path, output_path, segments)

    # ─────────────────────────────────────────────────────────────────────────
    # Main public API
    # ─────────────────────────────────────────────────────────────────────────

    def compose_clip_auto(
        self, input_path: str, output_path: str
    ) -> str:
        """
        AUTO COMPOSE: Uses PySceneDetect for scene detection + YOLO for layout.
        IMPORTANT: YOLO analyzes FIRST FRAME of each scene and SAVES nose positions.
        Render uses SAVED positions - no re-detection!
        """
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = total_frames / fps

        config = self._get_video_type_config()

        print(f" AutoComposer: Processing {os.path.basename(input_path)}")
        print(f" Input: {duration:.0f}s @ {fps:.1f}fps")

        if self.video_type:
            print(f" Video Type: {self.video_type}")
        else:
            print(" Video Type: auto (not specified)")

        print(f" Config: priority={config.get('priority')}, detect_webcam={config.get('detect_webcam')}")

        # ── Sports mode (frame-by-frame ball tracking) ────────────────────
        if config.get("detect_ball") and config.get("priority") == "ball":
            print(" Sports mode: Using ball/action tracking")
            return self._compose_sports_video(input_path, output_path, fps, config)

        # ── Movies mode (letterbox, no cropping) ──────────────────────────
        if config.get("preserve_full_frame") and config.get("detect_letterbox"):
            print(" Movies mode: Using full frame letterbox (no cropping)")
            return self._layout_letterbox_full_frame(input_path, output_path)

        # ── Scene detection ───────────────────────────────────────────────
        print(" Detecting scenes with PySceneDetect...")
        scenes = self.detect_scenes(input_path)
        print(f" Found {len(scenes)} scene(s)")

        # ── For single segment (short video / no cuts) ────────────────────
        if len(scenes) == 1:
            return self._process_single_scene(input_path, output_path, scenes[0], fps, config)

        # ── Multi-scene: process each scene separately ────────────────────
        MIN_SEGMENT_DURATION = 0.5
        merged_segments: list = []

        for i, (sc_start, sc_end) in enumerate(scenes):
            sc_dur = sc_end - sc_start
            if sc_dur < MIN_SEGMENT_DURATION and merged_segments:
                # Merge short segment into previous
                prev_start, prev_end, prev_layout, prev_centers = merged_segments[-1]
                merged_segments[-1] = (prev_start, sc_end, prev_layout, prev_centers)
                print(f" Merged short segment ({sc_dur:.1f}s) into previous")
                continue

            # Detect layout for this scene
            t_sample = sc_start + sc_dur * 0.1
            detections = self._detect_people_at_time(input_path, t_sample)

            if len(detections) >= 2:
                positions = [d[0] for d in detections]
                fits, center_x, spread = self._check_people_fit(positions, frame_w)
                if fits:
                    layout = "single_speaker"
                    person_centers = [center_x]
                else:
                    layout = "talking_heads"
                    person_centers = positions
            elif len(detections) == 1:
                layout = "single_speaker"
                person_centers = [detections[0][0]]
            else:
                layout = "wide_shot"
                person_centers = []

            merged_segments.append((sc_start, sc_end, layout, person_centers))

        # Build segment dicts
        segments: list = []
        for seg_start, seg_end, layout, centers in merged_segments:
            segments.append({
                "from": seg_start,
                "to": seg_end,
                "layout": layout,
                "person_centers": centers,
            })

        return self._process_segments_with_centers(input_path, output_path, segments)

    def _process_single_scene(
        self,
        input_path: str,
        output_path: str,
        scene: tuple,
        fps: float,
        config: dict,
    ) -> str:
        """Process a single scene (no cuts)."""
        sc_start, sc_end = scene
        duration = sc_end - sc_start

        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        # Webcam detection
        should_detect_webcam = config.get("detect_webcam", False)
        webcam_info: dict | None = None

        if should_detect_webcam:
            webcam_info = self._detect_webcam_overlay(input_path)

        if webcam_info and webcam_info.get("has_webcam"):
            content_type = webcam_info.get("content_type", "webcam_corner")
            is_full_screen = webcam_info.get("is_full_screen", False)

            if is_full_screen:
                print(f" Full-screen person (skipping webcam_content)")
                return self._layout_single_speaker(input_path, output_path)
            else:
                print(f" webcam+content @ {sc_start:.1f}s")
                return self._layout_webcam_content(input_path, output_path)

        # Screen-only detection (screenReacts with no webcam)
        if config.get("fallback_on_screen") and config.get("check_screen_share"):
            print(" screen_only (screenReacts, no large person)")
            return self._layout_screen_only(input_path, output_path)

        # People detection
        sample_times = [sc_start + duration * pct for pct in [0.1, 0.3, 0.5, 0.7, 0.9]]
        all_detections: list = []
        two_foreground_people_same_frame = False

        for t in sample_times:
            dets = self._detect_people_at_time(input_path, t)
            if len(dets) >= 2:
                two_foreground_people_same_frame = True
            all_detections.extend(dets)

        if not all_detections:
            # No people found
            if config.get("fallback_on_screen"):
                return self._layout_screen_only(input_path, output_path)
            return self._layout_wide_shot(input_path, output_path)

        # People classification
        all_foreground_positions = [d[0] for d in all_detections]
        num_samples = len(sample_times)

        if two_foreground_people_same_frame:
            # 2+ people detected simultaneously
            spread = max(all_foreground_positions) - min(all_foreground_positions)
            print(f" Split candidate found! spread={spread:.0f}")

            if config.get("split_never"):
                print(" split_never: forcing wide_shot instead of split screen")
                center_x = float(np.median(all_foreground_positions))
                return self._crop_centered(input_path, output_path, center_x)

            fits, center_x, spread_ratio = self._check_people_fit(
                all_foreground_positions, frame_w
            )

            if fits:
                if config.get("prefer_group_shot") or config.get("group_aware"):
                    print(f" Group fits (spread={spread_ratio:.0%}) group_shot @ {center_x:.0f}")
                    return self._crop_centered(input_path, output_path, center_x)

            if config.get("split_on_dialog"):
                print(f" Dialog detected  talking_heads split")
                poses = self._detect_with_yolo_pose(input_path, num_samples=5)
                if poses and len(poses) >= 2:
                    nose_positions = sorted([p[0] for p in poses])
                    return self._layout_split_by_nose(input_path, output_path, nose_positions)

            # Group shot or wide shot
            if config.get("group_aware"):
                center_x = float(np.median(all_foreground_positions))
                return self._layout_group_scene(input_path, output_path)

            return self._layout_wide_shot(input_path, output_path)

        # Single-ish situation
        positions = sorted(set(round(p) for p in all_foreground_positions))
        most_common_band = float(np.median(all_foreground_positions))

        print(f" Single person @ {most_common_band:.0f}")
        if config.get("cinematic"):
            return self._layout_cinematic(input_path, output_path)

        return self._crop_centered(input_path, output_path, most_common_band)

    def compose_clip(
        self,
        input_path: str,
        output_path: str,
        segments: list | None = None,
    ) -> str:
        """
        Compose a clip using Gemini's layout segments.
        GEMINI DISABLED - Using AUTO mode with PySceneDetect instead!
            input_path: Path to input video clip
            output_path: Path for output composed video
            segments: List of layout segments from Gemini (IGNORED - using auto detection)
        """
        print(" [GEMINI DISABLED] Using PySceneDetect + YOLO auto detection")
        print(" GEMINI DISABLED - Using AUTO mode with PySceneDetect instead!")
        return self.compose_clip_auto(input_path, output_path)
