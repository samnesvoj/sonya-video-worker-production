"""
lip_sync_detector.py
====================
Active Speaker Detection using MediaPipe Face Landmarker.
Detects who is speaking by analyzing lip movement (Lip Aspect Ratio / LAR).

Reconstructed from: lip_sync_detector.cp311-win_amd64.pyd
Method: static analysis of Nuitka-compiled binary.

──────────────────────────────────────────────────────────────
SETUP — download the MediaPipe model (run once):

    curl -L -o face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Place face_landmarker.task next to this file (or in the backend root).

INSTALL dependencies:
    pip install mediapipe>=0.10.0 opencv-python>=4.8.0 numpy>=1.24.0
──────────────────────────────────────────────────────────────

Confirmed recovered elements (static analysis):
  • LipSyncDetector — full class with all methods
  • TextDetector — full class with all methods
  • analyze_speakers_in_clip() — module-level convenience wrapper
  • speaking_threshold = 0.003   (decoded float constant from binary)
  • history_size     = 5         (decoded int constant from binary)
  • min_segment_duration = 0.5   (decoded float constant from binary)
  • Lip landmark indices decoded from Nuitka constants blob
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print(" MediaPipe not available. Install with: pip install mediapipe")

__all__ = [
    "LipSyncDetector",
    "TextDetector",
    "analyze_speakers_in_clip",
    "get_model_path",
    "MEDIAPIPE_AVAILABLE",
]


# ──────────────────────────────────────────────────────────────
# Model path helper
# ──────────────────────────────────────────────────────────────

def get_model_path(model_name: str) -> str:
    """
    Get path to a MediaPipe model file, checking multiple locations:
      1. Directory of this script
      2. PyInstaller _MEIPASS bundle directory
      3. Current working directory

    Raises FileNotFoundError with a clear download hint if not found.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name),
        os.path.join(getattr(sys, "_MEIPASS", ""), model_name),
        os.path.join(os.getcwd(), model_name),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"MediaPipe face landmarker model not found: {model_name}\n"
        "Download it with:\n"
        "  curl -L -o face_landmarker.task \\\n"
        "    https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task\n"
        "Then place it next to lip_sync_detector.py."
    )


# ──────────────────────────────────────────────────────────────
# LipSyncDetector
# ──────────────────────────────────────────────────────────────

class LipSyncDetector:
    """
    Detects active speakers by analyzing lip movement using MediaPipe Face Landmarker.

    How it works:
    1. Detect all faces in the frame using MediaPipe FaceLandmarker.
    2. For each face, calculate Lip Aspect Ratio (LAR):
           LAR = vertical lip distance / horizontal lip distance
    3. Track LAR variance over time per spatial zone:
           high LAR variance  →  person is speaking
    4. Return which zone/person is currently the active speaker.
    """

    # ── MediaPipe 478-point Face Mesh lip landmark groups ─────────────────────
    # Each tuple holds indices that are averaged to get a stable edge position.
    # Decoded directly from Nuitka constants blob in the .pyd binary.
    UPPER_LIP_TOP: Tuple[int, ...] = (13, 312, 311, 310, 415)    # upper inner lip
    UPPER_LIP_BOTTOM: Tuple[int, ...] = (14, 317, 402, 318, 324) # lower inner lip (top)
    LOWER_LIP_TOP: Tuple[int, ...] = (14, 87, 178, 88, 95)       # lower inner lip (bottom)
    LOWER_LIP_BOTTOM: Tuple[int, ...] = (17, 84, 181, 91, 146)   # outer lower lip

    # Single landmark indices used as shortcuts
    LIP_TOP: int = 13     # upper inner lip center
    LIP_BOTTOM: int = 14  # lower inner lip center
    LIP_LEFT: int = 61    # left mouth corner
    LIP_RIGHT: int = 291  # right mouth corner

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        num_faces: int = 2,
        min_face_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        frame_width: int = 1920,
    ) -> None:
        """
        Args:
            num_faces: Maximum faces to detect per frame.
            min_face_detection_confidence: MediaPipe detection threshold.
            min_tracking_confidence: MediaPipe tracking threshold.
            frame_width: Default frame width for zone calculation.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for lip sync detection")

        model_path = get_model_path("face_landmarker.task")

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Per-zone rolling LAR history  {zone_id: [lar, ...]}
        self.lar_history: Dict[str, List[float]] = {}

        # Constants decoded from the compiled binary:
        self.history_size: int = 5              # frames of history per zone
        self.speaking_threshold: float = 0.003  # LAR variance → speaking
        self.num_zones: int = 4                 # horizontal spatial zones
        self.frame_width: int = frame_width

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "LipSyncDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Zone helpers ──────────────────────────────────────────────────────────

    def _get_position_zone(self, center_x: float, frame_width: int) -> str:
        """Get zone ID based on X position. Zones are stable across frames."""
        zone_width = frame_width / self.num_zones
        zone_id = max(0, min(self.num_zones - 1, int(center_x / zone_width)))
        return f"zone_{zone_id}"

    # ── LAR calculation ───────────────────────────────────────────────────────

    def calculate_lip_aspect_ratio(
        self,
        landmarks,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Calculate Lip Aspect Ratio (LAR) = vertical_distance / horizontal_distance.
        Higher variance in LAR over time → person is speaking.
        """
        top_y = (
            np.mean([landmarks[i].y for i in self.UPPER_LIP_TOP]) * image_height
        )
        bottom_y = (
            np.mean([landmarks[i].y for i in self.LOWER_LIP_BOTTOM]) * image_height
        )
        left_x = landmarks[self.LIP_LEFT].x * image_width
        right_x = landmarks[self.LIP_RIGHT].x * image_width

        vertical = abs(bottom_y - top_y)
        horizontal = abs(right_x - left_x)
        return vertical / (horizontal + 1e-6)

    def get_face_center_x(
        self,
        landmarks,
        image_width: int,
        image_height: int,  # kept for API symmetry
    ) -> float:
        """Get horizontal face center via nose tip (for YOLO person matching)."""
        nose = landmarks[1]  # nose tip — landmark index 1 in MediaPipe 478-pt model
        return nose.x * image_width

    # ── Frame-level ───────────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Analyze one frame and return a list of per-face dicts.

        Each dict contains:
          zone              – str,   stable position-based zone ID
          center_x          – float, pixel X of face center
          center_y          – float, pixel Y of face center
          lar               – float, current Lip Aspect Ratio
          is_speaking       – bool,  True when LAR variance > threshold
          speaking_confidence – float, 0–1
          bbox              – dict  {left, right, top, bottom} in pixels
          face_id           – int,  index in MediaPipe result
        """
        if cv2 is None or self.face_landmarker is None:
            return []

        image_height, image_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_landmarker.detect(mp_image)

        faces_result: List[Dict] = []
        for face_id, landmarks in enumerate(results.face_landmarks):
            lar = self.calculate_lip_aspect_ratio(landmarks, image_width, image_height)
            center_x = self.get_face_center_x(landmarks, image_width, image_height)
            center_y = float(landmarks[1].y * image_height)

            zone = self._get_position_zone(center_x, image_width)

            # Rolling LAR history for variance tracking
            history = self.lar_history.setdefault(zone, [])
            history.append(lar)
            if len(history) > self.history_size:
                history.pop(0)

            lar_variance = float(np.var(history))
            is_speaking = lar_variance > self.speaking_threshold
            speaking_confidence = min(1.0, lar_variance / (self.speaking_threshold * 2))

            # Tight bounding box from all landmark coordinates
            xs = [lm.x * image_width for lm in landmarks]
            ys = [lm.y * image_height for lm in landmarks]
            bbox = {
                "left": float(min(xs)),
                "right": float(max(xs)),
                "top": float(min(ys)),
                "bottom": float(max(ys)),
            }

            faces_result.append({
                "zone": zone,
                "center_x": float(center_x),
                "center_y": center_y,
                "lar": float(lar),
                "is_speaking": is_speaking,
                "speaking_confidence": float(speaking_confidence),
                "bbox": bbox,
                "face_id": face_id,
            })

        return faces_result

    # ── Segment-level ─────────────────────────────────────────────────────────

    def analyze_video_segment(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        sample_fps: float = 2.0,
    ) -> Dict:
        """
        Analyze a video segment and return speaking patterns.

        Args:
            video_path:  Path to the video file.
            start_time:  Start time in seconds (default 0).
            duration:    How many seconds to analyze. None = entire video.
            sample_fps:  Frames per second to sample (lower → faster).

        Returns:
            {
                "speakers": [
                    {"speaker_id": "zone_0", "average_x": 320.0, "position": "left"},
                    ...
                ],
                "speaking_segments": [
                    {"start": 0.0, "end": 3.5, "speaker_id": "zone_0", "speaker_x": 315.0},
                    ...
                ],
                "active_speaker_timeline": [
                    {
                        "timestamp": 0.5,
                        "active_speaker_id": "zone_0",
                        "active_speaker_x": 320.0,
                        "confidence": 0.75,
                        "all_faces": [...],
                    },
                    ...
                ],
                "frames_analyzed": int,
                "duration_analyzed": float,
            }
        """
        if cv2 is None:
            return _empty_result()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_duration = total_frames / fps

        end_time = (start_time + duration) if duration is not None else video_duration
        frame_interval = max(1, int(round(fps / sample_fps)))
        start_frame = int(start_time * fps)
        end_frame = min(total_frames, int(end_time * fps))

        speaker_timeline: List[Dict] = []
        speaker_positions: Dict[str, List[float]] = {}
        frame_count = 0

        frame_idx = start_frame
        while frame_idx < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            all_faces = self.analyze_frame(frame)

            # Pick the face with the highest speaking confidence
            active_speaker: Optional[Dict] = None
            max_confidence = 0.0
            for face in all_faces:
                if face["is_speaking"] and face["speaking_confidence"] > max_confidence:
                    max_confidence = face["speaking_confidence"]
                    active_speaker = face

            # Accumulate position history per zone
            for face in all_faces:
                speaker_positions.setdefault(face["zone"], []).append(face["center_x"])

            active_speaker_id = active_speaker["zone"] if active_speaker else None
            active_speaker_x = active_speaker["center_x"] if active_speaker else None
            confidence = active_speaker["speaking_confidence"] if active_speaker else 0.0

            speaker_timeline.append({
                "timestamp": timestamp,
                "active_speaker_id": active_speaker_id,
                "active_speaker": active_speaker,
                "active_speaker_x": active_speaker_x,
                "confidence": confidence,
                "all_faces": all_faces,
            })

            frame_count += 1
            frame_idx += frame_interval

        cap.release()

        # Build speakers list sorted left → right
        speakers = [
            {
                "speaker_id": zone_id,
                "average_x": float(np.mean(positions)),
                "position": "left" if np.mean(positions) < image_width / 2 else "right",
            }
            for zone_id, positions in speaker_positions.items()
        ]
        speakers.sort(key=lambda s: s["average_x"])

        speaking_segments = self._create_speaking_segments(speaker_timeline)
        duration_analyzed = (end_frame - start_frame) / fps

        return {
            "speakers": speakers,
            "speaking_segments": speaking_segments,
            "active_speaker_timeline": speaker_timeline,
            "frames_analyzed": frame_count,
            "duration_analyzed": duration_analyzed,
        }

    def _create_speaking_segments(
        self,
        timeline: List[Dict],
        min_segment_duration: float = 0.5,
    ) -> List[Dict]:
        """Convert frame-by-frame timeline into continuous speaking segments."""
        segments: List[Dict] = []
        if not timeline:
            return segments

        current_speaker: Optional[str] = None
        segment_start: float = timeline[0]["timestamp"]
        x_samples: List[float] = []

        def _flush(end_ts: float) -> None:
            if current_speaker is not None and (end_ts - segment_start) >= min_segment_duration:
                segments.append({
                    "start": segment_start,
                    "end": end_ts,
                    "speaker_id": current_speaker,
                    "speaker_x": float(np.mean(x_samples)) if x_samples else 0.0,
                })

        for entry in timeline:
            ts: float = entry["timestamp"]
            spk_id: Optional[str] = entry["active_speaker_id"]
            spk_x: Optional[float] = entry["active_speaker_x"]

            if spk_id != current_speaker:
                _flush(ts)
                current_speaker = spk_id
                segment_start = ts
                x_samples = []

            if spk_x is not None:
                x_samples.append(spk_x)

        # Close the last segment
        if timeline:
            _flush(timeline[-1]["timestamp"])

        return segments

    def get_active_speaker_at_time(
        self,
        analysis_result: Dict,
        timestamp: float,
    ) -> Optional[Dict]:
        """Return the timeline entry nearest to the given timestamp."""
        timeline: List[Dict] = analysis_result.get("active_speaker_timeline", [])
        if not timeline:
            return None
        return min(timeline, key=lambda e: abs(e["timestamp"] - timestamp))

    def close(self) -> None:
        """Release MediaPipe resources."""
        if getattr(self, "face_landmarker", None) is not None:
            try:
                self.face_landmarker.close()
            except Exception:
                pass
            self.face_landmarker = None


# ──────────────────────────────────────────────────────────────
# TextDetector
# ──────────────────────────────────────────────────────────────

class TextDetector:
    """
    Detects text regions in video frames to prevent cropping important text
    (subtitles, on-screen UI, captions).

    Uses three strategies in descending order of accuracy:
      1. EasyOCR  (if installed and use_easyocr=True)
      2. MSER (Maximally Stable Extremal Regions) — OpenCV built-in, fast
      3. Edge + morphology heuristic — lightest fallback
    """

    def __init__(self, use_easyocr: bool = False) -> None:
        self.use_easyocr = use_easyocr
        self.reader = None

        if use_easyocr:
            try:
                import easyocr  # type: ignore
                self.reader = easyocr.Reader(["en"], gpu=False)
            except ImportError:
                print(" EasyOCR not available. Using heuristic text detection.")
                self.use_easyocr = False

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_text_regions(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect regions that contain text.

        Returns:
            List of dicts:  {left, top, right, bottom, text, confidence}
        """
        if self.use_easyocr and self.reader is not None:
            return self._detect_with_ocr(frame)
        if cv2 is not None:
            regions = self._detect_with_mser(frame)
            if not regions:
                regions = self._detect_with_heuristics(frame)
            return regions
        return []

    def has_text(self, frame: np.ndarray, min_area_ratio: float = 0.02) -> bool:
        """
        Quick check: does this frame contain visible text?

        Args:
            frame:          BGR video frame.
            min_area_ratio: Minimum combined text-region area as fraction of
                            total frame area (default 2 %).
        Returns:
            True if text occupies at least min_area_ratio of the frame.
        """
        return self.has_important_text(frame, min_area_ratio=min_area_ratio)

    def has_important_text(
        self,
        frame: np.ndarray,
        min_area_ratio: float = 0.05,
    ) -> bool:
        """
        Returns True if detected text regions cover at least min_area_ratio
        of the total frame area.  Use has_text() for a lighter 2 % threshold.
        """
        if cv2 is None:
            return False
        h, w = frame.shape[:2]
        total_area = w * h
        regions = self.detect_text_regions(frame)
        text_area = sum(
            (r["right"] - r["left"]) * (r["bottom"] - r["top"]) for r in regions
        )
        return (text_area / (total_area + 1e-6)) >= min_area_ratio

    # ── Detection backends ────────────────────────────────────────────────────

    def _detect_with_ocr(self, frame: np.ndarray) -> List[Dict]:
        """EasyOCR — most accurate, slowest."""
        if self.reader is None:
            return []
        raw = self.reader.readtext(frame)
        regions: List[Dict] = []
        for bbox_pts, text, conf in raw:
            if conf < 0.333:
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            regions.append({
                "text": text,
                "left": int(min(xs)),
                "top": int(min(ys)),
                "right": int(max(xs)),
                "bottom": int(max(ys)),
                "confidence": float(conf),
            })
        return regions

    def _detect_with_mser(self, frame: np.ndarray) -> List[Dict]:
        """
        MSER (Maximally Stable Extremal Regions) — good text/UI detector,
        no external model required.
        """
        if cv2 is None:
            return []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mser = cv2.MSER_create(
            _delta=5,
            _min_area=60,
            _max_area=int(w * h * 0.05),
        )
        regions_raw, _ = mser.detectRegions(gray)

        # Merge overlapping bounding boxes into text-line candidates
        bboxes: List[Dict] = []
        for pts in regions_raw:
            rx, ry, rw, rh = cv2.boundingRect(pts.reshape(-1, 1, 2))
            aspect = rw / (rh + 1e-6)
            area_ratio = (rw * rh) / (w * h)
            # Text-like: rectangular, not microscopic, not full-frame
            if 1.5 < aspect < 30 and 0.0005 < area_ratio < 0.15:
                bboxes.append({"left": rx, "top": ry,
                                "right": rx + rw, "bottom": ry + rh})

        if not bboxes:
            return []

        # Merge boxes that are horizontally close (text on the same line)
        merged = _merge_boxes(bboxes, x_gap=20, y_gap=10)
        return [
            {
                "text": "[detected]",
                "left": b["left"],
                "top": b["top"],
                "right": b["right"],
                "bottom": b["bottom"],
                "confidence": 0.8,
            }
            for b in merged
        ]

    def _detect_with_heuristics(self, frame: np.ndarray) -> List[Dict]:
        """
        Edge + morphology fallback.  Fast, less precise than MSER.
        Detects high-contrast horizontal bands typical of subtitles / captions.
        """
        if cv2 is None:
            return []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(edges, kernel)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions: List[Dict] = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / (ch + 1e-6)
            area_ratio = (cw * ch) / (w * h)
            if aspect > 2.0 and 0.001 < area_ratio < 0.3:
                regions.append({
                    "text": "[detected]",
                    "left": cx,
                    "top": cy,
                    "right": cx + cw,
                    "bottom": cy + ch,
                    "confidence": 1.0,
                })
        return regions


# ──────────────────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────────────────

def _merge_boxes(
    boxes: List[Dict],
    x_gap: int = 20,
    y_gap: int = 10,
) -> List[Dict]:
    """Merge bounding boxes that are close to each other."""
    if not boxes:
        return []
    merged = [boxes[0].copy()]
    for b in boxes[1:]:
        absorbed = False
        for m in merged:
            if (
                b["left"] <= m["right"] + x_gap
                and b["right"] >= m["left"] - x_gap
                and b["top"] <= m["bottom"] + y_gap
                and b["bottom"] >= m["top"] - y_gap
            ):
                m["left"] = min(m["left"], b["left"])
                m["top"] = min(m["top"], b["top"])
                m["right"] = max(m["right"], b["right"])
                m["bottom"] = max(m["bottom"], b["bottom"])
                absorbed = True
                break
        if not absorbed:
            merged.append(b.copy())
    return merged


def _empty_result() -> Dict:
    return {
        "speakers": [],
        "speaking_segments": [],
        "active_speaker_timeline": [],
        "frames_analyzed": 0,
        "duration_analyzed": 0.0,
    }


# ──────────────────────────────────────────────────────────────
# Module-level convenience wrapper
# ──────────────────────────────────────────────────────────────

def analyze_speakers_in_clip(
    video_path: str,
    duration: Optional[float] = None,
) -> Dict:
    """
    Analyze a video clip and return speaker information.

    Args:
        video_path: Path to the video file.
        duration:   Duration in seconds to analyze (None = full clip).

    Returns:
        Dict with keys: speakers, speaking_segments,
                        active_speaker_timeline, frames_analyzed, duration_analyzed.
        Returns an empty-result dict if MediaPipe is unavailable (pipeline-safe).
    """
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available — returning empty lip-sync result.")
        return _empty_result()

    try:
        with LipSyncDetector() as detector:
            return detector.analyze_video_segment(video_path, duration=duration)
    except FileNotFoundError as exc:
        print(f"[lip_sync_detector] {exc}")
        return _empty_result()
    except Exception as exc:  # noqa: BLE001
        print(f"[lip_sync_detector] Error during analysis: {exc}")
        return _empty_result()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lip_sync_detector.py <video_path> [duration_sec]")
        sys.exit(1)

    _video = sys.argv[1]
    _dur = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"Analyzing: {_video}")

    _result = analyze_speakers_in_clip(_video, duration=_dur)

    _speakers = _result.get("speakers", [])
    print(f"Found {len(_speakers)} speakers")

    _segs = _result.get("speaking_segments", [])
    print(f"Speaking segments: {len(_segs)}")
    for seg in _segs:
        print(f"  {seg['start']:.1f}s - {seg['end']:.1f}s: Speaker {seg['speaker_id']}")
