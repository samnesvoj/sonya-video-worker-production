"""
webcam_detector.py
==================
Webcam/person bounding box detector for SONYA streamer mode.
Model: models/common/webcam_detector.pt (downloaded via model_downloader).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class WebcamDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except ImportError:
                logger.warning("[webcam_detector] ultralytics not installed")
                self._model = False

    def detect(self, video_path: str, conf: float = 0.4) -> List[Dict[str, Any]]:
        """Returns list of dicts with frame index and bounding boxes."""
        self._load()
        if not self._model:
            return []
        results = self._model(video_path, conf=conf, stream=True, verbose=False)
        output = []
        for i, r in enumerate(results):
            if r.boxes is not None:
                output.append({
                    "frame": i,
                    "boxes": r.boxes.xyxy.tolist(),
                    "confs": r.boxes.conf.tolist(),
                })
        return output
