"""
yolo_common.py
==============
Shared YOLO detection helpers for SONYA production modes.
Models are downloaded at runtime via model_downloader.py — NOT bundled.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def detect_common(video_path: str, model_path: str, conf: float = 0.3) -> List[Dict[str, Any]]:
    """Run YOLO object detection on video frames. Returns list of frame detections."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("[yolo_common] ultralytics not installed — returning empty detections")
        return []

    model = YOLO(model_path)
    results = model(video_path, conf=conf, stream=True, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            detections.append({
                "frame": int(r.path) if r.path and r.path.isdigit() else None,
                "boxes": boxes.xyxy.tolist(),
                "confs": boxes.conf.tolist(),
                "classes": boxes.cls.tolist(),
            })
    return detections


def detect_pose(video_path: str, model_path: str, conf: float = 0.3) -> List[Dict[str, Any]]:
    """Run YOLO pose estimation on video frames. Returns list of frame poses."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("[yolo_common] ultralytics not installed — returning empty poses")
        return []

    model = YOLO(model_path)
    results = model(video_path, conf=conf, stream=True, verbose=False)
    poses = []
    for r in results:
        kp = r.keypoints
        if kp is not None:
            poses.append({
                "keypoints": kp.xy.tolist(),
                "confs": kp.conf.tolist() if kp.conf is not None else [],
            })
    return poses
