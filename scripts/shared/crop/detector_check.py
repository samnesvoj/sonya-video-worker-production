"""
detector_check.py
=================
Preflight for the face detector SmartCropper / analyze_faces use: the
OpenCV Haar frontal-face cascade shipped in the opencv wheels.

OpenCV 5 dropped the Haar cascades; an unpinned install silently turns
every face crop into a center crop. This check makes that explicit:

    face_detector_preflight() -> {"ok", "opencv_version", "cascade", "reason"}

Never raises, needs no GUI/display, does not download anything.
"""
from __future__ import annotations

import os
from typing import Any, Dict

EXPECTED_OPENCV_MAJOR = 4
CASCADE_NAME = "haarcascade_frontalface_default.xml"


def face_detector_preflight() -> Dict[str, Any]:
    res: Dict[str, Any] = {"ok": False, "opencv_version": None, "cascade": None, "reason": None,
                           "expected_major": EXPECTED_OPENCV_MAJOR}
    try:
        import cv2  # noqa: WPS433 — optional dependency, checked here on purpose
    except Exception as e:  # ImportError, or a broken native lib
        res["reason"] = f"cv2 import failed: {type(e).__name__}: {str(e)[:200]}"
        return res
    version = getattr(cv2, "__version__", "") or ""
    res["opencv_version"] = version
    try:
        major = int(version.split(".")[0])
    except ValueError:
        major = -1
    if major != EXPECTED_OPENCV_MAJOR:
        res["reason"] = f"OpenCV {version} — expected {EXPECTED_OPENCV_MAJOR}.x (Haar cascades)"
        return res
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not data_dir:
        res["reason"] = "cv2.data.haarcascades missing"
        return res
    path = os.path.join(data_dir, CASCADE_NAME)
    res["cascade"] = path
    if not os.path.isfile(path) or not os.access(path, os.R_OK) or os.path.getsize(path) == 0:
        res["reason"] = f"cascade not found or unreadable: {CASCADE_NAME}"
        return res
    try:
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            res["reason"] = "cascade failed to load (empty classifier)"
            return res
    except Exception as e:
        res["reason"] = f"CascadeClassifier failed: {type(e).__name__}"
        return res
    res["ok"] = True
    return res
