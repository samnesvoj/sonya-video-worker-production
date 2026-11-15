"""
Who is who across camera cuts (fill_9x16 crop): OpenCV SFace face embeddings (opencv_zoo
face_recognition_sface_2021dec_int8.onnx, Apache-2.0 — models/LICENSE-sface, sha256-pinned).

Use: when the speaker model is unsure at the start of a shot (tiny faces in a wide two-shot), the window holds
the person who was on screen right before the cut — the one whose face matches best. VSRAP wide shot (faces
60 px): same person 0.55–0.89, the other one 0.14–0.37.

similarities(video, start, ref_boxes, cand_boxes_list) → [cosine per candidate] | None. Never raises.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL = Path(__file__).resolve().parent / "models" / "face_recognition_sface_2021dec_int8.onnx"
MODEL_SHA256 = "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a"
SAMPLES = 5                      # face samples per person
_lock = threading.Lock()
_ok: Optional[bool] = None


def available() -> bool:
    global _ok
    with _lock:
        if _ok is None:
            try:
                import cv2
                _ok = (MODEL.is_file() and hasattr(cv2, "FaceRecognizerSF")
                       and hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_SHA256)
            except Exception:            # noqa: BLE001
                _ok = False
            if not _ok:
                logger.warning("[crop] face id model unavailable — no cross-cut identity")
        return bool(_ok)


def _feature(cap: Any, rec: Any, start: float, t: float, box: Sequence[float]) -> Optional[np.ndarray]:
    import cv2
    from .clip_analysis import MODEL_PATH
    cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000.0)
    ok, fr = cap.read()
    if not ok:
        return None
    x, y, w, h = (int(v) for v in box[:4])
    pad = int(0.6 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    crop = fr[y0:y + h + pad, x0:x + w + pad]
    if crop.size == 0:
        return None
    k = max(1.0, 160.0 / max(w, 1))
    crop = cv2.resize(crop, None, fx=k, fy=k)
    det = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (crop.shape[1], crop.shape[0]), 0.5, 0.3, 5)
    _, f = det.detect(crop)
    if f is None or not len(f):
        return None
    f = min(f, key=lambda r: abs(r[0] + r[2] / 2 - crop.shape[1] / 2) + abs(r[1] + r[3] / 2 - crop.shape[0] / 2))
    v = rec.feature(rec.alignCrop(crop, f)).flatten().astype(np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _person(cap: Any, rec: Any, start: float, boxes: Sequence[Tuple[float, Sequence[float]]]) -> Optional[np.ndarray]:
    bs = sorted(boxes, key=lambda x: x[0])
    if not bs:
        return None
    idx = np.unique(np.linspace(0, len(bs) - 1, min(SAMPLES, len(bs))).astype(int))
    fs = [f for f in (_feature(cap, rec, start, bs[i][0], bs[i][1]) for i in idx) if f is not None]
    if not fs:
        return None
    m = np.mean(fs, axis=0)
    return m / max(np.linalg.norm(m), 1e-9)


def similarities(video: str, start: float, ref: Sequence[Tuple[float, Sequence[float]]],
                 cands: List[Sequence[Tuple[float, Sequence[float]]]]) -> Optional[List[float]]:
    if not available():
        return None
    try:
        import cv2
        rec = cv2.FaceRecognizerSF.create(str(MODEL), "")
        cap = cv2.VideoCapture(video)
        try:
            r = _person(cap, rec, start, ref)
            if r is None:
                return None
            out = []
            for c in cands:
                v = _person(cap, rec, start, c)
                out.append(float(np.dot(r, v)) if v is not None else -1.0)
            return out
        finally:
            cap.release()
    except Exception as exc:              # noqa: BLE001 — optional cue
        logger.warning("[crop] face id failed: %s", str(exc)[:200])
        return None
