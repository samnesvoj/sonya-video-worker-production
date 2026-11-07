"""
Per-clip vision analysis for the 9:16 crop: shot cuts + faces, only inside
the clip's time range (a 35 s clip costs ~2 s, independent of source length).

  - cuts: every decoded frame → HSV histogram + gray thumbnail; a cut is a frame whose
    change vs the previous frame is high, a local peak, and well
    above the clip's typical frame-to-frame change.
  - faces: OpenCV YuNet (DNN, handles 3/4 and partial profiles far better
    than the old Haar cascade) at `face_fps`, boxes in source pixels + score.

Model: scripts/shared/crop/models/face_detection_yunet_2023mar.onnx
(opencv_zoo, MIT, see LICENSE-yunet), pinned by sha256.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

CLIP_ANALYSIS_VERSION = "clip-yunet-cuts-text-v4"   # v4: cheaper per-frame signature
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECT_WIDTH = 640          # faces are detected on frames downscaled to this width
SCORE_MIN = 0.6
CUT_MIN_DIST = 0.35         # Bhattacharyya distance of HSV histograms
CUT_PIX = 30.0             # mean abs gray change (0..255) of the 64x36 thumbnail
CUT_REL = 4.0               # … and this many times the clip's median frame change
MIN_SHOT_SEC = 0.4


def model_ok() -> Dict[str, Any]:
    if not MODEL_PATH.is_file():
        return {"ok": False, "reason": f"model missing: {MODEL_PATH.name}"}
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        return {"ok": False, "reason": "model sha256 mismatch"}
    if not hasattr(cv2, "FaceDetectorYN"):
        return {"ok": False, "reason": f"OpenCV {cv2.__version__} has no FaceDetectorYN"}
    return {"ok": True, "detector": "opencv-yunet-2023mar", "opencv": cv2.__version__}


def _sig(frame: np.ndarray):
    """(HSV histogram, 64x36 gray thumbnail) — colour and layout signals of a frame."""
    # strided view first (free), then a small area resize: ~10× cheaper than resizing 1080p directly
    step = max(1, frame.shape[1] // 384)
    small = cv2.resize(frame[::step, ::step], (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    g = cv2.cvtColor(cv2.resize(small, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.normalize(h, h).flatten(), g.astype(np.float32)


def _dist(a, b) -> float:
    """One change score per frame pair: ≥1.0 looks like a cut. Colour alone misses
    cuts between two similar-looking shots of one room, so layout counts too."""
    dh = float(cv2.compareHist(a[0], b[0], cv2.HISTCMP_BHATTACHARYYA))
    dp = float(np.abs(a[1] - b[1]).mean())
    return max(dh / CUT_MIN_DIST, min(dh / 0.2, dp / CUT_PIX))


def _mouth_motion(gray: np.ndarray, prev: Optional[np.ndarray], f, k: float = 1.0) -> Optional[float]:
    """Change in the mouth region since the previous face sample, relative to the
    change of the whole face (so a moving head is not mistaken for talking).
    YuNet landmarks 10..13 = mouth corners (x, y)."""
    if prev is None or prev.shape != gray.shape:
        return None
    H, W = gray.shape
    x, y, w, h = (float(v) * k for v in f[:4])
    mx = (float(f[10]) + float(f[12])) / 2.0 * k
    my = (float(f[11]) + float(f[13])) / 2.0 * k
    mw = max(6.0, abs(float(f[12]) - float(f[10])) * k * 1.4)
    mh = max(6.0, 0.28 * h)

    def region(cx, cy, rw, rh):
        x0, x1 = int(max(0, cx - rw / 2)), int(min(W, cx + rw / 2))
        y0, y1 = int(max(0, cy - rh / 2)), int(min(H, cy + rh / 2))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return float(np.abs(gray[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean())
    m = region(mx, my, mw, mh)
    face = region(x + w / 2, y + h * 0.35, w * 0.8, h * 0.4)          # eyes/forehead: head motion only
    if m is None or face is None:
        return None
    return round(max(0.0, m - face), 2)


TEXT_EVERY = 5               # text check on every 5th face sample (~2 fps)


def text_boxes(frame: np.ndarray) -> List[List[int]]:
    """Burned-in text lines (lower thirds, captions, phone screens) in source pixels.
    Gradient → Otsu → horizontal closing → long thin dense high-contrast components.
    Measured on a 2.5 h interview: titles found, no false boxes in 150 random frames."""
    H, W = frame.shape[:2]
    k = 960.0 / W
    small = cv2.resize(frame, (960, max(1, int(round(H * k)))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if w >= 80 and 10 < h < 45 and w / h > 3 and a / (w * h) > 0.5 and g[y:y + h, x:x + w].std() >= 50:
            out.append([int(x / k), int(y / k), int(w / k), int(h / k)])
    return out


def detect_cuts(times: List[float], dists: List[float]) -> List[float]:
    """Cut times (start of the new shot) from per-frame histogram distances."""
    if len(dists) < 3:
        return []
    base = float(np.median(dists)) or 1e-3
    cuts: List[float] = []
    for i, d in enumerate(dists):
        if d < 1.0 or d < CUT_REL * base:
            continue
        if i > 0 and dists[i - 1] > d or i + 1 < len(dists) and dists[i + 1] >= d:
            continue                                   # not the local peak (fades/flashes)
        if cuts and times[i] - cuts[-1] < MIN_SHOT_SEC:
            continue
        cuts.append(round(times[i], 3))
    return cuts


def _frames(video: str, start: float, duration: float, W: int, H: int, fps: float):
    """(t relative to start, BGR frame, scale vs source) from OpenCV (hardware decode
    on macOS: ~0.8 s for 40 s of 1080p — decoding is not the bottleneck)."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - 0.5) * 1000.0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - start, frame, 1.0
    finally:
        cap.release()


def analyze_clip(video: str, start: float, duration: float, *, face_fps: float = 10.0,
                 check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """{"meta", "cuts": [t], "samples": [{"t", "faces": [[x,y,w,h,score], …]}]}; t relative to `start`."""
    st = model_ok()
    if not st["ok"]:
        raise RuntimeError(st["reason"])
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    times: List[float] = []
    dists: List[float] = []
    samples: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    prev = None
    prev_gray = None            # previous face-sample frame for mouth motion
    next_face_t = 0.0
    det = None
    for t, frame, fk in _frames(video, start, duration, W, H, fps):
        # frame is the source scaled by fk (analysis size); boxes go back to source pixels
        if det is None:
            fh, fw = frame.shape[:2]
            scale = min(1.0, DETECT_WIDTH / max(fw, 1))
            dw, dh = int(round(fw * scale)), int(round(fh * scale))
            det = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (dw, dh), SCORE_MIN, 0.3, 50)
        if t < -0.05:
            prev = _sig(frame)
            continue
        if t > duration + 0.05:
            break
        if check is not None:
            check()
        h = _sig(frame)
        if prev is not None:
            times.append(t)
            dists.append(_dist(prev, h))
        prev = h
        if t >= next_face_t - 1e-6:
            small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
            _, faces = det.detect(small)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)   # analysis res: wide-shot mouths
            back = 1.0 / (scale * fk)
            boxes = []
            for f in (faces if faces is not None else []):
                x, y, w, hh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])
                boxes.append([int(x * back), int(y * back), int(w * back), int(hh * back), round(score, 3),
                              _mouth_motion(gray, prev_gray, f, 1.0 / scale)])
            prev_gray = gray
            boxes.sort(key=lambda b: (-b[2] * b[3], b[0], b[1]))
            samples.append({"t": round(t, 3), "faces": boxes})
            if (len(samples) - 1) % TEXT_EVERY == 0:
                texts.append({"t": round(t, 3), "boxes": [[int(v / fk) for v in b] for b in text_boxes(frame)]})
            next_face_t += 1.0 / face_fps
    return {"meta": {"version": CLIP_ANALYSIS_VERSION, "detector": st["detector"], "opencv": st["opencv"],
                     "width": W, "height": H, "fps": fps, "start": start, "duration": duration,
                     "face_fps": face_fps, "frames": len(times) + 1},
            "cuts": detect_cuts(times, dists), "samples": samples, "text": texts}
