"""
placement.py — where a card may go on the finished frame, or nowhere.

Uses what SONYA already has: scripts/shared/crop/clip_analysis (YuNet faces, burned-in text boxes, scene cuts) run
on the rendered clip, so coordinates are output pixels in any aspect. A candidate rectangle is refused if it touches
a face, burned-in text, the subtitle band or the mobile UI zones, or sits on a visually busy area (a board, a slide).
Nothing safe → the overlay is skipped with a reason. Better no card than a card over the explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]        # x, y, w, h

BUSY_MAX = 0.085                # share of edge pixels above which an area counts as "something is shown here"
CUT_SNAP_SEC = 0.5


@dataclass(frozen=True)
class Frame:
    width: int
    height: int

    @property
    def aspect(self) -> str:
        r = self.width / self.height
        return "9:16" if r < 0.75 else ("1:1" if r < 1.3 else "16:9")

    @property
    def short(self) -> int:
        return min(self.width, self.height)


def safe_margins(f: Frame) -> Dict[str, int]:
    """Mobile platform UI in 9:16 (top bar, right-hand icons, caption block at the bottom); title safe otherwise."""
    if f.aspect == "9:16":
        return {"top": int(f.height * 0.115), "bottom": int(f.height * 0.2), "left": int(f.width * 0.037), "right": int(f.width * 0.037),
                "right_low": int(f.width * 0.13)}
    m = int(f.short * 0.045)
    return {"top": m, "bottom": m, "left": m, "right": m, "right_low": m}


def card_width(f: Frame) -> int:
    return int(f.width * {"9:16": 0.86, "1:1": 0.62, "16:9": 0.36}[f.aspect])


def text_scale(f: Frame) -> float:
    """A phone shows a 1080-wide frame about 2.5 times smaller than a laptop shows a 1920-wide one."""
    return {"9:16": 1.32, "1:1": 1.1, "16:9": 1.0}[f.aspect]


def max_card_height(f: Frame) -> int:
    return int(f.height * {"9:16": 0.34, "1:1": 0.5, "16:9": 0.62}[f.aspect])


def candidates(f: Frame, w: int, h: int, subtitle_top: int) -> Dict[str, Rect]:
    m = safe_margins(f)
    W, H = f.width, f.height
    low_y = subtitle_top - h - int(H * 0.012)
    if f.aspect == "9:16":
        cx = (W - w) // 2
        out = {"top": (cx, m["top"], w, h)}
        out["upper"] = (cx, m["top"] + int(H * 0.09), w, h)
        low_x = min(cx, W - m["right_low"] - w)
        out["lower"] = (max(m["left"], low_x), low_y, w, h)
        out["middle"] = (cx, (H - h) // 2, w, h)
        return out
    return {"top_right": (W - m["right"] - w, m["top"], w, h), "top_left": (m["left"], m["top"], w, h),
            "bottom_right": (W - m["right"] - w, low_y, w, h), "bottom_left": (m["left"], low_y, w, h)}


def inside_frame(f: Frame, r: Rect) -> bool:
    m = safe_margins(f)
    x, y, w, h = r
    return x >= m["left"] - 1 and y >= m["top"] - 1 and x + w <= f.width - m["right"] + 1 and y + h <= f.height - m["bottom"] / 2


def _touch(a: Rect, b: Sequence[float], pad: int = 0) -> bool:
    return not (a[0] + a[2] + pad <= b[0] or b[0] + b[2] + pad <= a[0] or a[1] + a[3] + pad <= b[1] or b[1] + b[3] + pad <= a[1])


def face_rect(box: Sequence[float]) -> Tuple[float, float, float, float]:
    """YuNet box is the bare face: add hair above and chin + a bit of neck below."""
    x, y, w, h = box[:4]
    return x - w * 0.25, y - h * 0.45, w * 1.5, h * 1.8


def occupied(analysis: Dict[str, Any], t0: float, t1: float) -> Tuple[List[Sequence[float]], List[Sequence[float]]]:
    faces, texts = [], []
    for s in analysis.get("samples", []):
        if t0 - 0.3 <= s["t"] <= t1 + 0.3:
            faces += [face_rect(b) for b in s.get("faces", [])]
    for s in analysis.get("text", []):
        if t0 - 0.5 <= s["t"] <= t1 + 0.5:
            texts += s.get("boxes", [])
    return faces, texts


def _caption_box(box: Sequence[float], width: int) -> bool:
    """Captions are wide and centred; a label in a corner or part of a drawing is not a caption."""
    x, _, w, _ = box
    return w >= width * 0.2 and abs(x + w / 2 - width / 2) <= width * 0.2


def burned_caption_share(video: str, band_top: int, band_bottom: int, duration: float, samples: int = 20) -> float:
    """Share of sampled frames with burned-in captions in a horizontal band. Text is searched in the band alone:
    with the threshold computed over the whole frame, a caption plate on a detailed background is mostly lost."""
    from scripts.shared.crop.clip_analysis import text_boxes
    cap = cv2.VideoCapture(video)
    hits = total = 0
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, (i + 0.5) / samples * duration * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        total += 1
        band = frame[max(0, band_top):band_bottom]
        hits += any(_caption_box(b, frame.shape[1]) for b in text_boxes(band))
    cap.release()
    return hits / total if total else 0.0


def edge_density_maps(video: str, t0: float, t1: float, step: float = 1.0) -> Optional[np.ndarray]:
    """Mean Canny map over the window, at full frame size."""
    cap = cv2.VideoCapture(video)
    acc, n = None, 0
    t = t0
    while t <= t1 + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        e = (cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 60, 160) > 0).astype(np.float32)
        acc = e if acc is None else acc + e
        n += 1
        t += step
    cap.release()
    return acc / n if n else None


def choose(f: Frame, w: int, h: int, subtitle_top: int, analysis: Dict[str, Any], t0: float, t1: float, *,
           edges: Optional[np.ndarray], preferred: str = "auto", busy_max: float = BUSY_MAX) -> Tuple[Optional[Rect], Dict[str, Any]]:
    """Calmest allowed candidate. Returns (rect or None, report of why each candidate was refused)."""
    faces, texts = occupied(analysis, t0, t1)
    cands = candidates(f, w, h, subtitle_top)
    order = [preferred] if preferred in cands else list(cands)
    report: Dict[str, Any] = {"faces": len(faces), "text_boxes": len(texts), "refused": {}}
    allowed: List[Tuple[float, str, Rect]] = []
    for name in order:
        r = cands[name]
        if not inside_frame(f, r) or r[1] + r[3] > subtitle_top:
            report["refused"][name] = "outside_safe_zone"
            continue
        if any(_touch(r, b, int(f.short * 0.012)) for b in faces):
            report["refused"][name] = "face"
            continue
        if any(_touch(r, b, int(f.short * 0.008)) for b in texts):
            report["refused"][name] = "burned_text"
            continue
        busy = float(edges[r[1]:r[1] + r[3], r[0]:r[0] + r[2]].mean()) if edges is not None else 0.0
        if busy > busy_max:
            report["refused"][name] = f"busy:{busy:.3f}"
            continue
        allowed.append((busy, name, r))
    if not allowed:
        return None, report
    allowed.sort(key=lambda a: (round(a[0], 2), order.index(a[1])))
    report["chosen"], report["busy"] = allowed[0][1], round(allowed[0][0], 3)
    return allowed[0][2], report


def snap_to_cuts(start: float, end: float, cuts: Sequence[float]) -> Tuple[float, float]:
    """A card should not appear a moment before a cut or vanish a moment after one."""
    for c in cuts:
        if start < c < start + CUT_SNAP_SEC:
            start = float(c)
        if end - CUT_SNAP_SEC < c < end:
            end = float(c)
    return start, end
