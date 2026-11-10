"""
stream_layout.py
================
Where things are on a stream frame, for the game layout «веб-камера сверху, игра снизу» (modes/streamer v2):

  * the WEBCAM rectangle, exactly by its borders — so the top half shows the camera only, never the chat,
    alerts or OBS frames next to it (owner, 2026-11-10: a Twitch chat right under the webcam got zoomed
    into the top half and «запорол видос»);
  * CHAT / overlay columns — static screen regions full of small text that keeps changing (Twitch / YouTube
    chat boxes, event lists) — kept out of both halves where possible.

Everything is measured on ~24 frames spread over the clip, at 960 px width:

  webcam  YuNet finds the streamer's face (a small face that does not move: modes/streamer game_track); from
          the face we walk outwards to the nearest STRAIGHT LINE that holds in every frame: an overlay border, or
          the seam between the camera picture and the game (two independent pictures: the gradient across the
          seam is high in every frame, while inside a picture it comes and goes). The frame edge also counts.
          Optionally the webcam_detector.pt box (YOLO, `detector_box`) is snapped to those lines the same way.
  chat    per 24x24 cell: how often it holds small text lines (thin, dense, high-contrast components) and how
          often that text changes between frames; a connected block of «text-and-changing» cells that is tall
          (≥ 12 % of the frame) is a chat / event column. Static HUD text (money, HP, round timer) does not change
          and is ignored; the kill feed is small and does not form a tall block.

No model downloads: OpenCV only. ~0.3 s per clip on a Mac mini.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

VERSION = "stream-layout-v2"
WORK_W = 960                   # analysis width
CELL = 24                      # chat grid cell (analysis px)
PRESENT_MIN = 0.06             # mean share of the cell covered by small text lines
CHANGE_MIN = 0.04              # mean share of the cell whose text differs between two looks


# ── frames ──────────────────────────────────────────────────────────────────

def sample_frames(video: str, start: float, duration: float, n: int = 24) -> Tuple[List[np.ndarray], float]:
    """n gray frames spread over [start, start+duration], resized to WORK_W. -> (frames, scale source→work)."""
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or WORK_W
    k = WORK_W / float(W)
    out: List[np.ndarray] = []
    for i in range(n):
        t = start + duration * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, (WORK_W, int(round(g.shape[0] * k))), interpolation=cv2.INTER_AREA))
    cap.release()
    return out, k


# ── persistent straight lines ───────────────────────────────────────────────

def line_maps(frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(V, Hz): for every pixel, the share of frames where a vertical (V) / horizontal (Hz) edge is there.
    An overlay border or a seam between two pictures is an edge in (almost) every frame."""
    v = np.zeros(frames[0].shape, np.float32)
    h = np.zeros(frames[0].shape, np.float32)
    for g in frames:
        f = g.astype(np.float32)
        gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
        v += (gx > 40.0) & (gx > 1.5 * gy)
        h += (gy > 40.0) & (gy > 1.5 * gx)
    return v / len(frames), h / len(frames)


def _profile(m: np.ndarray, axis_pos: int, a: int, b: int, vertical: bool) -> float:
    """Share of the line (column axis_pos rows a..b, or row axis_pos cols a..b) that is a persistent edge
    (±2 px tolerance: a border may sit between two pixels, a scaled overlay may wobble by one)."""
    if b - a < 4:
        return 0.0
    if vertical:
        x0, x1 = max(0, axis_pos - 2), min(m.shape[1], axis_pos + 3)
        seg = m[a:b, x0:x1].max(axis=1)
    else:
        y0, y1 = max(0, axis_pos - 2), min(m.shape[0], axis_pos + 3)
        seg = m[y0:y1, a:b].max(axis=0)
    return float((seg >= 0.7).mean())


def segments(m: np.ndarray, vertical: bool, min_len: int, thr: float = 0.6, gap: int = 4) -> List[Tuple[int, int, int]]:
    """Long persistent straight segments: (pos, a, b) = column x rows a..b (vertical) or row y cols a..b.
    Neighbouring parallel duplicates (a 2-3 px wide border) are merged into the longest one."""
    B = m >= thr
    if not vertical:
        B = B.T
    n_pos, n_len = B.shape[1], B.shape[0]
    raw: List[Tuple[int, int, int]] = []
    for p in range(n_pos):
        col = B[:, max(0, p - 1):p + 2].any(axis=1)
        idx = np.flatnonzero(col)
        if len(idx) < min_len:
            continue
        brk = np.flatnonzero(np.diff(idx) > gap)
        starts = np.concatenate([[idx[0]], idx[brk + 1]])
        ends = np.concatenate([idx[brk], [idx[-1]]])
        for a, b in zip(starts, ends):
            if b - a >= min_len:
                raw.append((p, int(a), int(b)))
    raw.sort(key=lambda s: (-(s[2] - s[1]), s[0]))
    out: List[List[int]] = []
    for s in raw:
        for o in out:            # same line (±3 px), overlapping or with a short break (a border behind an alert)
            if abs(s[0] - o[0]) <= 3 and min(s[2], o[2]) - max(s[1], o[1]) > -25:
                o[1], o[2] = min(o[1], s[1]), max(o[2], s[2])
                break
        else:
            out.append(list(s))
    return [tuple(o) for o in out]


def _span_cover(seg, a: int, b: int) -> float:
    """Share of [a, b] covered by the segment (pos, s0, s1) — ±8 px tolerance at both ends."""
    return max(0.0, min(seg[2] + 8, b) - max(seg[1] - 8, a)) / float(max(1, b - a))


def find_rects(maps, *, min_side: int = 40, min_cover: float = 0.55) -> List[Dict[str, Any]]:
    """Rectangles whose 4 sides are persistent lines (or the frame edge), work px. Each:
    {"rect": [x, y, w, h], "cover": min side coverage, "edges": number of sides on the frame edge}."""
    V, Hz = maps
    Hh, Ww = V.shape
    vs = segments(V, True, min_side) + [(0, 0, Hh - 1), (Ww - 1, 0, Hh - 1)]
    hs = segments(Hz, False, min_side) + [(0, 0, Ww - 1), (Hh - 1, 0, Ww - 1)]
    rects: List[Dict[str, Any]] = []
    seen = set()
    for i, l in enumerate(vs):
        for r in vs:
            if r[0] - l[0] < min_side:
                continue
            y0, y1 = max(l[1], r[1]), min(l[2], r[2])
            if y1 - y0 < min_side:
                continue
            for t in hs:
                if not (abs(t[0] - y0) <= 8 or (t[0] > y0 and t[0] < y1 - min_side)):
                    continue
                if _span_cover(t, l[0], r[0]) < 0.75:   # a chat icon / alert may break the line's end
                    continue
                for b in hs:
                    if b[0] - t[0] < min_side or b[0] > y1 + 8:
                        continue
                    if _span_cover(b, l[0], r[0]) < 0.75:   # a chat icon / alert may break the line's end
                        continue
                    key = (l[0] // 4, t[0] // 4, r[0] // 4, b[0] // 4)
                    if key in seen:
                        continue
                    seen.add(key)
                    cov = min(_profile(V, l[0], t[0], b[0], True) if l[0] > 0 else 1.0,
                              _profile(V, r[0], t[0], b[0], True) if r[0] < Ww - 1 else 1.0,
                              _profile(Hz, t[0], l[0], r[0], False) if t[0] > 0 else 1.0,
                              _profile(Hz, b[0], l[0], r[0], False) if b[0] < Hh - 1 else 1.0)
                    if cov < min_cover:
                        continue
                    edges = (l[0] == 0) + (r[0] == Ww - 1) + (t[0] == 0) + (b[0] == Hh - 1)
                    rects.append({"rect": [l[0], t[0], r[0] - l[0], b[0] - t[0]], "cover": round(cov, 3),
                                  "edges": int(edges)})
    return rects


def _motion(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Share of frame pairs where the pixel changed (> 12 grey levels)."""
    st = np.stack(frames).astype(np.int16)
    return (np.abs(np.diff(st, axis=0)) > 12).mean(axis=0)


def webcam_rect(frames: Sequence[np.ndarray], face: Optional[Sequence[float]], *, maps=None,
                seed: Optional[Sequence[float]] = None, chat: Sequence[Sequence[int]] = ()) -> Optional[Dict[str, Any]]:
    """The webcam rectangle (work px) among the line rectangles of the frame.
    With a webcam face: the smallest rectangle that holds the face with room around it.
    Without one (face paint, a mask, looking away): the most camera-like rectangle — a picture of its own
    (2–30 % of the frame, 1.1–2.4 wide), not chat, mostly still (a room) with some motion (the person).
    `seed`: a rough box from webcam_detector: the rectangle must cover most of it.
    -> {"rect", "cover", "edges", "ok", "why", "by"} or None."""
    if not frames:
        return None
    maps = maps if maps is not None else line_maps(frames)
    Hh, Ww = maps[0].shape
    rects = find_rects(maps, min_side=max(30, Hh // 14))
    mot = None
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for c in rects:
        x, y, w, h = c["rect"]
        area, ar = w * h / float(Ww * Hh), w / float(h)
        if not (0.015 <= area <= 0.4 and 0.9 <= ar <= 2.6) or c["edges"] > 2:
            continue
        if c["cover"] < (0.55 if face is not None else 0.8):   # a face inside makes a weak border believable
            continue
        if any(_overlap(c["rect"], z) > 0.3 * w * h for z in chat):
            continue
        if face is not None:
            fx, fy, fw, fh = face
            cx, cy = fx + fw / 2, fy + fh / 2
            if not (x < cx < x + w and y < cy < y + h) or w < 1.8 * fw or h < 1.4 * fh:
                continue
            key = area                                   # the tightest frame around the face
            by = "face"
        elif seed is not None:
            ov = _overlap(c["rect"], seed) / float(seed[2] * seed[3])
            if ov < 0.6:
                continue
            key = -ov + 0.5 * area
            by = "detector"
        else:
            if mot is None:
                mot = _motion(frames)
            m = mot[y:y + h, x:x + w]
            still, moving = float((m < 0.05).mean()), float((m > 0.25).mean())
            if still < 0.35 or moving < 0.02:            # a game view moves everywhere; a static panel never
                continue
            key = -(still * min(moving, 0.3) * 4 + c["cover"] - 0.3 * c["edges"] + min(area, 0.12) * 3)
            by = "lines"
        if best is None or key < best[0]:
            best = (key, dict(c, by=by))
    if best is None:
        return None
    out = best[1]
    out.update(ok=True, why=None)
    return out


# ── chat / changing small text ──────────────────────────────────────────────

def text_mask(g: np.ndarray) -> np.ndarray:
    """Small text lines (chat-size: 6–22 px high at 960 px width) as a binary mask."""
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = (grad > 60).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        keep[i] = 5 <= h <= 22 and w >= 2.2 * h and a / float(w * h) > 0.35
    return keep[lab]


def chat_zones(frames: Sequence[np.ndarray], *, min_height_frac: float = 0.12,
               exclude: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Chat / event columns (work px boxes) + the per-cell maps (text share, change share).
    `exclude` (work px box): skip it (the webcam: a face talking is not chat)."""
    if len(frames) < 4:
        return {"zones": [], "cells": None}
    Hh, Ww = frames[0].shape
    gh, gw = Hh // CELL, Ww // CELL
    masks = [text_mask(g)[: gh * CELL, : gw * CELL] for g in frames]
    def cells(m):
        return m.reshape(gh, CELL, gw, CELL).mean(axis=(1, 3))
    # average text coverage of the cell, and how much of it differs between consecutive looks
    present = np.mean([cells(m) for m in masks], axis=0)
    change = np.mean([cells(np.logical_xor(a, b)) for a, b in zip(masks, masks[1:])], axis=0)
    hot = (present >= PRESENT_MIN) & (change >= CHANGE_MIN)
    if exclude is not None:
        x, y, w, h = exclude
        hot[max(0, y // CELL): (y + h) // CELL + 1, max(0, x // CELL): (x + w) // CELL + 1] = False
    # close 1-cell gaps between chat lines
    hot8 = cv2.morphologyEx(hot.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot8, connectivity=8)
    zones = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h * CELL < min_height_frac * Hh or a < 6:
            continue
        zones.append({"box": [int(x * CELL), int(y * CELL), int(w * CELL), int(h * CELL)],
                      "text": round(float(present[lab == i].mean()), 3),
                      "change": round(float(change[lab == i].mean()), 3)})
    return {"zones": zones, "cells": {"present": present, "change": change}}


# ── windows that avoid what must not be shown ───────────────────────────────

def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return w * h


def cam_window(rect: Sequence[int], face_cx: float, face_cy: float, aspect: float = 1080 / 960) -> List[int]:
    """Largest window of `aspect` INSIDE the webcam rectangle (nothing outside it can show),
    centred on the face where the rectangle allows. Source px, even numbers."""
    x, y, w, h = (float(v) for v in rect)
    if w / h >= aspect:
        ch, cw = h, h * aspect
    else:
        cw, ch = w, w / aspect
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    cx = min(max(x, face_cx - cw / 2), x + w - cw)
    cy = min(max(y, face_cy - ch * 0.45), y + h - ch)
    return [int(np.ceil(cx / 2) * 2), int(np.ceil(cy / 2) * 2), cw, ch]


def game_window(W: int, H: int, avoid: Sequence[Sequence[float]], *, aspect: float = 1080 / 960,
                heights: Sequence[float] = (0.9, 0.8, 0.7, 0.6)) -> Tuple[List[int], float]:
    """Game crop (aspect 1080:960) that shows as little of `avoid` (chat columns, the webcam) as it can, nearest
    the centre (the crosshair stays in the middle 60 % of the window, both ways). Tallest window first; a smaller
    one (a light zoom) only when the taller ones cannot leave `avoid` out. -> (box, share of the box covered)."""
    best: Optional[Tuple[float, float, List[int]]] = None
    for hf in heights:
        gh = int(H * hf) // 2 * 2
        gw = int(gh * aspect) // 2 * 2
        if gw > W:
            gw = W // 2 * 2
            gh = int(gw / aspect) // 2 * 2
        y_lo = max(0, int(H / 2 - 0.8 * gh))
        y_hi = min(H - gh, int(H / 2 - 0.2 * gh))
        x_lo = max(0, int(W / 2 - 0.8 * gw))
        x_hi = min(W - gw, int(W / 2 - 0.2 * gw))
        ys = sorted(set([(H - gh) // 2] + list(range(y_lo, max(y_lo, y_hi) + 1, 8))))
        for gy in ys:
            for gx in range(x_lo, max(x_lo, x_hi) + 1, 8):
                box = [gx, gy, gw, gh]
                cov = sum(_overlap(box, a) for a in avoid) / float(gw * gh)
                off = abs(gx + gw / 2 - W / 2) / W + abs(gy + gh / 2 - H / 2) / H
                key = (round(cov, 3), off - 0.05 * hf)
                if best is None or key < best[:2]:
                    best = (key[0], key[1], [gx // 2 * 2, gy // 2 * 2, gw, gh])
        if best and best[0] <= 0.002:
            break
    assert best is not None
    return best[2], best[0]


def analyze_layout(video: str, start: float, duration: float, face_src: Optional[Sequence[float]],
                   W: int, H: int, *, detector_box: Optional[Sequence[float]] = None,
                   known_rects: Sequence[Sequence[int]] = (), n_frames: int = 24) -> Dict[str, Any]:
    """-> {"version", "webcam": {rect (src px), by, cover, edges} | None, "chat": [boxes src px], "frames": n}.
    `known_rects` (src px, the stream survey): when this clip finds no webcam of its own, a known rectangle whose
    border is visible in THIS clip's frames is taken (by="survey")."""
    frames, k = sample_frames(video, start, duration, n_frames)
    out: Dict[str, Any] = {"version": VERSION, "frames": len(frames), "webcam": None, "chat": []}
    if len(frames) < 4:
        return out
    maps = line_maps(frames)
    cz = chat_zones(frames)
    chat_w = [z["box"] for z in cz["zones"]]
    face = [v * k for v in face_src[:4]] if face_src is not None else None
    seed = [v * k for v in detector_box] if detector_box is not None else None
    cam = webcam_rect(frames, face, maps=maps, seed=seed, chat=chat_w)
    if cam is None:
        for kr in known_rects:
            x, y, w, h = (int(round(v * k)) for v in kr)
            V, Hz = maps
            cov = min(_profile(V, x, y, y + h, True) if x > 1 else 1.0,
                      _profile(V, x + w, y, y + h, True) if x + w < V.shape[1] - 2 else 1.0,
                      _profile(Hz, y, x, x + w, False) if y > 1 else 1.0,
                      _profile(Hz, y + h, x, x + w, False) if y + h < V.shape[0] - 2 else 1.0)
            if cov >= 0.5:
                cam = {"rect": [x, y, w, h], "cover": round(cov, 3), "by": "survey", "ok": True, "why": None,
                       "edges": int((x <= 1) + (y <= 1) + (x + w >= V.shape[1] - 2) + (y + h >= V.shape[0] - 2))}
                break
    if cam is not None:
        r = cam["rect"]
        # a talking face inside the webcam is not chat: drop chat cells that fall in the webcam
        chat_w = [z for z in chat_w if _overlap(z, r) < 0.5 * z[2] * z[3]]
        cam["rect"] = [int(round(v / k)) for v in r]
    out["webcam"] = cam
    out["chat"] = [[int(round(v / k)) for v in z] for z in chat_w]
    out["chat_detail"] = cz["zones"]
    return out


# ── whole-stream survey + the game track ────────────────────────────────────

def webcam_face(analysis: Optional[Dict[str, Any]], W: int, H: int, *, min_presence: float = 0.25,
                min_score: float = 0.6) -> Optional[List[float]]:
    """The streamer's webcam face in a per-clip analysis (clip_analysis samples): a small face (< 22 % of the width)
    that holds its place and size. -> median [x, y, w, h] (source px) or None. Looser than game_track (score 0.6,
    25 % of samples): a face in paint / a mask scores 0.6–0.75 (Buster «Халк» stream); steadiness still guards."""
    if not analysis or not analysis.get("samples"):
        return None
    boxes = []
    for s in analysis["samples"]:
        small = [f for f in s.get("faces") or [] if f[4] >= min_score and f[2] < 0.22 * W]
        if small:
            boxes.append(max(small, key=lambda b: b[2] * b[3])[:4])
    if len(boxes) < min_presence * len(analysis["samples"]):
        return None
    a = np.array(boxes, dtype=float)
    c = np.stack([a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2, a[:, 2]], axis=1)
    q = np.percentile(c, 90, axis=0) - np.percentile(c, 10, axis=0)
    med = np.median(c, axis=0)
    if q[0] / W > 0.08 or q[1] / H > 0.08 or q[2] > 0.5 * med[2]:
        return None
    return [float(v) for v in np.median(a, axis=0)]


def _faces_quick(video: str, t: float, dur: float, n: int = 8) -> Optional[List[float]]:
    """Webcam face from n YuNet looks over [t, t+dur] (survey only; clips use the 10 fps analysis)."""
    from .clip_analysis import MODEL_PATH
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not W or not H:
        return None
    k = 640.0 / W
    det = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (640, int(H * k)), 0.6, 0.3, 50)
    samples = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + dur * (i + 0.5) / n) * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        _, f = det.detect(cv2.resize(fr, (640, int(H * k)), interpolation=cv2.INTER_AREA))
        samples.append({"faces": [[float(v) / k for v in x[:4]] + [float(x[14])] for x in (f if f is not None else [])]})
    cap.release()
    return webcam_face({"samples": samples}, W, H)


def survey(video: str, duration: float, *, points: int = 8, window: float = 20.0,
           skip: float = 120.0) -> Dict[str, Any]:
    """Webcam rectangles seen across the whole stream (OBS scenes rarely move the camera): `points` windows
    spread over the VOD. -> {"version", "rects": [{"rect" src px, "seen": n}], "points"}. Used by the clips whose
    own frames do not show the border (a blue menu behind a borderless camera, a scene transition)."""
    found: List[Dict[str, Any]] = []
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    span = max(0.0, duration - skip - window)
    for i in range(points):
        t = skip + span * (i + 0.5) / points
        face = _faces_quick(video, t, window)
        lay = analyze_layout(video, t, window, face, W, H, n_frames=16)
        cam = lay.get("webcam")
        if not cam:
            continue
        r = cam["rect"]
        for f in found:
            if _overlap(f["rect"], r) > 0.85 * max(f["rect"][2] * f["rect"][3], r[2] * r[3]):
                f["seen"] += 1
                break
        else:
            found.append({"rect": r, "seen": 1, "by": cam.get("by")})
    found.sort(key=lambda f: -f["seen"])
    return {"version": VERSION, "rects": found, "points": points, "width": W, "height": H}


def fallback_cam(face: Sequence[float], W: int, H: int, avoid: Sequence[Sequence[float]],
                 aspect: float = 1080 / 960) -> List[int]:
    """No webcam border anywhere: a window around the face (≈ 4 face widths, like game-v1), narrowed until it
    leaves out every chat / overlay box (never below 2.2 face widths)."""
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    cw = min(0.45 * W, max(4.0 * fw, 0.18 * W))
    while True:
        ch = cw / aspect
        x = min(max(0.0, cx - cw / 2), W - cw)
        y = min(max(0.0, cy - ch * 0.42), H - ch)
        box = [x, y, cw, ch]
        if cw <= 2.2 * fw or not any(_overlap(box, a) > 0 for a in avoid):
            break
        cw *= 0.92
    cw, ch = int(cw) // 2 * 2, int(cw / aspect) // 2 * 2
    return [int(x) // 2 * 2, int(y) // 2 * 2, cw, ch]


def _grow(b: Sequence[int], m: int, W: int, H: int) -> List[int]:
    x, y, w, h = b
    x0, y0 = max(0, x - m), max(0, y - m)
    return [x0, y0, min(W, x + w + m) - x0, min(H, y + h + m) - y0]


# ── full-game layout (owner 2026-11-10 ~05:40 / 05:45) ──────────────────────
# «надо чтоб ВСЮ игру было видно»: the bottom is the WHOLE game frame fit to 1080 wide (16:9 → 1080x608), only chat /
# overlay columns at the frame edges are trimmed off; what lies ON the game (the webcam overlay, a chat box in a
# corner) is blurred over instead of cropping half the game away. «качество вебки плохое»: the webcam is never
# upscaled more than MAX_UPSCALE; a small camera is shown smaller, centred, on a blurred copy of itself.

OUT_W, OUT_H = 1080, 1920
MAX_UPSCALE = 2.0              # owner: «не растягивать вебку сильнее ×1.5–2»
EDGE_TRIM_MAX = 0.17           # a chat column at the frame edge narrower than this share of the width is cut off
MIN_TOP_H = 560                # the webcam area never gets less than this (a very tall trimmed game is capped)


def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


def game_region(W: int, H: int, avoid: Sequence[Sequence[float]], *,
                edge_trim_max: float = EDGE_TRIM_MAX) -> Tuple[List[int], List[List[int]]]:
    """The whole game frame minus edge columns / rows that are chat or overlays.
    -> (game box [x, y, w, h] source px, masks [[x, y, w, h]] in GAME-box px: boxes to blur over).
    A box touching the left / right (top / bottom) edge whose width (height) is ≤ edge_trim_max of the frame is
    trimmed off with its whole column (row); anything else (a webcam in a corner, a chat box on the game) is masked."""
    x0, y0, x1, y1 = 0, 0, W, H
    rest: List[Sequence[float]] = []
    for b in avoid:
        bx, by, bw, bh = (float(v) for v in b)
        if bw <= 0 or bh <= 0:
            continue
        cuts = []
        if bx <= 0.03 * W and bx + bw <= edge_trim_max * W:
            cuts.append(("l", bx + bw, (bx + bw) / W))
        if bx + bw >= 0.97 * W and W - bx <= edge_trim_max * W:
            cuts.append(("r", bx, (W - bx) / W))
        if by <= 0.03 * H and by + bh <= edge_trim_max * H and bw >= 0.5 * W:
            cuts.append(("t", by + bh, (by + bh) / H))
        if by + bh >= 0.97 * H and H - by <= edge_trim_max * H and bw >= 0.5 * W:
            cuts.append(("b", by, (H - by) / H))
        if not cuts:
            rest.append(b)
            continue
        side, pos, _ = min(cuts, key=lambda c: c[2])
        if side == "l":
            x0 = max(x0, int(np.ceil(pos)))
        elif side == "r":
            x1 = min(x1, int(pos))
        elif side == "t":
            y0 = max(y0, int(np.ceil(pos)))
        else:
            y1 = min(y1, int(pos))
    x0, y0 = _even(x0 + 1) if x0 else 0, _even(y0 + 1) if y0 else 0
    gw, gh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    masks: List[List[int]] = []
    for b in rest:
        bx, by, bw, bh = (float(v) for v in b)
        mx0, my0 = max(x0, int(bx) - 4), max(y0, int(by) - 4)            # 4 px: the overlay's own frame line
        mx1, my1 = min(x0 + gw, int(np.ceil(bx + bw)) + 4), min(y0 + gh, int(np.ceil(by + bh)) + 4)
        if mx1 - mx0 >= 8 and my1 - my0 >= 8:
            masks.append([(mx0 - x0) // 2 * 2, (my0 - y0) // 2 * 2, (mx1 - mx0) // 2 * 2, (my1 - my0) // 2 * 2])
    return [x0, y0, gw, gh], masks


def cam_fit(rect: Sequence[float], face_cx: float, top_h: int, *, out_w: int = OUT_W,
            max_up: float = MAX_UPSCALE, min_keep: float = 0.55) -> Tuple[List[int], List[int]]:
    """Webcam crop + its on-screen size. The crop keeps the full camera height and min_keep..100 % of its width
    (centred on the face); its scale is ≤ max_up, ≤ out_w / crop width and ≤ top_h / crop height; the variant
    that shows the most screen area wins (a big camera may be cropped to fill the top, a small one is never
    stretched — it stays smaller). -> (crop [x, y, w, h] source px, size [w, h] output px)."""
    x, y, w, h = (float(v) for v in rect)
    best = None
    for f in np.linspace(1.0, min_keep, 10):
        cw = w * f
        s = min(max_up, out_w / cw, top_h / h)
        area = s * s * cw * h
        if best is None or area > best[0] * 1.02:
            best = (area, f, s)
    _, f, s = best
    cw = max(16, int(w * f) // 2 * 2)
    ch = max(16, int(h) // 2 * 2)
    cx = min(max(x, face_cx - cw / 2.0), x + w - cw)
    crop = [int(np.ceil(cx / 2.0)) * 2, int(np.ceil(y / 2.0)) * 2, cw, ch]
    crop[0] = min(crop[0], int(x + w - cw) // 2 * 2)
    crop[1] = min(crop[1], int(y + h - ch) // 2 * 2)
    size = [min(out_w, _even(cw * s)), min(top_h, _even(ch * s))]
    return crop, size


def full_game_geometry(W: int, H: int, cam_rect: Sequence[float], face_cx: float,
                       avoid: Sequence[Sequence[float]], *, max_up: float = MAX_UPSCALE) -> Dict[str, Any]:
    """Output geometry of the full-game layout: game box + masks, game height on screen, webcam crop / size /
    position (centred in the area above the game)."""
    game, masks = game_region(W, H, avoid)
    game_h = min(OUT_H - MIN_TOP_H, _even(OUT_W * game[3] / float(game[2])))
    top_h = OUT_H - game_h
    crop, size = cam_fit(cam_rect, face_cx, top_h, max_up=max_up)
    pos = [(OUT_W - size[0]) // 2 // 2 * 2, (top_h - size[1]) // 2 // 2 * 2]
    return {"game": game, "game_masks": masks, "game_h": game_h, "top_h": top_h, "cam": crop, "cam_size": size,
            "cam_pos": pos, "cam_upscale": round(size[0] / float(crop[2]), 3)}


def stream_game_track(analysis: Optional[Dict[str, Any]], layout: Dict[str, Any], duration: float, *,
                      width: int, height: int, survey_rects: Sequence[Dict[str, Any]] = (),
                      max_up: float = MAX_UPSCALE) -> Optional[Dict[str, Any]]:
    """Full-game layout track (game-v3): top = ONLY the webcam (inside its rectangle, never upscaled more than
    max_up, on a blurred copy of itself), bottom = the WHOLE game fit to 1080 wide, edge chat columns trimmed,
    the webcam / chat boxes on the game blurred over. Subtitles go right above the game (the seam).
    None → no webcam: the caller uses the person / centre crop."""
    W, H = int(width), int(height)
    face = webcam_face(analysis, W, H)
    cam = layout.get("webcam")
    chat = [_grow(c, int(0.02 * W), W, H) for c in layout.get("chat") or []]
    source = None
    rect = None
    if cam:
        rect, source = cam["rect"], f"rect:{cam.get('by')}"
    if rect is None and face is not None:
        cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
        fits = [s["rect"] for s in survey_rects
                if s["rect"][0] < cx < s["rect"][0] + s["rect"][2] and s["rect"][1] < cy < s["rect"][1] + s["rect"][3]
                and s["rect"][2] >= 1.8 * face[2]]
        if fits:
            rect, source = min(fits, key=lambda r: r[2] * r[3]), "rect:survey"
    if rect is None and face is None:
        return None
    if rect is not None:
        # a 2 px inset: the overlay's own border line stays out of the picture
        cam_rect = [rect[0] + 2, rect[1] + 2, rect[2] - 4, rect[3] - 4]
    else:
        cam_rect = fallback_cam(face, W, H, chat, aspect=16 / 9.0)
        source = "face_window"
    face_cx = face[0] + face[2] / 2 if face is not None else cam_rect[0] + cam_rect[2] / 2
    geo = full_game_geometry(W, H, cam_rect, face_cx, chat + [rect if rect is not None else cam_rect], max_up=max_up)
    game = geo["game"]
    cov = sum(_overlap(game, a) for a in chat) / float(game[2] * game[3])
    return {"crop_version": "game-v3", "layout": "game", "crop_status": "tracked", "fallback_reason": None,
            "detector": ((analysis or {}).get("meta") or {}).get("detector"), "source_width": W, "source_height": H,
            "output_aspect": "9:16", **geo, "webcam_rect": rect, "webcam_source": source,
            "chat_zones": layout.get("chat") or [], "game_overlap": round(cov, 3),
            "seam_y": geo["top_h"],
            "seam": [[0.0, round(duration, 3)]], "keyframes": [[0.0, game[0]], [round(duration, 3), game[0]]],
            "cuts": (analysis or {}).get("cuts", []), "shots": [], "layout_version": VERSION}
