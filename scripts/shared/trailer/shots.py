"""
Trailer shots: ONE ffmpeg pass over the storyline (5 fps, 640 px) → per sample faces (YuNet, the Cut crop engine's
detector), motion (Farneback flow on a 128-px grey frame) and hard cuts → the episode's shots (cut to cut) with
features → labelled frame collages for the writer (Gemini sees what each shot K… shows and picks shots per phrase).

Pure helpers (Scan.window / score / crop_x) are unit-tested without video.
"""
from __future__ import annotations

import base64
import bisect
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCAN_VERSION = "trailer-scan-v2"
SCAN_FPS = 5.0
DET_W = 640
CUT_DIFF = 28.0                  # mean abs grey difference (0–255) between two samples = a hard cut
MIN_SHOT = 0.8                   # shorter source shots are merged into their neighbour for the collages


def scan(src: str, a: float, b: float, *, ffmpeg: str = "ffmpeg", cache: Optional[str] = None,
         fps: float = SCAN_FPS) -> Dict[str, Any]:
    """[a, b] of the source → {"meta", "samples": [{t, motion, cut, faces: [[x, y, w, h, score] source px]}]}."""
    if cache and Path(cache).is_file():
        try:
            d = json.loads(Path(cache).read_text())
            m = d.get("meta") or {}
            if m.get("version") == SCAN_VERSION and abs(m.get("start", -1) - a) < 0.01 and abs(m.get("end", -1) - b) < 0.01:
                return d
        except (OSError, ValueError):
            pass
    import cv2
    import numpy as np
    from ..crop import clip_analysis as ca
    W, H = probe_dims(src)
    if not W or not H:
        raise RuntimeError("cannot read the source size")
    dw = min(DET_W, W) // 2 * 2
    dh = int(round(H * dw / W)) // 2 * 2
    st = ca.model_ok()
    det = cv2.FaceDetectorYN.create(str(ca.MODEL_PATH), "", (dw, dh), ca.SCORE_MIN, 0.3, 50) if st["ok"] else None
    if det is None:
        logger.warning("[trailer] face detector unavailable (%s): shots scored by motion only", st.get("reason"))
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-threads", "0", "-ss", f"{max(0.0, a):.3f}", "-t", f"{b - a:.3f}", "-i", src,
           "-an", "-sn", "-vf", f"fps={fps},scale={dw}:{dh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = dw * dh * 3
    sx = W / float(dw)
    samples: List[Dict[str, Any]] = []
    prev, prev_hist = None, None
    k = 0
    gw = 128
    gh = max(8, int(round(dh * gw / dw)))
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(dh, dw, 3)
            t = a + k / fps
            k += 1
            boxes = []
            if det is not None:
                _, faces = det.detect(frame)
                boxes = [[int(f[0] * sx), int(f[1] * sx), int(f[2] * sx), int(f[3] * sx), round(float(f[14]), 3)]
                         for f in (faces if faces is not None else [])]
                boxes.sort(key=lambda x: -x[2] * x[3])
            small = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
            motion, cut = None, False
            if prev is not None:
                diff = float(cv2.absdiff(prev, g).mean())
                hd = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if diff > CUT_DIFF or (diff > 9.0 and hd > 0.38) or hd > 0.55:
                    cut = True
                else:
                    fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                    motion = float(cv2.cartToPolar(fl[..., 0], fl[..., 1])[0].mean())
            prev, prev_hist = g, hist
            samples.append({"t": round(t, 3), "motion": None if motion is None else round(motion, 4), "cut": cut,
                            "faces": boxes[:4], "luma": round(float(g.mean()), 1)})
    finally:
        p.stdout.close()
        p.wait()
    if not samples:
        raise RuntimeError("the scan read no frames")
    data = {"meta": {"version": SCAN_VERSION, "width": W, "height": H, "fps": fps, "start": a, "end": b,
                     "samples": len(samples), "detector": st.get("detector") if st["ok"] else None}, "samples": samples}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(data))
    return data


def probe_dims(src: str) -> Tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "json", src], capture_output=True, text=True)
    s = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    return int(s.get("width") or 0), int(s.get("height") or 0)


def _pct(vals: Sequence[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


class Scan:
    """Window features, shot score and crop position over the scan samples. Pure."""

    FACE_REF = 0.34                         # face height / frame height of a full close-up

    def __init__(self, data: Dict[str, Any]):
        self.s = data["samples"]
        self.t = [x["t"] for x in self.s]
        self.W = int(data["meta"]["width"])
        self.H = int(data["meta"]["height"])
        self.fps = float(data["meta"].get("fps") or SCAN_FPS)
        mot = [x["motion"] for x in self.s if x["motion"] is not None]
        self.ref = max(0.05, _pct(mot, 0.8))
        self.low = _pct(mot, 0.35)
        self.calm = _pct(mot, 0.5)
        self.cuts = [round(x["t"] - 0.5 / self.fps, 3) for x in self.s if x["cut"]]

    def cuts_in(self, a: float, b: float) -> List[float]:
        i, j = bisect.bisect_left(self.cuts, a), bisect.bisect_right(self.cuts, b)
        return self.cuts[i:j]

    def window(self, t0: float, t1: float, crop: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Features of [t0, t1); with crop = (x, cw) only faces inside the crop window count."""
        i, j = bisect.bisect_left(self.t, t0), bisect.bisect_left(self.t, t1)
        ws = self.s[i:j]
        if not ws:
            return {"n": 0, "presence": 0.0, "face": 0.0, "motion": 0.0, "cuts": 0, "dark": 0.0}

        def inside(f):
            return crop is None or crop[0] + 0.1 * crop[1] <= f[0] + f[2] / 2.0 <= crop[0] + 0.9 * crop[1]
        fh = []
        for x in ws:
            fs = [f for f in x["faces"] if inside(f)]
            if fs:
                fh.append(max(f[3] for f in fs) / self.H)
        mots = [x["motion"] for x in ws if x["motion"] is not None]
        dark = sum(1 for x in ws if x.get("luma", 100) < 18) / len(ws)
        return {"n": len(ws), "presence": len(fh) / len(ws), "face": sum(fh) / len(fh) if fh else 0.0,
                "motion": sum(mots) / len(mots) if mots else 0.0, "cuts": sum(1 for x in ws[1:] if x["cut"]),
                "dark": dark}

    def score(self, f: Dict[str, Any]) -> float:
        """Big faces and motion win; empty, dark, static wide shots lose."""
        if f["n"] == 0:
            return -9.0
        face = min(f["face"] / self.FACE_REF, 1.0) * f["presence"]
        mot = min(f["motion"] / self.ref, 1.5) * (0.4 + 0.6 * f["presence"])
        sc = 2.0 * face + 1.2 * mot
        if f["presence"] < 0.4:
            sc -= 1.2 + (0.8 if f["motion"] <= self.low else 0.0)
        elif f["face"] < 0.10:
            sc -= 0.4
        sc -= 2.5 * f.get("dark", 0.0)
        return round(sc - 3.0 * f["cuts"], 3)

    def is_static(self, f: Dict[str, Any]) -> bool:
        return f["motion"] <= max(self.calm, 0.12 * self.ref)

    def crop_x(self, t0: float, t1: float, cw: int) -> Tuple[int, int]:
        """Crop window x at the start and the end of the shot: centred on the main subject (the biggest face, else the
        middle), drifting at most 15 % of the window (never reads as a whip pan)."""
        mid = (self.W - cw) // 2

        def cx(a: float, b: float) -> Optional[float]:
            i, j = bisect.bisect_left(self.t, a), bisect.bisect_right(self.t, b)
            num = den = 0.0
            for x in self.s[i:j]:
                if x["faces"]:
                    f = x["faces"][0]
                    w = f[2] * f[3] * max(0.3, f[4])
                    num += (f[0] + f[2] / 2.0) * w
                    den += w
            return num / den if den else None
        d = max(0.1, t1 - t0)
        whole = cx(t0, t1)
        c0 = cx(t0 - 0.1, t0 + d / 3) or whole
        c1 = cx(t1 - d / 3, t1 + 0.1) or whole
        if c0 is None and c1 is None:
            return mid // 2 * 2, mid // 2 * 2
        c0 = c0 if c0 is not None else c1
        c1 = c1 if c1 is not None else c0
        x0 = min(max(0.0, c0 - cw / 2.0), self.W - cw)
        x1 = min(max(0.0, c1 - cw / 2.0), self.W - cw)
        x1 = x0 + max(-0.15 * cw, min(0.15 * cw, x1 - x0))
        return int(x0) // 2 * 2, int(min(max(0.0, x1), self.W - cw)) // 2 * 2

    def face_rel(self, t0: float, t1: float, x: float, cw: int) -> Tuple[float, float]:
        """Where the main face sits in the crop window (0..1, 0..1): the push-in keeps that point still."""
        i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
        fs = [f for s in self.s[i:j] for f in s["faces"][:1]]
        if not fs:
            return 0.5, 0.4
        fx = (sum(f[0] + f[2] / 2.0 for f in fs) / len(fs) - x) / cw
        fy = sum(f[1] + f[3] / 2.0 for f in fs) / len(fs) / self.H
        return min(0.8, max(0.2, fx)), min(0.7, max(0.2, fy))


def source_shots(sc: Scan, a: float, b: float) -> List[Dict[str, Any]]:
    """The episode cut into its own shots (cut to cut); very short ones merged into the previous shot."""
    bounds = [a] + [c for c in sc.cuts if a + 0.05 < c < b - 0.05] + [b]
    raw = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    merged: List[List[float]] = []
    for s, e in raw:
        if merged and (e - s < MIN_SHOT or merged[-1][1] - merged[-1][0] < MIN_SHOT):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for k, (s, e) in enumerate(merged):
        f = sc.window(s + 0.1, e - 0.1) if e - s > 0.4 else sc.window(s, e)
        out.append({"id": f"K{k + 1}", "start": round(s, 3), "end": round(e, 3), "dur": round(e - s, 3),
                    "face": round(f["face"], 3), "presence": round(f["presence"], 2), "motion": round(f["motion"], 3),
                    "score": sc.score(dict(f, cuts=0))})
    return out


def pick_for_collage(shots: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """At most n shots for the writer: the whole episode evenly (time buckets), the best-scored shot of each bucket."""
    if len(shots) <= n:
        return list(shots)
    a, b = shots[0]["start"], shots[-1]["end"]
    step = (b - a) / n
    out = []
    for k in range(n):
        lo, hi = a + k * step, a + (k + 1) * step
        cand = [s for s in shots if lo <= (s["start"] + s["end"]) / 2 < hi and s["dur"] >= 1.0] or \
               [s for s in shots if lo <= (s["start"] + s["end"]) / 2 < hi]
        if cand:
            out.append(max(cand, key=lambda s: s["score"]))
    return out


def collages(src: str, shots: Sequence[Dict[str, Any]], *, ffmpeg: str = "ffmpeg", per: int = 16, cols: int = 4,
             tile_w: int = 320, tile_h: int = 180) -> List[str]:
    """Labelled 4×4 frame collages (the middle frame of each shot) → base64 JPEGs. One collage ≈ 1100 tokens."""
    import cv2
    import numpy as np
    from ..hooks import grab_frame
    out = []
    for k in range(0, len(shots), per):
        chunk = shots[k:k + per]
        rows = int(math.ceil(len(chunk) / cols))
        canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for n, s in enumerate(chunk):
            img = grab_frame(src, (s["start"] + s["end"]) / 2.0, width=tile_w, ffmpeg=ffmpeg)
            r, c = divmod(n, cols)
            if img is not None:
                h, w = img.shape[:2]
                sc = min(tile_w / w, tile_h / h)
                im = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
                y0 = r * tile_h + (tile_h - im.shape[0]) // 2
                x0 = c * tile_w + (tile_w - im.shape[1]) // 2
                canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.rectangle(canvas, (c * tile_w, r * tile_h), (c * tile_w + 64, r * tile_h + 28), (0, 0, 0), -1)
            cv2.putText(canvas, s["id"], (c * tile_w + 4, r * tile_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                        2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if ok:
            out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return out
