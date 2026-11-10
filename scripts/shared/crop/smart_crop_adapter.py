"""
smart_crop_adapter.py
=====================
Smart vertical crop adapter for SONYA production.
Output: 1080x1920 (9:16)

Priority:
  1. SMART_CROP_ENABLED=true + GeminiComposer available → compose_clip_auto (AI-guided)
  2. SMART_CROP_ENABLED=true + SmartCropper available   → crop_video (face-following)
  3. fallback → ffmpeg center crop
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


def _flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def get_crop_hints(
    input_video_path: str,
    yolo_detections: Optional[List] = None,
    webcam_layout: Optional[Dict] = None,
    pose_detections: Optional[List] = None,
    layout_segments: Optional[List] = None,
) -> Dict[str, Any]:
    """Synthesize crop region hints from available analysis data."""
    hints: Dict[str, Any] = {"method": "center", "strategy": "default"}

    # Highest priority: layout-aware strategy
    if layout_segments:
        types = [s.get("layout", s.get("type", "")) for s in layout_segments]
        dominant = max(set(types), key=types.count) if types else ""
        hints["dominant_layout"] = dominant
        if dominant in ("screen_share", "screen"):
            hints["strategy"] = "letterbox"
        elif dominant in ("single_speaker", "person"):
            hints["strategy"] = "person_center"

    # Person tracking from webcam detector
    if webcam_layout and webcam_layout.get("boxes"):
        hints["method"] = "webcam"
        hints["webcam_boxes"] = webcam_layout["boxes"]

    # YOLO person detection
    if yolo_detections:
        hints["method"] = hints.get("method", "yolo") if hints["method"] == "center" else hints["method"]
        hints["yolo_detections"] = yolo_detections

    # Pose keypoints
    if pose_detections:
        hints["pose_detections"] = pose_detections

    return hints


def compose_vertical_clip(
    input_video_path: str,
    output_path: str,
    crop_hints: Optional[Dict[str, Any]] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> str:
    """
    Compose a vertical 1080x1920 clip.
    Respects start_time/duration by pre-cutting with ffmpeg before composing.
    Returns path to output file.
    """
    output_path = str(output_path)

    # Pre-cut segment if time range is specified
    if start_time is not None or duration is not None:
        tmp_cut = _cut_segment(input_video_path, start_time, duration)
        source = tmp_cut
        cleanup_tmp = True
    else:
        source = input_video_path
        cleanup_tmp = False

    try:
        if _flag("SMART_CROP_ENABLED"):
            # Try GeminiComposer first (AI-guided, highest quality)
            if _flag("GEMINI_COMPOSER_ENABLED"):
                try:
                    return _compose_gemini(source, output_path, crop_hints)
                except Exception as exc:
                    logger.warning("[smart_crop] GeminiComposer failed (%s) — trying SmartCropper", exc)

            # Try SmartCropper (face-following Haar cascade)
            try:
                return _compose_smart_cropper(source, output_path)
            except Exception as exc:
                logger.warning("[smart_crop] SmartCropper failed (%s) — ffmpeg fallback", exc)

        # Final fallback: ffmpeg center crop
        return _ffmpeg_center_crop(source, output_path)

    finally:
        if cleanup_tmp and os.path.exists(source):
            try:
                os.unlink(source)
            except OSError:
                pass


def _cut_segment(input_path: str, start_time: Optional[float], duration: Optional[float]) -> str:
    """Cut video segment to a temp file. Returns temp file path."""
    suffix = Path(input_path).suffix or ".mp4"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    cmd = ["ffmpeg", "-y"]
    if start_time is not None and start_time > 0:
        cmd += ["-ss", str(start_time)]
    cmd += ["-i", input_path]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-avoid_negative_ts", "1", tmp]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {result.stderr[-300:]}")
    return tmp


def _compose_gemini(source: str, output_path: str, hints: Optional[Dict]) -> str:
    """Use GeminiComposer.compose_clip_auto for AI-guided vertical composition."""
    from scripts.shared.gemini.gemini_composer import GeminiComposer

    output_dir = str(Path(output_path).parent)
    composer = GeminiComposer(output_dir=output_dir)

    # compose_clip_auto handles scene detection + YOLO layout → produces 9:16 output
    result = composer.compose_clip_auto(input_path=source, output_path=output_path)
    if result and os.path.exists(result):
        return result
    raise RuntimeError("GeminiComposer produced no output")


def _compose_smart_cropper(source: str, output_path: str) -> str:
    """Use SmartCropper (face-following Haar cascade) for 9:16 crop."""
    from scripts.shared.crop.cropper import SmartCropper

    cropper = SmartCropper()
    # crop_video returns the output path; output_path treated as destination file
    result = cropper.crop_video(input_video=source, output_path=str(Path(output_path).parent))
    if result and os.path.exists(result):
        # SmartCropper writes its own filename — rename to expected output_path
        if result != output_path:
            import shutil
            shutil.move(result, output_path)
        return output_path
    raise RuntimeError("SmartCropper produced no output")


def _ffmpeg_center_crop(source: str, output_path: str) -> str:
    """Fallback: scale to fill 1080x1920, then center crop."""
    vf = (
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", source,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac",
        output_path,
    ]
    logger.info("[smart_crop] ffmpeg center crop → %s", output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg center crop failed: {result.stderr[-500:]}")
    return output_path


# ── Tracked 9:16 crop from a precomputed face analysis (one pass per source) ──

CROP_VERSION = "crop-track-v1"


def crop_window(width: int, height: int) -> Tuple[int, int]:
    """Largest EXACT 9:16 window inside the frame: 9m x 16m with m even, so
    both sides are even (yuv420p) and scaling to 1080x1920 never distorts."""
    m = min(height // 16, width // 9)
    m -= m % 2
    m = max(2, m)
    return 9 * m, 16 * m


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def crop_track(analysis: Optional[Dict[str, Any]], start: float, duration: float,
               *, width: Optional[int] = None, height: Optional[int] = None,
               dead_zone: float = 0.08, hold_sec: float = 1.0, move_sec: float = 0.6,
               median_window: int = 5) -> Dict[str, Any]:
    """Deterministic crop path for [start, start+duration] (times relative to the clip).

    Face center x of the largest face per sample → hold-last gap fill →
    median filter (kills single-sample outliers) → dead zone: the window only
    moves when the subject stays ≥ dead_zone·W away for ≥ hold_sec, then glides
    over move_sec. x is clamped to [0, W-cw]. No detections → centered window.
    """
    meta = (analysis or {}).get("meta") or {}
    W = int(width or meta.get("width") or 0)
    H = int(height or meta.get("height") or 0)
    info: Dict[str, Any] = {"crop_version": CROP_VERSION, "source_width": W, "source_height": H,
                            "output_aspect": "9:16", "detector": meta.get("detector"),
                            "analysis_version": meta.get("version")}
    if W <= 0 or H <= 0:
        raise ValueError("crop_track needs source dimensions")
    cw, ch = crop_window(W, H)
    max_x = W - cw
    center = max_x // 2 // 2 * 2
    info.update(crop_w=cw, crop_h=ch, y=(H - ch) // 2 // 2 * 2)
    if analysis is None:
        return dict(info, crop_status="center_fallback", fallback_reason="detector_unavailable",
                    keyframes=[[0.0, center], [round(duration, 3), center]])
    end = start + duration
    pts = [(s["t"] - start, s["faces"][0]) for s in analysis.get("samples", [])
           if start - 1.0 <= s["t"] <= end + 1.0 and s.get("faces")]
    if not pts:
        return dict(info, crop_status="center_fallback", fallback_reason="no_subject_found",
                    keyframes=[[0.0, center], [round(duration, 3), center]])

    def clamp(x: float) -> int:
        return int(max(0, min(max_x, round(x))) // 2 * 2)

    raw = [(t, clamp(b[0] + b[2] / 2.0 - cw / 2.0)) for t, b in pts]
    half = median_window // 2
    smooth = [(t, _median([x for _, x in raw[max(0, i - half):i + half + 1]])) for i, (t, _) in enumerate(raw)]
    cur = clamp(smooth[0][1])
    keys: List[List[float]] = [[0.0, cur]]
    away_since: Optional[float] = None
    for t, x in smooth:
        if t < 0:
            cur = clamp(x)
            keys = [[0.0, cur]]
            continue
        if abs(x - cur) >= dead_zone * W:
            away_since = t if away_since is None else away_since
            if t - away_since >= hold_sec - 1e-6:
                t0 = max(t, keys[-1][0])
                t1 = min(duration, t0 + move_sec)
                if t1 > t0:
                    keys.append([round(t0, 3), cur])
                    cur = clamp(x)
                    keys.append([round(t1, 3), cur])
                away_since = None
        else:
            away_since = None
    if keys[-1][0] < duration:
        keys.append([round(duration, 3), cur])
    return dict(info, crop_status="tracked", fallback_reason=None, keyframes=keys,
                samples_used=len(pts))


# ── v2: shot-aware crop from a per-clip analysis (scripts/shared/crop/clip_analysis) ──

CROP_VERSION_V2 = "crop-shots-v3"   # v3: wider square window on close-ups


def _clusters(dets: List[Tuple[float, List[float]]], W: int) -> List[Dict[str, Any]]:
    """Group a shot's face boxes into people: a box joins the person whose LAST
    position is nearest (so a walking person stays one person); people in one
    shot rarely swap places. Each: {"xs": [(t, cx, w)], "cx", "w", "weight"}."""
    out: List[Dict[str, Any]] = []
    for t, b in sorted(dets, key=lambda d: (d[0], d[1][0])):
        cx, w = b[0] + b[2] / 2.0, float(b[2])
        best = None
        for c in out:
            if c["last_t"] == t:
                continue                                   # one box per person per sample
            gap = abs(c["last"] - cx)
            if gap <= max(0.6 * max(w, c["w"]), 0.04 * W) and (best is None or gap < abs(best["last"] - cx)):
                best = c
        if best is None:
            best = {"xs": [], "cx": cx, "w": w, "weight": 0.0, "last": cx, "last_t": None}
            out.append(best)
        best["xs"].append((t, cx, w))
        n = len(best["xs"])
        best["cx"] += (cx - best["cx"]) / n
        best["w"] += (w - best["w"]) / n
        best["last"], best["last_t"] = cx, t
        best["weight"] += w * w * (b[4] if len(b) > 4 else 1.0)    # big, confident, present
    return out


def _talk(cluster: Dict[str, Any], dets: List[Tuple[float, List[float]]]) -> float:
    """Mean mouth motion of one person over the shot (0 when unknown)."""
    mine = {(t, round(cx, 1)) for t, cx, _ in cluster["xs"]}
    vals = [float(b[5]) for t, b in dets
            if len(b) > 5 and b[5] is not None and (t, round(b[0] + b[2] / 2.0, 1)) in mine]
    return sum(vals) / len(vals) if len(vals) >= 5 else 0.0     # talking is bursty: mean, not median


def _follow(xs: List[Tuple[float, float, float]], s0: float, s1: float, win_w: int, max_x: int, *,
            dead_zone: float, hold_sec: float, move_sec: float, median_window: int) -> List[List[float]]:
    """Keyframes [t, x] that keep one person's face inside a window of win_w over
    [s0, s1]: median filter, dead zone relative to the window, hold-then-glide,
    immediate move when the face would touch the window edge."""
    def clamp(x: float) -> int:
        return int(max(0, min(max_x, round(x))) // 2 * 2)
    raw = [(t, cx) for t, cx, _ in xs]
    fw = _median([w for _, _, w in xs])
    half = median_window // 2
    smooth = [(t, _median([v for _, v in raw[max(0, i - half):i + half + 1]])) for i, (t, _) in enumerate(raw)]
    margin = max(0.0, win_w / 2.0 - fw / 2.0 - 0.04 * win_w)   # how far off-centre the face may sit
    dz = max(0.04 * win_w, min(dead_zone * win_w, margin))      # tiny moves read as jitter
    cur = clamp(_median([v for _, v in smooth[:max(1, min(len(smooth), 5))]]) - win_w / 2.0)
    keys: List[List[float]] = [[round(s0, 3), cur]]
    away = None
    for t, v in smooth:
        want = clamp(v - win_w / 2.0)
        if abs(want - cur) >= dz:
            away = t if away is None else away
            urgent = v - fw / 2.0 < cur + 0.02 * win_w or v + fw / 2.0 > cur + win_w - 0.02 * win_w
            if urgent or t - away >= hold_sec - 1e-6:
                t0 = max(t, keys[-1][0])
                t1 = min(s1, t0 + (move_sec * 0.6 if urgent else move_sec))
                if t1 > t0:
                    keys += [[round(t0, 3), cur], [round(t1, 3), want]]
                    cur = want
                away = None
        else:
            away = None
    keys.append([round(s1, 3), cur])
    return keys


def _absorb_short(runs: List[List[Any]], min_seg: float) -> List[List[Any]]:
    """Merge runs [state, t0, t1] shorter than min_seg into their longer neighbour
    (no layout flicker), then join neighbours with the same state."""
    runs = [list(r) for r in runs]
    while len(runs) > 1:
        dur = [r[2] - r[1] for r in runs]
        i = min(range(len(runs)), key=lambda k: dur[k])
        if dur[i] >= min_seg:
            break
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            j = i - 1 if dur[i - 1] >= dur[i + 1] else i + 1
        a, b = sorted((i, j))
        runs[a:b + 1] = [[runs[j][0], runs[a][1], runs[b][2]]]
        joined: List[List[Any]] = []
        for r in runs:
            if joined and joined[-1][0] == r[0]:
                joined[-1][2] = r[2]
            else:
                joined.append(r)
        runs = joined
    return runs


CLOSEUP_FACE_FRAC = 0.5      # face wider than half the 9:16 window = a close-up: show it wider


def _is_closeup(xs: List[Tuple[float, float, float]], cw: int) -> bool:
    return bool(xs) and _median([w for _, _, w in xs]) > CLOSEUP_FACE_FRAC * cw


def split_window(W: int, H: int, gap: Optional[float] = None) -> Tuple[int, int]:
    """Window for one half of a split screen: 9:8 (→ 1080x960). No face zoom: the
    largest window (up to the full source height) whose width stays within the
    distance between the two people, so a half shows its own person only;
    never smaller than half the source height."""
    h = float(H) if not gap else max(0.5 * H, min(float(H), gap * 8.0 / 9.0))
    m = max(2, int(h) // 8)
    m -= m % 2
    return min(9 * m, W - W % 2), 8 * m


GAME_VERSION = "game-v1"


def game_track(analysis: Optional[Dict[str, Any]], duration: float, *, width: int, height: int,
               min_presence: float = 0.3, min_score: float = 0.7) -> Optional[Dict[str, Any]]:
    """Stream / gameplay layout (owner's Streamer rule «веб-камера сверху, игра снизу»):
    top 1080x960 = the streamer's webcam (found as the small face that stays in one place),
    bottom 1080x960 = the centre of the game (crosshair), subtitles on the seam.
    None when no steady webcam face is found (the caller falls back to the person layout)."""
    W, H = int(width), int(height)
    if not analysis or not analysis.get("samples"):
        return None
    pts = []
    for s in analysis["samples"]:
        small = [f for f in s.get("faces") or [] if f[4] >= min_score and f[2] < 0.22 * W]
        if small:
            f = max(small, key=lambda b: b[2] * b[3])
            pts.append((f[0] + f[2] / 2, f[1] + f[3] / 2, f[2]))
    if len(pts) < min_presence * len(analysis["samples"]):
        return None
    xs, ys, ws = (sorted(v) for v in zip(*pts))
    cx, cy, fw = xs[len(xs) // 2], ys[len(ys) // 2], ws[len(ws) // 2]
    # a webcam face sits in one place; a moving one is a person in the game / camera shot
    q = lambda v: (v[int(len(v) * 0.9)] - v[int(len(v) * 0.1)])
    if q(xs) / W > 0.08 or q(ys) / H > 0.08 or q(ws) > 0.5 * fw:
        return None     # a webcam face holds its place AND size; anything else is in the game
    camw = int(min(0.45 * W, max(4.0 * fw, 0.18 * W))) // 2 * 2
    camh = int(camw * 960 / 1080) // 2 * 2
    camx = int(min(max(0, cx - camw / 2), W - camw)) // 2 * 2
    camy = int(min(max(0, cy - camh * 0.42), H - camh)) // 2 * 2
    gh = int(H * 0.9) // 2 * 2
    gw = int(gh * 1080 / 960) // 2 * 2
    if gw > W:                         # narrow source: fit by width, keep the 1080:960 aspect
        gw = W // 2 * 2
        gh = int(gw * 960 / 1080) // 2 * 2
    gx, gy = (W - gw) // 2 // 2 * 2, (H - gh) // 2 // 2 * 2
    return {"crop_version": GAME_VERSION, "layout": "game", "crop_status": "tracked", "fallback_reason": None,
            "detector": (analysis.get("meta") or {}).get("detector"), "source_width": W, "source_height": H,
            "output_aspect": "9:16", "cam": [camx, camy, camw, camh], "game": [gx, gy, gw, gh],
            "seam": [[0.0, round(duration, 3)]], "keyframes": [[0.0, gx], [round(duration, 3), gx]],
            "cuts": analysis.get("cuts", []), "shots": [], "webcam_presence": round(len(pts) / len(analysis["samples"]), 3)}


def crop_track_v2(analysis: Optional[Dict[str, Any]], duration: float, *, width: int, height: int,
                  dead_zone: float = 0.12, hold_sec: float = 0.8, move_sec: float = 0.5,
                  median_window: int = 5, min_presence: float = 0.25,
                  min_score: float = 0.8, min_seg: float = 0.5) -> Dict[str, Any]:
    """Crop path for one clip from a per-clip analysis (times relative to the clip).

    The clip is split into shots at the detected cuts; the window JUMPS at a cut
    (an editor's cut, never a pan across it). Inside a shot it frames the main
    person — or two people together when both fit — and follows them with a
    dead zone (relative to the crop width) that also keeps the whole face inside
    the window. Shot without a usable face → centred window.
    """
    W, H = int(width), int(height)
    cw, ch = crop_window(W, H)
    max_x = W - cw
    center = max_x // 2 // 2 * 2
    meta = (analysis or {}).get("meta") or {}
    info: Dict[str, Any] = {"crop_version": CROP_VERSION_V2, "source_width": W, "source_height": H,
                            "output_aspect": "9:16", "detector": meta.get("detector"),
                            "analysis_version": meta.get("version"), "crop_w": cw, "crop_h": ch,
                            "y": (H - ch) // 2 // 2 * 2}
    if analysis is None:
        return dict(info, crop_status="center_fallback", fallback_reason="detector_unavailable",
                    keyframes=[[0.0, center], [round(duration, 3), center]], shots=[])

    def clamp(x: float) -> int:
        return int(max(0, min(max_x, round(x))) // 2 * 2)

    cuts = [c for c in analysis.get("cuts") or [] if 0.0 < c < duration]
    bounds = [0.0] + cuts + [duration]
    samples = analysis.get("samples") or []
    keys: List[List[float]] = []
    shots: List[Dict[str, Any]] = []
    splits_raw: List[Dict[str, Any]] = []
    closeups: List[Tuple[float, float, List[Tuple[float, float, float]]]] = []
    for s0, s1 in zip(bounds, bounds[1:]):
        in_shot = [x for x in samples if s0 - 1e-6 <= x["t"] < s1 - 1e-6]
        # confident, plausible faces only: a blurred foreground shoulder / back of a
        # head scores ~0.6 and is huge; real faces (incl. 3/4 views) score ≥ 0.8
        dets = [(x["t"], b) for x in in_shot for b in x.get("faces") or []
                if (b[4] if len(b) > 4 else 1.0) >= min_score and b[3] <= 0.8 * H]
        people = sorted(_clusters(dets, W), key=lambda c: -c["weight"])
        people = [c for c in people if len({t for t, _, _ in c["xs"]}) >= min_presence * max(1, len(in_shot))]
        shot = {"start": round(s0, 3), "end": round(s1, 3), "faces": len(people)}
        if not people:
            x0 = keys[-1][1] if keys and s1 - s0 < 1.0 else center   # a short faceless insert keeps framing
            shot["target"] = "none"
            keys += [[round(s0, 3), x0], [round(s1, 3), x0]]
            shots.append(shot)
            continue
        main = people[0]
        others = [c for c in people[1:] if c["w"] >= 0.35 * main["w"]]   # a real second face, not a poster
        if not others:
            shot["target"] = "person"
            keys += _follow(main["xs"], s0, s1, cw, max_x, dead_zone=dead_zone, hold_sec=hold_sec,
                            move_sec=move_sec, median_window=median_window)
            if _is_closeup(main["xs"], cw):
                closeups.append((s0, s1, main["xs"]))
                shot["target"] = "closeup"
            shots.append(shot)
            continue
        # two faces close together (both fit in ONE 9:16 window): a split would show the same
        # pair twice (owner 2026-11-09: «почему в кадре два одинаковых фрагмента») → one window
        # following the middle of the pair.
        pair_people = sorted((main, others[0]), key=lambda c: c["cx"])
        gap0 = abs(pair_people[1]["cx"] - pair_people[0]["cx"])
        half_w = min(int(0.5 * H) * 9 // 8, W)   # smallest split half: 9:8 at half the source height (no face zoom)
        if gap0 < 0.5 * half_w + 0.5 * max(pair_people[0]["w"], pair_people[1]["w"]):
            other = others[0]
            oxs = sorted(other["xs"])
            def mid(t: float, cx: float, w: float) -> Tuple[float, float, float]:
                near_o = min(oxs, key=lambda o: abs(o[0] - t)) if oxs else None
                if near_o is None or abs(near_o[0] - t) > 1.0:
                    return (t, cx, w)
                return (t, (cx + near_o[1]) / 2.0, w)
            shot["target"] = "pair"
            keys += _follow([mid(*x) for x in main["xs"]], s0, s1, cw, max_x, dead_zone=dead_zone,
                            hold_sec=hold_sec, move_sec=move_sec, median_window=median_window)
            shots.append(shot)
            continue
        # two faces: decide per moment (not per shot) — split while both are visible,
        # single while only one is, switching fast but never for a flicker < min_seg
        times = sorted({x["t"] for x in in_shot})
        seen = [{t for t, _, _ in c["xs"]} for c in pair_people]
        def near(st, t):
            """Person on screen at t: seen right then, or t lies in a short detector gap
            (a turned head) between two sightings — never extended past the first/last one."""
            before = [u for u in st if u <= t + 0.15]
            after = [u for u in st if u >= t - 0.15]
            if not before or not after:
                return False
            return min(after) - max(before) <= 2.5 or abs(t - max(before)) <= 0.15 or abs(min(after) - t) <= 0.15
        state = [("split" if near(seen[0], t) and near(seen[1], t) else
                  0 if near(seen[0], t) else 1 if near(seen[1], t) else None) for t in times]
        first = next((st for st in state if st is not None), 0)
        runs: List[List[Any]] = []                                        # [state, t_first, t_last]
        for t, st in zip(times, state):
            st = (runs[-1][0] if runs else first) if st is None else st  # detector gap: keep layout
            if runs and runs[-1][0] == st:
                runs[-1][2] = t
            else:
                runs.append([st, t, t])
        merged = _absorb_short(runs, min_seg)
        edges = [s0] + [(a[2] + b[1]) / 2.0 for a, b in zip(merged, merged[1:])] + [s1]
        shot["target"] = "split" if any(m[0] == "split" for m in merged) else "person"
        shot["segments"] = []
        for m, a, b in zip(merged, edges, edges[1:]):
            part = lambda c: [x for x in c["xs"] if a - 0.31 <= x[0] <= b + 0.31] or c["xs"]  # noqa: E731
            shot["segments"].append({"start": round(a, 3), "end": round(b, 3),
                                     "layout": "split" if m[0] == "split" else "single"})
            if m[0] == "split":
                gap = abs(pair_people[1]["cx"] - pair_people[0]["cx"])
                people2 = []
                for name, c in zip(("top", "bottom"), pair_people):
                    xs = part(c)
                    fy = _median([bb[1] + bb[3] / 2.0 for t, bb in dets if (t, round(bb[0] + bb[2] / 2.0, 1))
                                  in {(u, round(cx, 1)) for u, cx, _ in xs}] or [H / 3.0])
                    people2.append((name, xs, fy))
                splits_raw.append({"start": round(a, 3), "end": round(b, 3), "gap": gap, "_people": people2})
                x0 = keys[-1][1] if keys else center
                keys += [[round(a, 3), x0], [round(b, 3), x0]]           # single window hidden here
            else:
                keys += _follow(part(pair_people[m[0]]), a, b, cw, max_x, dead_zone=dead_zone,
                                hold_sec=hold_sec, move_sec=move_sec, median_window=median_window)
                if _is_closeup(part(pair_people[m[0]]), cw):
                    closeups.append((a, b, part(pair_people[m[0]])))
        shots.append(shot)
    # drop zero-length duplicates, keep jumps (same t, different x)
    clean: List[List[float]] = []
    for k in keys:
        if clean and clean[-1] == k:
            continue
        clean.append(k)
    status = "tracked" if any(sh["target"] != "none" for sh in shots) else "center_fallback"
    out = dict(info, crop_status=status, fallback_reason=None if status == "tracked" else "no_subject_found",
               keyframes=clean, shots=shots, cuts=cuts)
    if closeups:
        # Owner: «СЛИШКОМ СИЛЬНО ПРИБЛИЖАЕШЬ ЛИЦА». A face filling the 9:16 slice is shown in a
        # wider square window (full source height) instead, framed by the same scene blurred.
        sq = (min(W, H) // 2) * 2
        xk: List[List[float]] = []
        for a, b, xs in closeups:
            xk += _follow(xs, a, b, sq, W - sq, dead_zone=dead_zone, hold_sec=hold_sec, move_sec=move_sec,
                          median_window=median_window)
        out["closeup"] = {"size": sq, "y": (H - sq) // 2 // 2 * 2,
                          "intervals": [[round(a, 3), round(b, 3)] for a, b, _ in closeups], "x": xk}
    splits = splits_raw
    if splits:
        # one window size per clip (ffmpeg crop w/h are fixed). No zoom on faces: as
        # big as the source allows while each half still shows only its own person.
        sw, sh_ = split_window(W, H, min(x["gap"] for x in splits))
        for x in splits:
            halves = {}
            for name, xs, fy in x.pop("_people"):
                # face in the upper part of each half (head + body visible, subtitles on the seam)
                y = int(max(0, min(H - sh_, round(fy - 0.3 * sh_))) // 2 * 2)
                halves[name] = {"keys": _follow(xs, x["start"], x["end"], sw, W - sw, dead_zone=dead_zone,
                                                hold_sec=hold_sec, move_sec=move_sec,
                                                median_window=median_window), "y": y}
            x["split"] = halves
        tracks: Dict[str, Any] = {"w": sw, "h": sh_, "intervals": [[x["start"], x["end"]] for x in splits]}
        out["splits"] = [{"start": x["start"], "end": x["end"]} for x in splits]
        for name in ("top", "bottom"):
            xk: List[List[float]] = []
            yk: List[List[float]] = []
            for x in splits:
                xk += x["split"][name]["keys"]
                yk += [[x["start"], x["split"][name]["y"]], [x["end"], x["split"][name]["y"]]]
            tracks[name] = {"x": xk, "y": yk}
        out["split"] = tracks
    return out


FIT_VERSION = "fit-v1"
CLOSEUP_TOP = (OUTPUT_HEIGHT - OUTPUT_WIDTH) // 2 - 80      # square sits a bit above centre


def fit_track(width: int, height: int, duration: float, background: str = "black") -> Dict[str, Any]:
    """Whole source frame, uncropped and unzoomed, centred in 9:16; the rest of the
    screen is black bars or (background="blur") a blurred, darkened full-screen copy."""
    blur = background == "blur"
    return {"crop_version": "fit-blur-v1" if blur else FIT_VERSION, "crop_status": "fit", "fallback_reason": None,
            "layout": "fit", "background": "blur" if blur else "black",
            "detector": None, "source_width": int(width), "source_height": int(height), "output_aspect": "9:16",
            "keyframes": [], "shots": [], "cuts": []}


def _value_at(keys: List[List[float]], t: float) -> float:
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    return keys[-1][1] if keys else 0.0


def text_zones(analysis: Optional[Dict[str, Any]], track: Dict[str, Any], *, out_w: int = OUTPUT_WIDTH,
               out_h: int = OUTPUT_HEIGHT, hold: float = 0.5) -> List[List[float]]:
    """Where burned-in source text lands in the OUTPUT frame: [[t0, t1, y_top, y_bottom], …]
    (clip time, output pixels), mapped through the same layout the renderer uses."""
    W, H = int(track.get("source_width") or 0), int(track.get("source_height") or 0)
    if not analysis or not W or not H or track.get("layout") == "game":
        return []      # game layout: subtitles live on the seam; the HUD / kill feed is not a «title»
    sp = track.get("split") or {}
    spans: List[Tuple[float, float, float]] = []
    for smp in analysis.get("text") or []:
        t, boxes = float(smp["t"]), smp.get("boxes") or []
        if not boxes:
            continue
        ys: List[Tuple[float, float]] = []
        in_split = any(a <= t < b for a, b in sp.get("intervals") or [])
        cu = track.get("closeup") or {}
        in_cu = not in_split and any(a <= t < b for a, b in cu.get("intervals") or [])
        for x, y, w, h in boxes:
            if track.get("layout") == "fit":
                k = out_w / W
                off = (out_h - H * k) / 2
                ys.append((off + y * k, off + (y + h) * k))
            elif in_cu:
                k = out_w / cu["size"]
                a, b = CLOSEUP_TOP + (y - cu["y"]) * k, CLOSEUP_TOP + (y + h - cu["y"]) * k
                if b > CLOSEUP_TOP and a < CLOSEUP_TOP + out_w:
                    ys.append((max(float(CLOSEUP_TOP), a), min(float(CLOSEUP_TOP + out_w), b)))
            elif in_split:
                half = out_h / 2
                for name, base in (("top", 0.0), ("bottom", half)):
                    y0 = _value_at(sp[name]["y"], t)
                    a, b = (y - y0) * half / sp["h"], (y + h - y0) * half / sp["h"]
                    if b > 0 and a < half:
                        ys.append((base + max(0.0, a), base + min(half, b)))
            else:
                k = out_h / float(track.get("crop_h") or H)
                y0 = float(track.get("y") or 0)
                a, b = (y - y0) * k, (y + h - y0) * k
                if b > 0 and a < out_h:
                    ys.append((max(0.0, a), min(float(out_h), b)))
        if ys:
            spans.append((t, min(a for a, _ in ys), max(b for _, b in ys)))
    zones: List[List[float]] = []
    for t, a, b in spans:
        if zones and t - hold <= zones[-1][1] + 1e-6:
            z = zones[-1]
            z[1], z[2], z[3] = t + hold, min(z[2], a), max(z[3], b)
        else:
            zones.append([max(0.0, t - hold), t + hold, a, b])
    return [[round(a, 3), round(b, 3), round(c), round(d)] for a, b, c, d in zones]


def x_expression(keyframes: List[List[float]]) -> str:
    """Piecewise-linear ffmpeg expression of t for the crop x position."""
    expr = str(int(keyframes[-1][1]))
    for (t0, x0), (t1, x1) in reversed(list(zip(keyframes, keyframes[1:]))):
        if t1 <= t0 or x0 == x1:
            seg = str(int(x0))
        else:
            seg = f"{int(x0)}+({int(x1) - int(x0)})*(t-{t0})/({t1 - t0:.3f})"
        expr = f"if(lt(t,{t1}),{seg},{expr})"
    return expr


def _env_encoder_args() -> Optional[List[str]]:
    """Encoder chosen by the Desktop probe (SONYA_RENDER_ARGS, JSON list), if any."""
    raw = os.environ.get("SONYA_RENDER_ARGS")
    if not raw:
        return None
    try:
        args = json.loads(raw)
        return [str(a) for a in args] if isinstance(args, list) and args else None
    except ValueError:
        return None


def ffmpeg_bin() -> str:
    """SONYA_FFMPEG lets Desktop point at a build with libass (subtitles)."""
    return os.environ.get("SONYA_FFMPEG") or "ffmpeg"


def _glass_graph(vf: str, mask_name: str) -> str:
    """Liquid-glass plates under the subtitles: the finished frame is blurred (at 1/4 size: the
    blur is the same to the eye and 16× cheaper), lifted and slightly saturated, then laid back
    over the frame through the pill mask (`mask_name`, drawn by libass from the .glass.json)."""
    return (f"{vf},split=3[gbase][gsrc][gm0];"
            f"[gsrc]scale=iw/4:ih/4,gblur=sigma=6,eq=brightness=0.10:saturation=1.25:contrast=0.92,"
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=bicubic,format=yuv420p[gbl];"
            f"[gm0]lutyuv=y=16:u=128:v=128,ass=filename={mask_name},format=gray[gmask];"
            f"[gbl][gmask]alphamerge[gp];[gbase][gp]overlay=0:0:format=yuv420")


def render_tracked(input_video_path: str, output_path: str, start_time: float, duration: float,
                   track: Dict[str, Any], encoder_args: Optional[List[str]] = None,
                   subtitle_path: Optional[str] = None, report: Optional[Dict[str, Any]] = None) -> str:
    """Single ffmpeg pass: accurate seek → crop along the track → 1080x1920
    [→ burn-in ASS subtitles, same pass]. A failed / hung NVENC encode is re-run on libx264 (cut_profile.run_encode);
    ``report`` (if given) gets render_encoder ("nvenc" | "cpu" | "nvenc->cpu") and render_s."""
    if track.get("layout") == "game":
        cx, cy, cw_, ch_ = track["cam"]
        gx, gy, gw, gh = track["game"]
        vf = (f"split=2[gc][gg];[gc]crop={cw_}:{ch_}:{cx}:{cy},scale={OUTPUT_WIDTH}:960:flags=lanczos,setsar=1[gtop];"
              f"[gg]crop={gw}:{gh}:{gx}:{gy},scale={OUTPUT_WIDTH}:960:flags=lanczos,setsar=1[gbot];"
              f"[gtop][gbot]vstack=inputs=2,drawbox=x=0:y=957:w=iw:h=6:color=black:t=fill")
    elif track.get("layout") == "fit" and track.get("background") == "blur":
        vf = (f"split=2[bg0][fg0];"
              f"[bg0]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
              f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},gblur=sigma=28,eq=brightness=-0.12:saturation=0.85[bg];"
              f"[fg0]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    elif track.get("layout") == "fit":
        vf = (f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,"
              f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    else:
        x = x_expression(track["keyframes"]).replace(",", r"\,")
        vf = (f"crop=w={track['crop_w']}:h={track['crop_h']}:x='{x}':y={track['y']},"
              f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos,setsar=1")
    cu = track.get("closeup")
    if cu:
        ex = lambda k: x_expression(k).replace(",", r"\,")          # noqa: E731
        enable = "+".join(f"between(t\\,{a:.3f}\\,{b - 0.001:.3f})" for a, b in cu["intervals"])
        top = CLOSEUP_TOP
        vf = (f"split=3[c0][c1][c2];[c0]{vf}[cbase];"
              f"[c1]crop=w={cu['size']}:h={cu['size']}:x='{ex(cu['x'])}':y={cu['y']},"
              f"scale={OUTPUT_WIDTH}:{OUTPUT_WIDTH}:flags=lanczos,setsar=1[csq];"
              f"[c2]scale=-2:{OUTPUT_HEIGHT},crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},gblur=sigma=30,"
              f"eq=brightness=-0.15:saturation=0.85,setsar=1[cbg];"
              f"[cbg][csq]overlay=0:{top}[cwide];[cbase][cwide]overlay=0:0:enable='{enable}'")
    sp = track.get("split")
    if sp:
        # split screen during two-person shots: [single 9:16] with [top / bottom] overlaid
        half_h = OUTPUT_HEIGHT // 2
        ex = lambda k: x_expression(k).replace(",", r"\,")          # noqa: E731
        enable = "+".join(f"between(t\\,{a:.3f}\\,{b - 0.001:.3f})" for a, b in sp["intervals"])
        parts = [f"split=3[s0][s1][s2]",
                 f"[s0]{vf}[single]" if not cu else f"[s0]null[s0n];[s0n]{vf}[single]"]
        for i, name in ((1, "top"), (2, "bottom")):
            parts.append(f"[s{i}]crop=w={sp['w']}:h={sp['h']}:x='{ex(sp[name]['x'])}':y='{ex(sp[name]['y'])}',"
                         f"scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos,setsar=1[{name}]")
        parts.append(f"[top][bottom]vstack=inputs=2,drawbox=x=0:y={half_h - 3}:w=iw:h=6:color=black:t=fill[pair]")
        parts.append(f"[single][pair]overlay=0:0:enable='{enable}'")
        vf = ";".join(parts)
    cwd = None
    # ffmpeg runs from the subtitle directory (below): relative source / output paths would point there
    input_video_path, output_path = os.path.abspath(input_video_path), os.path.abspath(output_path)
    if subtitle_path:
        # Reference the .ass by its bare name from its own directory: project
        # paths may contain ', :, spaces… which filtergraph quoting can't express
        # reliably. SONYA writes subtitle names itself (viral_clip_NN.ass).
        sub = Path(subtitle_path).resolve()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", sub.name):
            raise ValueError(f"unsafe subtitle file name: {sub.name}")
        cwd = str(sub.parent)
        from scripts.shared.subtitle_styles import FONTS_DIR    # bundled Montserrat (OFL), Cyrillic
        fd = str(FONTS_DIR).replace("\\", "/").replace(":", "\\:").replace("'", "")
        glass = sub.with_suffix(".glass.json")
        if glass.is_file():
            from scripts.shared.subtitle_styles import glass_mask_ass
            mask = sub.with_suffix(".glassmask.ass")
            mask.write_text(glass_mask_ass(json.loads(glass.read_text(encoding="utf-8"))), encoding="utf-8")
            vf = _glass_graph(vf, mask.name)
        vf += f",ass=filename={sub.name}:fontsdir='{fd}'"
    from scripts.shared.cut_profile import encode_timeout, run_encode
    make = lambda enc: [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",          # noqa: E731
                        "-ss", f"{max(0.0, start_time):.3f}", "-i", input_video_path, "-t", f"{duration:.3f}",
                        "-vf", vf, *enc,
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path]
    enc = encoder_args or _env_encoder_args() or ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]
    result, used, secs = run_encode(make, enc, timeout=encode_timeout(duration), cwd=cwd, what=Path(output_path).name)
    if report is not None:
        report.update(render_encoder=used, render_s=secs)
    if result.returncode != 0:
        raise RuntimeError(f"tracked crop failed: {result.stderr[-500:]}")
    return output_path
