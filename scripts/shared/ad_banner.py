"""SONYA mini ad banner over a finished clip (owner 2026-11-15, per user: params["ad_banner"] = "mini" | "mini_light").

The API puts ``ad_banner`` into the job params for the users listed in its config (only the admin account «maloy» for now);
the worker calls ``apply_to_outputs`` after the mode rendered its clips and before previews / upload.

Where: BELOW the subtitles — between the lowest subtitle line and the bottom of the frame, centred, small (~56 % of the
width on 9:16), with a safe margin at the bottom. The subtitle lines come from the clip's own burned-in ASS file
(``<clip>.ass``: exact text, position and time of every cue, incl. lines moved off burned-in text); faces from YuNet on the
rendered frames. Not enough room under the text / a face that low -> a smaller card (down to ``MIN_SHARE`` of the full
size); still no room -> another moment; never covers text or a face: no room at all -> the clip stays as it is
(report ``status=skipped``). Nothing here can fail a job.

When: appears at ~3-5 s of the clip, on screen 4.5 s (the asset), fades + slides in, fades out (no sound).

Asset: own render (sonya-creative-pipeline campaigns/ad-mini, ProRes 4444 with alpha) in S3 worker-v2/assets/ad-mini/,
pinned by sha256 in release/ad_banner.manifest.json, cached under $MODELS_LOCAL_DIR/assets/ad-mini/.
Kill switch for every job: SONYA_AD_BANNER=off.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VERSION = "ad-banner-mini-v1"
MANIFEST = Path(__file__).resolve().parents[2] / "release" / "ad_banner.manifest.json"
KINDS = {"mini": "dark", "mini_dark": "dark", "mini_light": "light"}
MODES = ("virality", "hooks", "beep", "educational", "streamer")

# geometry of the asset (reel/src/ad/AdMini.tsx): canvas 1072x592, card 960x480 inside a 56 px shadow margin
CANVAS_W, CANVAS_H, CARD_W, CARD_H, PAD = 1072, 592, 960, 480, 56
BANNER_SEC = 4.5
FADE, SLIDE = 0.3, 0.35
STARTS = (3.0, 3.5, 4.0, 4.5, 5.0)  # owner: «появление ~на 3-5 с клипа»
WIDTH_SHARE = 0.56                  # card width on 9:16 (owner: ~50-60 %)
HEIGHT_SHARE = 0.36                 # … at most this share of the frame HEIGHT as width (landscape stays small)
MIN_SHARE = 0.65                    # smallest card = 65 % of the normal one (≈ 36 % of a 9:16 frame)
BOTTOM_MARGIN = 0.025               # of H: safe margin under the card (48 px on 1920)
TEXT_GAP = 0.015                    # of H: between the lowest subtitle line and the card (29 px on 1920)
FACE_GAP = 0.01
FACE_FPS = 2.0


def kind_of(params: Optional[Mapping[str, Any]]) -> Optional[str]:
    """params["ad_banner"] -> theme ("dark" | "light") or None (off / unknown value / global kill switch)."""
    if str(os.environ.get("SONYA_AD_BANNER", "")).strip().lower() in ("0", "off", "false", "no"):
        return None
    raw = (params or {}).get("ad_banner")
    if isinstance(raw, dict):
        raw = raw.get("kind")
    return KINDS.get(str(raw or "").strip().lower())


# ── asset ────────────────────────────────────────────────────────────────────

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _entry(theme: str, manifest: Path = MANIFEST) -> Dict[str, Any]:
    items = json.loads(Path(manifest).read_text(encoding="utf-8"))["items"]
    for it in items:
        if it.get("theme") == theme:
            return it
    raise KeyError(f"no ad banner asset for theme {theme}")


def ensure_asset(theme: str, *, manifest: Path = MANIFEST, download=None) -> Path:
    """Local, sha256-verified copy of the banner (downloaded once per server from S3)."""
    override = os.environ.get("SONYA_AD_BANNER_ASSET")       # local tests only
    if override:
        return Path(override)
    it = _entry(theme, manifest)
    root = Path(os.environ.get("SONYA_ASSETS_DIR")
                or Path(os.environ.get("MODELS_LOCAL_DIR", "/opt/sonya/models")) / "assets") / "ad-mini"
    dst = root / it["file"]
    ok = dst.with_name(dst.name + ".verified")
    if dst.is_file() and ok.is_file() and ok.read_text().strip() == f"{it['sha256']} {dst.stat().st_size}":
        return dst
    if dst.is_file() and _sha(dst) == it["sha256"]:
        ok.write_text(f"{it['sha256']} {dst.stat().st_size}")
        return dst
    root.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(f"{dst.name}.{os.getpid()}.part")
    if download is None:
        from scripts.prod_s3_storage import download_file as download
    download(it["s3_key"], part)
    got = _sha(part)
    if got != it["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ad banner asset sha256 mismatch ({got[:12]} != {it['sha256'][:12]})")
    os.replace(part, dst)                                     # atomic: a parallel worker sees the old or the full file
    ok.write_text(f"{it['sha256']} {dst.stat().st_size}")
    return dst


# ── what is on screen ────────────────────────────────────────────────────────

def _secs(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def subtitle_bands(ass_text: str, H: int) -> List[Tuple[float, float, float, float]]:
    """[(t0, t1, top, bottom)] in output px for every Dialogue line of a burned-in ASS file (subtitles, hook title, promo
    plate). Libass placement as in scripts/shared/subtitle_styles.glass_pills: default = the style's alignment + MarginV,
    \\an2/\\an5/\\an8 + \\pos overrides. Height = rows x font size x 1.3 (+ outline): a slight over-estimate on purpose."""
    play_h = H
    styles: Dict[str, Tuple[float, int, int, float]] = {}       # name -> (size, alignment, margin_v, outline)
    fmt: List[str] = []
    out = []
    for ln in ass_text.splitlines():
        if ln.startswith("PlayResY:"):
            try:
                play_h = int(ln.split(":", 1)[1])
            except ValueError:
                pass
        elif ln.startswith("Format:") and not fmt:
            fmt = [x.strip() for x in ln.split(":", 1)[1].split(",")]
        elif ln.startswith("Style:") and fmt:
            v = [x.strip() for x in ln.split(":", 1)[1].split(",")]
            d = dict(zip(fmt, v))
            try:
                styles[d["Name"]] = (float(d["Fontsize"]), int(d["Alignment"]), int(d["MarginV"]), float(d.get("Outline", 0)))
            except (KeyError, ValueError):
                continue
        elif ln.startswith("Dialogue:"):
            f = ln.split(":", 1)[1].split(",", 9)
            if len(f) < 10:
                continue
            size, align, margin_v, outline = styles.get(f[3].strip(), (80.0, 2, 380, 4.0))
            text = f[9]
            if r"\p1" in text:                                  # vector drawing (glass mask): not text
                continue
            m = re.search(r"\\fs(\d+(?:\.\d+)?)", text)
            if m:
                size = float(m.group(1))
            fscy = [float(x) for x in re.findall(r"\\fscy(\d+(?:\.\d+)?)", text)]
            grow = max(fscy) / 100.0 if fscy else 1.0          # spoken word grows (112 %), long line scaled down (< 100)
            rows = len(re.sub(r"\{[^}]*\}", "", text).split("\\N"))
            band = rows * size * 1.3 * grow + 2 * outline
            an = re.search(r"\\an(\d)", text)
            a = int(an.group(1)) if an else align
            pos = re.search(r"\\pos\(\s*[\d.]+\s*,\s*([\d.]+)\s*\)", text)
            y = float(pos.group(1)) if pos else None
            row = (a - 1) // 3                                  # 0 bottom, 1 middle, 2 top (numpad)
            if y is None:
                y = {0: play_h - margin_v, 1: play_h / 2, 2: margin_v}[row]
            top = {0: y - band, 1: y - band / 2, 2: y}[row]
            k = H / float(play_h)
            out.append((_secs(f[1].strip()), _secs(f[2].strip()), top * k, (top + band) * k))
    return out


def face_boxes(clip: str, t0: float, t1: float, fps: float = FACE_FPS) -> List[Tuple[float, List[Tuple[float, float, float, float]]]]:
    """[(t, [(x, y, w, h), …])] in output px, YuNet (the crop's own detector) on frames every 1/fps s of [t0, t1].
    No OpenCV / model -> [] (the text rule still holds)."""
    try:
        import cv2
        from scripts.shared.crop.clip_analysis import MODEL_PATH, SCORE_MIN
    except Exception as exc:                                     # pragma: no cover - worker image always has both
        logger.warning("[ad_banner] face check unavailable: %s", type(exc).__name__)
        return []
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return []
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = min(1.0, 640.0 / max(W, 1))
    dw, dh = int(round(W * k)), int(round(H * k))
    det = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (dw, dh), SCORE_MIN, 0.3, 50)
    out = []
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0) * 1000.0)
        nxt = t0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > t1 + 0.05:
                break
            if t + 1e-3 < nxt:
                continue
            nxt = t + 1.0 / fps
            small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if k < 1 else frame
            _, faces = det.detect(small)
            boxes = [] if faces is None else [(float(f[0]) / k, float(f[1]) / k, float(f[2]) / k, float(f[3]) / k) for f in faces]
            out.append((t, boxes))
    finally:
        cap.release()
    return out


# ── where and when ──────────────────────────────────────────────────────────

def card_limits(W: int, H: int) -> Tuple[int, int]:
    """(normal, smallest) card WIDTH in px."""
    full = int(min(WIDTH_SHARE * W, HEIGHT_SHARE * H)) // 2 * 2
    return full, int(full * MIN_SHARE) // 2 * 2


def plan(W: int, H: int, duration: float, bands: Sequence[Tuple[float, float, float, float]],
         faces: Sequence[Tuple[float, Sequence[Tuple[float, float, float, float]]]] = (),
         starts: Sequence[float] = STARTS) -> Dict[str, Any]:
    """Best moment + size: {"status": "ok", "at", "card_w", "card_h", "card_x", "card_y"} or {"status": "skipped", "reason"}.
    The card is centred and sits on the bottom margin; its height is what is left between the lowest text line / face
    of the window (+ gap) and that margin, capped at the normal size. The biggest card wins, then the earliest moment."""
    full_w, min_w = card_limits(W, H)
    bottom = H - round(BOTTOM_MARGIN * H)
    gap_t, gap_f = TEXT_GAP * H, FACE_GAP * H
    ats = [a for a in starts if a + BANNER_SEC + 0.3 <= duration]
    if not ats:
        if duration < BANNER_SEC + 1.0:
            return {"status": "skipped", "reason": f"clip_too_short ({duration:.1f} s)"}
        ats = [max(0.5, duration - BANNER_SEC - 0.3)]
    best = None
    why = "no_room_below_subtitles"
    for at in ats:
        end = at + BANNER_SEC
        lowest = 0.0
        for t0, t1, top, bot in bands:
            if t1 > at and t0 < end and top < bottom:          # a line on screen during the banner, above its bottom
                lowest = max(lowest, bot + gap_t)
        face_low = 0.0
        x0 = (W - full_w) / 2
        for t, boxes in faces:
            if not (at - 0.1 <= t <= end + 0.1):
                continue
            for fx, fy, fw, fh in boxes:
                if fx + fw < x0 or fx > W - x0:                 # beside the card column
                    continue
                face_low = max(face_low, fy + 1.15 * fh + gap_f)   # + chin
        room = bottom - max(lowest, face_low)
        card_h = min(full_w / 2.0, room)
        card_w = int(card_h * 2) // 2 * 2
        if card_w < min_w:
            why = "face_low_in_frame" if face_low > lowest else "no_room_below_subtitles"
            continue
        if best is None or card_w > best["card_w"] + 1:
            best = {"at": round(at, 3), "card_w": card_w}
    if best is None:
        return {"status": "skipped", "reason": why}
    card_h = best["card_w"] // 2
    return {"status": "ok", "at": best["at"], "card_w": best["card_w"], "card_h": card_h,
            "card_x": (W - best["card_w"]) // 2, "card_y": bottom - card_h, "full_w": full_w}


def overlay_filter(p: Dict[str, Any], fps: str) -> str:
    """Banner canvas scaled so the card is card_w wide; alpha fade in/out; eased slide up by ~3 % of the card's height x 6."""
    s = p["card_w"] / float(CARD_W)
    bw = int(round(CANVAS_W * s / 2)) * 2
    bh = int(round(CANVAS_H * s / 2)) * 2
    x = p["card_x"] - int(round(PAD * s))
    y = p["card_y"] - int(round(PAD * s))
    at, end = p["at"], p["at"] + BANNER_SEC
    dy = max(8, int(round(0.18 * p["card_h"])))
    yexpr = (f"{y}+{dy}*pow(max(0\\,1-(t-{at:.3f})/{SLIDE}),3)"
             f"+{dy // 2}*pow(max(0\\,(t-{end - FADE:.3f})/{FADE}),2)")
    return (f"[1:v]fps={fps},scale={bw}:{bh}:flags=lanczos,format=yuva444p,"
            f"fade=t=in:st=0:d={FADE}:alpha=1,fade=t=out:st={BANNER_SEC - FADE}:d={FADE}:alpha=1,"
            f"setpts=PTS-STARTPTS+{at:.3f}/TB[b];"
            f"[0:v][b]overlay=x={x}:y='{yexpr}':eval=frame:eof_action=pass:format=auto,format=yuv420p[v]")


def _probe(path: str) -> Dict[str, Any]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,r_frame_rate,color_primaries,color_transfer,color_space",
                          "-of", "json", path], capture_output=True, text=True, timeout=60, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": v.get("r_frame_rate") or "30/1",
            "dur": float(j["format"]["duration"]), "audio": any(s["codec_type"] == "audio" for s in j["streams"]),
            "colors": {k: v.get(k) for k in ("color_primaries", "color_transfer", "color_space")}}


def apply(clip: str, asset: str, *, ass_path: Optional[str] = None, ffmpeg: str = "ffmpeg") -> Dict[str, Any]:
    """Overlay the banner onto ``clip`` in place (re-encode with the render's encoder, audio copied). -> report."""
    from scripts.shared.cut_profile import encode_timeout, encoder_args, encoder_family, run_encode
    info = _probe(clip)
    W, H, dur = info["w"], info["h"], info["dur"]
    bands = []
    if ass_path and Path(ass_path).is_file():
        bands = subtitle_bands(Path(ass_path).read_text(encoding="utf-8", errors="replace"), H)
    ats = [a for a in STARTS if a + BANNER_SEC + 0.3 <= dur] or [max(0.5, dur - BANNER_SEC - 0.3)]
    faces = face_boxes(clip, min(ats) - 0.1, max(ats) + BANNER_SEC + 0.1)
    p = plan(W, H, dur, bands, faces)
    p.update(version=VERSION, subtitles_from="ass" if bands else "none", face_frames=len(faces))
    if p["status"] != "ok":
        return p
    tmp = str(Path(clip).with_name(Path(clip).stem + ".ad.mp4"))
    col = [arg for k, flag in (("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"),
                               ("color_space", "-colorspace"))
           if info["colors"].get(k) not in (None, "", "unknown") for arg in (flag, info["colors"][k])]

    def cmd(a: List[str]) -> List[str]:
        c = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", clip, "-i", asset,
             "-filter_complex", overlay_filter(p, info["fps"]), "-map", "[v]"]
        if info["audio"]:
            c += ["-map", "0:a", "-c:a", "copy"]
        return c + list(a) + col + ["-r", info["fps"], "-map_metadata", "0", "-movflags", "+faststart", tmp]

    r, used, sec = run_encode(cmd, encoder_args(encoder_family(ffmpeg)), timeout=encode_timeout(dur), what="ad_banner")
    if r.returncode != 0 or not Path(tmp).is_file() or Path(tmp).stat().st_size == 0:
        Path(tmp).unlink(missing_ok=True)
        return dict(p, status="failed", reason=(r.stderr or "")[-200:].strip())
    os.replace(tmp, clip)
    return dict(p, encoder=used, encode_s=sec)


def _record(output_dir: Path, reports: Dict[str, Dict[str, Any]]) -> None:
    """Add the per-clip report to selected_segments.json (bookkeeping, never fails)."""
    sel = output_dir / "selected_segments.json"
    try:
        doc = json.loads(sel.read_text(encoding="utf-8"))
        for seg in doc.get("segments") or []:
            if seg.get("file") in reports:
                seg["ad_banner"] = reports[seg["file"]]
        sel.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def apply_to_outputs(output_dir: str, mode: str, params: Optional[Mapping[str, Any]],
                     drop_ass: bool = False) -> Dict[str, Dict[str, Any]]:
    """Banner on every finished clip of the job when params ask for it. -> {clip name: report}. Never raises.
    drop_ass: the mode does not upload its subtitle files (streamer / educational keep them only for this step)."""
    out_dir = Path(output_dir)
    clips = sorted(c for c in out_dir.rglob("*.mp4") if not c.stem.endswith(("_preview", ".ad")))
    theme = kind_of(params) if mode in MODES else None
    reports: Dict[str, Dict[str, Any]] = {}
    try:
        if theme and clips:
            try:
                asset = str(ensure_asset(theme))
            except Exception as exc:
                logger.error("[ad_banner] asset unavailable (%s): clips go out without the banner", str(exc)[:160])
                return {c.name: {"status": "failed", "reason": "asset_unavailable"} for c in clips}
            for c in clips:
                ass = c.with_suffix(".ass")
                try:
                    rep = apply(str(c), asset, ass_path=str(ass) if ass.is_file() else None)
                except Exception as exc:                        # never lose a rendered clip over the banner
                    rep = {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
                rep["theme"] = theme
                reports[c.name] = rep
                logger.info("[ad_banner] %s: %s", c.name, {k: rep.get(k) for k in
                                                         ("status", "reason", "at", "card_w", "card_y", "encoder")})
            _record(out_dir, reports)
    finally:
        if drop_ass:
            for c in clips:
                c.with_suffix(".ass").unlink(missing_ok=True)
    return reports
