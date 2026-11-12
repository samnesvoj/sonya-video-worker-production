"""Split-screen «attention» layout shared by the streamer and virality modes (owner 2026-11-10).

Top (1080 x top_h, 1000..1150): the clip itself (streamer: webcam/face only; virality: the viral cut). Bottom
(1080 x 1920-top_h): a silent filler loop (bottom_filler = gameplay | asmr) from our own library. Subtitles sit on the seam.

RIGHTS: only library entries whose manifest row has an allowed licence + sha256 + no_watermark are ever used; the file's sha256 is
re-checked before use. Nothing is taken from third-party channels. Library: release/fillers.manifest.json, files in
$FILLERS_LOCAL_DIR (default /opt/sonya/assets/fillers; S3 worker-v2/assets/fillers/).
Stdlib only (ffmpeg via subprocess)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

W, H = 1080, 1920
TOP_MIN, TOP_MAX, TOP_DEFAULT = 1000, 1150, 1100
MODES = ("none", "gameplay", "asmr")
ALLOWED_LICENSES = {"own-generated", "CC0", "Pexels-commercial", "Pixabay-commercial"}
MANIFEST = Path(__file__).resolve().parents[2] / "release" / "fillers.manifest.json"
TOP_FOCUS = 0.15            # share of the crop slack taken from the top (faces sit in the upper part of a 9:16 crop)


def normalize(params: Optional[Mapping[str, Any]], settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """params / site settings -> {"mode", "filler_id", "top_h"}. Unknown mode -> none; top_h clamped to 1000..1150 and even."""
    params, settings = params or {}, settings or {}
    mode = str(params.get("bottom_filler", settings.get("bottom_filler", "none")) or "none").strip().lower()
    if mode not in MODES:
        mode = "none"
    fid = params.get("filler_id", settings.get("filler_id"))
    try:
        top = int(params.get("filler_top_h", settings.get("filler_top_h", TOP_DEFAULT)))
    except (TypeError, ValueError):
        top = TOP_DEFAULT
    top = max(TOP_MIN, min(TOP_MAX, top)) // 2 * 2
    return {"mode": mode, "filler_id": str(fid) if fid else None, "top_h": top}


def load_library(root: Optional[str] = None, manifest: Path = MANIFEST) -> List[Dict[str, Any]]:
    """Usable entries only: allowed licence, no watermark, file present, sha256 matches the manifest."""
    root_p = Path(root or os.environ.get("FILLERS_LOCAL_DIR") or "/opt/sonya/assets/fillers")
    try:
        rows = json.loads(Path(manifest).read_text())["items"]
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for r in rows:
        p = root_p / r["file"]
        if r.get("license") not in ALLOWED_LICENSES or r.get("watermark") or not p.is_file():
            continue
        if _sha(p) != r.get("sha256"):
            continue
        out.append(dict(r, path=str(p)))
    return out


_sha_cache: Dict[Tuple[str, int, int], str] = {}


def _sha(p: Path) -> str:
    st = p.stat()
    k = (str(p), st.st_size, int(st.st_mtime))
    if k not in _sha_cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        _sha_cache[k] = h.hexdigest()
    return _sha_cache[k]


def choose(library: List[Dict[str, Any]], mode: str, filler_id: Optional[str] = None, seed: Any = 0) -> Optional[Dict[str, Any]]:
    """The requested id if it is usable AND of the requested kind; else a seeded random pick of that kind; None if nothing fits."""
    pool = [e for e in library if e.get("kind") == mode]
    if filler_id:
        for e in pool:
            if e["id"] == filler_id:
                return e
    if not pool:
        return None
    return random.Random(f"{seed}").choice(sorted(pool, key=lambda e: e["id"]))


def segment_plan(filler_dur: float, need: float, seed: Any = 0) -> Dict[str, float]:
    """Random start; if the filler is shorter than the clip it loops (ffmpeg -stream_loop) from that start."""
    rnd = random.Random(f"{seed}:seg")
    if filler_dur <= 0:
        raise ValueError("filler has no duration")
    if filler_dur > need + 0.5:
        return {"ss": round(rnd.uniform(0, filler_dur - need), 3), "loops": 0}
    return {"ss": round(rnd.uniform(0, max(0.0, filler_dur - 0.5)), 3), "loops": int(need // filler_dur) + 1}


def seam_intervals(out_dur: float, top_h: int) -> List[List[float]]:
    """`centered` argument of subtitles_for_clip: captions centred on the seam for the whole clip."""
    return [[0.0, round(out_dur, 3), int(top_h)]]


def top_filter(top_h: int, focus: float = TOP_FOCUS) -> str:
    """1080x1920 portrait render -> 1080 x top_h (cropped, upper-biased); anything else is scale-to-fill then cropped."""
    return (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"crop={W}:{top_h}:0:{int((H - top_h) * focus)}")


def build_cmd(ffmpeg: str, top_clip: str, filler: str, out: str, top_h: int, need: float, plan: Dict[str, float],
              ass: Optional[str] = None, enc: Optional[List[str]] = None, top_vf: Optional[str] = None) -> List[str]:
    bot_h = H - top_h
    pre = ["-stream_loop", str(plan["loops"])] if plan.get("loops") else []
    g = (f"[0:v]{top_vf or top_filter(top_h)}[t];"
         f"[1:v]scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h},setsar=1,fps=30[b];"
         f"[t]setsar=1,fps=30[t2];[t2][b]vstack=inputs=2[v0]")
    if ass:
        esc = str(ass).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")      # option level
        for ch in "\\'[],;":                                                      # filtergraph level (no quotes)
            esc = esc.replace(ch, "\\" + ch)
        g += f";[v0]ass=filename={esc}[v]"
    else:
        g += ";[v0]null[v]"
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", top_clip, *pre, "-ss", str(plan["ss"]), "-i", filler,
            "-filter_complex", g, "-map", "[v]", "-map", "0:a?", "-t", f"{need:.3f}",   # filler's own audio is never mapped
            *(enc or ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]


def _probe_dur(ffprobe: str, path: str) -> float:
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def apply(top_clip: str, out: str, cfg: Mapping[str, Any], *, seed: Any = 0, ass: Optional[str] = None,
          report: Optional[Dict[str, Any]] = None, library: Optional[List[Dict[str, Any]]] = None,
          ffmpeg: Optional[str] = None, enc: Optional[List[str]] = None, top_vf: Optional[str] = None) -> bool:
    """Compose top_clip (a finished 1080x1920 render WITHOUT burned subtitles) over a filler. Returns False (report says why) when
    there is no usable filler: the caller then keeps its normal layout. Never raises for missing assets."""
    ffmpeg = ffmpeg or os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    report = report if report is not None else {}
    lib = library if library is not None else load_library()
    f = choose(lib, cfg["mode"], cfg.get("filler_id"), seed)
    if not f:
        report["bottom_filler"] = "no_usable_filler"
        return False
    ffprobe = str(Path(ffmpeg).with_name("ffprobe")) if os.sep in ffmpeg else "ffprobe"
    need = _probe_dur(ffprobe, top_clip)
    plan = segment_plan(_probe_dur(ffprobe, f["path"]), need, seed)
    r = subprocess.run(build_cmd(ffmpeg, top_clip, f["path"], out, cfg["top_h"], need, plan, ass, enc, top_vf), capture_output=True, text=True)
    if r.returncode != 0 or not Path(out).is_file():
        report["bottom_filler"] = f"ffmpeg_failed: {r.stderr[-200:]}"
        return False
    report.update(bottom_filler=cfg["mode"], filler_id=f["id"], filler_license=f["license"], filler_sha256=f["sha256"],
                  filler_ss=plan["ss"], filler_loops=plan["loops"], filler_top_h=cfg["top_h"])
    return True
