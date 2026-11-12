#!/usr/bin/env python3
"""
compare_cut.py - is a Cut (virality) output the same as the Mac mini reference?

  python tools/compare_cut.py REF_DIR NEW_DIR [--json out.json] [--ssim]

REF_DIR = a SONYA Desktop export (exports/<run>/), NEW_DIR = the worker's output/ (local or downloaded from S3).
Both hold selected_segments.json, viral_clip_NN.mp4 and viral_clip_NN.ass. Checked per clip:
  * moment:    start / end within --tol-sec (default 0.10 s)
  * crop:      same shot list (count, targets, faces) and keyframe x / layout within --tol-px (default 8 px)
  * subtitles: identical Dialogue text (timing within --tol-sec), same style / font
  * video:     1080x1920, codecs, duration within 0.2 s; optional SSIM (frame-level, ffmpeg) >= 0.90
Exit 0 = match, 1 = differences (printed), 2 = bad input. Different encoders (VideoToolbox vs NVENC) make the
pixels differ slightly, so SSIM is advisory; moments / crop / subtitles are the hard criteria.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List


def _segments(d: pathlib.Path) -> List[Dict[str, Any]]:
    p = d / "selected_segments.json"
    if not p.is_file():
        raise SystemExit(f"no selected_segments.json in {d}")
    data = json.loads(p.read_text(encoding="utf-8"))
    segs = data.get("segments") if isinstance(data, dict) else data
    return [s for s in segs or [] if s.get("output_path") or s.get("output") or s.get("file") or True]


def _clip_name(s: Dict[str, Any], i: int) -> str:
    for k in ("output_path", "output", "file"):
        if s.get(k):
            return pathlib.Path(str(s[k])).name
    return f"viral_clip_{i + 1:02d}.mp4"


def _ass(path: pathlib.Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"present": False, "lines": [], "styles": []}
    lines, styles = [], []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("Dialogue:"):
            parts = ln.split(",", 9)
            text = re.sub(r"\{[^}]*\}", "", parts[9]) if len(parts) > 9 else ""
            lines.append({"start": parts[1].strip(), "end": parts[2].strip(), "style": parts[3].strip(), "text": text.strip()})
        elif ln.startswith("Style:"):
            f = ln[6:].split(",")
            styles.append((f[0].strip(), f[1].strip() if len(f) > 1 else ""))
    return {"present": True, "lines": lines, "styles": styles}


def _nums(x: Any) -> List[float]:
    if isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in _nums(x[k])]
    if isinstance(x, (list, tuple)):
        return [v for y in x for v in _nums(y)]
    return []


def _t(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _probe(path: pathlib.Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {"w": v.get("width"), "h": v.get("height"), "vcodec": v.get("codec_name"), "acodec": a.get("codec_name"),
            "duration": float((j.get("format") or {}).get("duration") or 0)}


def _ssim(a: pathlib.Path, b: pathlib.Path) -> float:
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", str(a), "-i", str(b), "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"All:([0-9.]+)", r.stderr)
    return float(m[-1]) if m else 0.0


def compare(ref: pathlib.Path, new: pathlib.Path, tol_sec: float = 0.10, tol_px: float = 8.0,
            ssim: bool = False) -> Dict[str, Any]:
    rs, ns = _segments(ref), _segments(new)
    diffs: List[str] = []
    clips = []
    if len(rs) != len(ns):
        diffs.append(f"clip count: ref {len(rs)} vs new {len(ns)}")
    # pair by moment (start within 1 s), not by position: one changed pick must not hide that the rest match
    pairs, used = [], set()
    for i, r in enumerate(rs):
        j = next((j for j, n in enumerate(ns) if j not in used
                  and abs(float(n.get("start") or 0) - float(r.get("start") or 0)) <= 1.0), None)
        if j is None:
            diffs.append(f"{_clip_name(r, i)}: moment {r.get('start')}-{r.get('end')} only in ref")
            continue
        used.add(j)
        pairs.append((i, r, j, ns[j]))
    for j, n in enumerate(ns):
        if j not in used:
            diffs.append(f"{_clip_name(n, j)}: moment {n.get('start')}-{n.get('end')} only in new")
    for i, r, j, n in pairs:
        name, nname = _clip_name(r, i), _clip_name(n, j)
        if name != nname:
            diffs.append(f"{name}: is {nname} in new (rank changed)")
        c: Dict[str, Any] = {"clip": name, "ref": [r.get("start"), r.get("end")], "new": [n.get("start"), n.get("end")]}
        for k in ("start", "end"):
            if abs(float(r.get(k) or 0) - float(n.get(k) or 0)) > tol_sec:
                diffs.append(f"{name}: {k} ref {r.get(k)} vs new {n.get(k)}")
        rsh, nsh = r.get("crop_shots") or [], n.get("crop_shots") or []
        if [(s.get("target"), s.get("faces")) for s in rsh] != [(s.get("target"), s.get("faces")) for s in nsh]:
            diffs.append(f"{name}: crop shots differ (ref {len(rsh)} vs new {len(nsh)})")
        rk, nk = r.get("crop_keyframes") or [], n.get("crop_keyframes") or []
        if len(rk) != len(nk):
            diffs.append(f"{name}: crop keyframes ref {len(rk)} vs new {len(nk)}")
        else:
            worst, worst_t = 0.0, 0.0
            for a, b in zip(rk, nk):        # [t, x] (single) or [t, {...}] / dicts (split): flatten numbers
                fa, fb = _nums(a), _nums(b)
                if len(fa) != len(fb):
                    diffs.append(f"{name}: crop keyframe shape ref {a} vs new {b}")
                    break
                if fa:
                    worst_t = max(worst_t, abs(fa[0] - fb[0]))
                    worst = max([worst] + [abs(x - y) for x, y in zip(fa[1:], fb[1:])])
            c["crop_max_delta"] = round(worst, 2)
            if worst > tol_px or worst_t > tol_sec:
                diffs.append(f"{name}: crop keyframes differ by up to {worst:.1f} px / {worst_t:.2f} s")
        for k in ("crop_status", "crop_version", "subtitle_status", "subtitle_burn", "subtitle_style", "subtitle_font"):
            if k in r or k in n:
                if r.get(k) != n.get(k):
                    diffs.append(f"{name}: {k} ref {r.get(k)} vs new {n.get(k)}")
        ra, na = _ass((ref / name).with_suffix(".ass")), _ass((new / nname).with_suffix(".ass"))
        if ra["present"] != na["present"]:
            diffs.append(f"{name}: subtitles file ref {ra['present']} vs new {na['present']}")
        if [x["text"] for x in ra["lines"]] != [x["text"] for x in na["lines"]]:
            rt, nt = [x["text"] for x in ra["lines"]], [x["text"] for x in na["lines"]]
            first = next((j for j, (a, b) in enumerate(zip(rt, nt)) if a != b), min(len(rt), len(nt)))
            diffs.append(f"{name}: subtitle text differs at line {first} "
                         f"(ref {rt[first] if first < len(rt) else '-'!r} vs new {nt[first] if first < len(nt) else '-'!r}); "
                         f"lines ref {len(rt)} vs new {len(nt)}")
        else:
            dt = max([abs(_t(a["start"]) - _t(b["start"])) for a, b in zip(ra["lines"], na["lines"])] or [0.0])
            c["subtitle_lines"], c["subtitle_max_dt"] = len(ra["lines"]), round(dt, 3)
            if dt > tol_sec:
                diffs.append(f"{name}: subtitle timing differs by up to {dt:.2f} s")
        if ra["styles"] != na["styles"]:
            diffs.append(f"{name}: subtitle styles ref {ra['styles']} vs new {na['styles']}")
        rp, np_ = _probe(ref / name), _probe(new / nname)
        c["ref_video"], c["new_video"] = rp, np_
        if not np_:
            diffs.append(f"{name}: missing in new")
        else:
            if (np_.get("w"), np_.get("h")) != (1080, 1920):
                diffs.append(f"{name}: new is {np_.get('w')}x{np_.get('h')}, not 1080x1920")
            if np_.get("vcodec") != "h264" or np_.get("acodec") != "aac":
                diffs.append(f"{name}: new codecs {np_.get('vcodec')}/{np_.get('acodec')}")
            if rp and abs(rp["duration"] - np_["duration"]) > 0.2:
                diffs.append(f"{name}: duration ref {rp['duration']:.2f} vs new {np_['duration']:.2f}")
            if ssim and rp:
                c["ssim"] = round(_ssim(ref / name, new / nname), 4)
        clips.append(c)
    return {"match": not diffs, "differences": diffs, "clips": clips}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref", type=pathlib.Path)
    ap.add_argument("new", type=pathlib.Path)
    ap.add_argument("--tol-sec", type=float, default=0.10)
    ap.add_argument("--tol-px", type=float, default=8.0)
    ap.add_argument("--ssim", action="store_true")
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args(argv)
    if not a.ref.is_dir() or not a.new.is_dir():
        print("both arguments must be directories", file=sys.stderr)
        return 2
    rep = compare(a.ref, a.new, a.tol_sec, a.tol_px, a.ssim)
    if a.json:
        a.json.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in rep["clips"]:
        print(f"{c['clip']}: ref {c['ref']} new {c['new']} crop_delta={c.get('crop_max_delta')} "
              f"subs={c.get('subtitle_lines')} ssim={c.get('ssim')}")
    print("MATCH" if rep["match"] else "DIFFERENT:\n  " + "\n  ".join(rep["differences"]))
    return 0 if rep["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
