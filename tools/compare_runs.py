#!/usr/bin/env python3
"""
compare_runs.py - compare the output of a mode on a CONTROL video between the current tree and a baseline git ref
(the previous release), so a regression is visible before a PR is opened. Videos stay out of Git: only the small
report.json / report.md are meant to be attached to the PR.

  python tools/compare_runs.py --mode virality --input ~/controls/ctrl01.mp4 --baseline worktree-ref [--params p.json] [--out results/NAME]
  python tools/compare_runs.py --runner-cmd 'python my_runner.py {input} {out}' ...   (custom runner; tests use this)

For each output video: duration, resolution, fps, codecs, size, bitrate; SSIM (ffmpeg) when frame size & duration match; JSON outputs
are diffed key by key. Verdict flags: output count change, duration delta > 2 s, resolution change, SSIM < 0.90, missing audio.
Baseline runs inside a temporary `git worktree` (nothing in your working tree is touched).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}


def ffprobe(path: pathlib.Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    return {"duration": round(float(d["format"].get("duration", 0)), 3), "size_bytes": int(d["format"].get("size", 0)),
            "bitrate": int(d["format"].get("bit_rate", 0) or 0), "width": v.get("width"), "height": v.get("height"),
            "fps": round(float(num) / float(den or 1), 2) if float(den or 1) else 0, "vcodec": v.get("codec_name"),
            "acodec": a.get("codec_name") if a else None, "has_audio": a is not None}


def ssim(a: pathlib.Path, b: pathlib.Path) -> float | None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(a), "-i", str(b), "-lavfi", "ssim", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"All:([0-9.]+)", r.stderr)
    return float(m.group(1)) if m else None


def json_diff(a, b, path="") -> list[str]:
    if type(a) is not type(b):
        return [f"{path or '/'}: type {type(a).__name__} -> {type(b).__name__}"]
    if isinstance(a, dict):
        out = [f"{path}/{k}: removed" for k in a.keys() - b.keys()] + [f"{path}/{k}: added" for k in b.keys() - a.keys()]
        for k in a.keys() & b.keys():
            out += json_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        out = [f"{path}: length {len(a)} -> {len(b)}"] if len(a) != len(b) else []
        for i, (x, y) in enumerate(zip(a, b)):
            out += json_diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {str(a)[:60]!r} -> {str(b)[:60]!r}"]


def collect(out_dir: pathlib.Path) -> dict:
    vids = {p.name: p for p in sorted(out_dir.rglob("*")) if p.suffix.lower() in VIDEO_EXT}
    jsons = {p.name: p for p in sorted(out_dir.rglob("*.json"))}
    return {"videos": vids, "jsons": jsons}


def compare(base_dir: pathlib.Path, new_dir: pathlib.Path) -> dict:
    b, n = collect(base_dir), collect(new_dir)
    rep = {"videos": [], "json_diffs": {}, "flags": []}
    if len(b["videos"]) != len(n["videos"]):
        rep["flags"].append(f"output video count {len(b['videos'])} -> {len(n['videos'])}")
    for (bn, bp), (nn, np_) in zip(b["videos"].items(), n["videos"].items()):
        mb, mn = ffprobe(bp), ffprobe(np_)
        row = {"baseline": bn, "new": nn, "baseline_meta": mb, "new_meta": mn,
               "ssim": ssim(bp, np_) if (mb["width"], mb["height"]) == (mn["width"], mn["height"]) and abs(mb["duration"] - mn["duration"]) < 0.5 else None}
        rep["videos"].append(row)
        if abs(mb["duration"] - mn["duration"]) > 2:
            rep["flags"].append(f"{nn}: duration {mb['duration']}s -> {mn['duration']}s")
        if (mb["width"], mb["height"]) != (mn["width"], mn["height"]):
            rep["flags"].append(f"{nn}: resolution {mb['width']}x{mb['height']} -> {mn['width']}x{mn['height']}")
        if mb["has_audio"] and not mn["has_audio"]:
            rep["flags"].append(f"{nn}: audio track lost")
        if row["ssim"] is not None and row["ssim"] < 0.90:
            rep["flags"].append(f"{nn}: SSIM {row['ssim']:.3f} < 0.90 (visual change)")
    for name in sorted(b["jsons"].keys() & n["jsons"].keys()):
        try:
            d = json_diff(json.loads(b["jsons"][name].read_text()), json.loads(n["jsons"][name].read_text()))
        except ValueError:
            continue
        if d:
            rep["json_diffs"][name] = d[:50]
    rep["verdict"] = "REVIEW" if rep["flags"] else "NO_VISIBLE_REGRESSION_IN_THESE_CHECKS"
    return rep


def render_md(rep: dict, meta: dict) -> str:
    L = [f"# Comparison: {meta.get('mode')} on {meta.get('input_name')}", f"baseline `{meta.get('baseline')}` vs current `{meta.get('current')}`", f"**Verdict: {rep['verdict']}** (automatic checks only - watch the videos)", ""]
    L += [f"- ⚠️ {f}" for f in rep["flags"]] or ["- no flags"]
    L += ["", "| output | duration | resolution | fps | size MB | SSIM |", "|---|---|---|---|---|---|"]
    for r in rep["videos"]:
        b, n = r["baseline_meta"], r["new_meta"]
        L.append(f"| {r['new']} | {b['duration']}→{n['duration']} | {b['width']}x{b['height']}→{n['width']}x{n['height']} | {b['fps']}→{n['fps']} | {b['size_bytes']/1e6:.1f}→{n['size_bytes']/1e6:.1f} | {r['ssim'] if r['ssim'] is not None else 'n/a'} |")
    for k, v in rep["json_diffs"].items():
        L += ["", f"**{k}** changed:", *[f"- {x}" for x in v[:15]]]
    return "\n".join(L) + "\n"


def run_tree(tree: pathlib.Path, runner_cmd: str | None, mode: str, inp: pathlib.Path, params: str | None, out: pathlib.Path) -> float:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if runner_cmd:
        cmd = runner_cmd.format(input=str(inp), out=str(out), params=params or "")
        subprocess.run(cmd, shell=True, cwd=tree, check=True)   # operator-supplied command (your own shell line), run inside the tree
    else:
        code = ("import json,sys; from scripts.mode_registry import get_runner; p=json.load(open(sys.argv[3])) if sys.argv[3] else {};"
                "r=get_runner(sys.argv[1])(input_video_path=sys.argv[2], output_dir=sys.argv[4], params=p, progress_callback=lambda s,x: None);"
                "print(json.dumps({'keys': sorted(r) if isinstance(r, dict) else None}))")
        env = {**__import__("os").environ, "ENABLED_MODES": mode, "PYTHONPATH": str(tree), "MODELS_LOCAL_DIR": __import__("os").environ.get("MODELS_LOCAL_DIR", str(pathlib.Path.home() / "SONYA-models"))}
        subprocess.run([sys.executable, "-c", code, mode, str(inp), params or "", str(out)], cwd=tree, env=env, check=True)
    return round(time.time() - t0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True); ap.add_argument("--input", required=True, type=pathlib.Path)
    ap.add_argument("--baseline", default="origin/main", help="git ref of the previous version (default origin/main)")
    ap.add_argument("--params"); ap.add_argument("--runner-cmd"); ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.mode.startswith("trailer"):
        sys.exit("Trailer is disabled")
    out = a.out or ROOT / "results" / f"{a.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    wt = pathlib.Path(tempfile.mkdtemp(prefix="sonya-baseline-")) / "tree"
    subprocess.check_call(["git", "worktree", "add", "--detach", "-q", str(wt), a.baseline], cwd=ROOT)
    try:
        tb = run_tree(wt, a.runner_cmd, a.mode, a.input, a.params, out / "baseline")
        tn = run_tree(ROOT, a.runner_cmd, a.mode, a.input, a.params, out / "new")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT)
        shutil.rmtree(wt.parent, ignore_errors=True)
    rep = compare(out / "baseline", out / "new")
    meta = {"mode": a.mode, "input_name": a.input.name, "baseline": a.baseline, "current": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "seconds": {"baseline": tb, "new": tn}}
    (out / "report.json").write_text(json.dumps({**meta, **rep}, indent=1, default=str))
    (out / "report.md").write_text(render_md(rep, meta))
    print((out / "report.md").read_text())
    print(f"report: {out}/report.md  (attach report.md/report.json to the PR; keep the videos out of Git)")
    return 0 if rep["verdict"] != "REVIEW" else 3


if __name__ == "__main__":
    sys.exit(main())
