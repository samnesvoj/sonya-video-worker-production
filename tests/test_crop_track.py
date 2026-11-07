"""Tracked 9:16 crop from one face analysis (scripts/shared/crop)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from scripts.shared.crop import smart_crop_adapter as sca

W, H = 1920, 1080


def analysis(face_x_at, t_end=30.0, fps=2.0, size=200):
    """face_x_at(t) -> face center x or None (no face)."""
    samples, t = [], 0.0
    while t <= t_end + 1e-9:
        cx = face_x_at(t)
        samples.append({"t": round(t, 3), "faces": [] if cx is None else [[int(cx - size / 2), 300, size, size]]})
        t += 1.0 / fps
    return {"meta": {"version": "faces-haar-v1", "detector": "opencv-haar-frontalface-default",
                     "width": W, "height": H}, "samples": samples}


def eval_expr(expr: str, t: float) -> float:
    """Tiny evaluator for the piecewise expression grammar we emit."""
    def ev(e):
        m = re.match(r"^if\(lt\(t,([\d.]+)\),(.*)\)$", e)
        if not m:
            return eval(e, {"t": t})
        thr, rest = float(m.group(1)), m.group(2)
        depth, i = 0, 0
        for i, ch in enumerate(rest):
            depth += ch == "("
            depth -= ch == ")"
            if ch == "," and depth == 0:
                break
        a, b = rest[:i], rest[i + 1:]
        return ev(a) if t < thr else ev(b)
    return ev(expr)


def test_crop_window_is_9_16_and_even():
    assert sca.crop_window(1920, 1080) == (594, 1056)
    cw, ch = sca.crop_window(1080, 1920)
    assert (cw, ch) == (1080, 1920)
    cw, ch = sca.crop_window(1000, 3000)
    assert cw % 2 == 0 and ch % 2 == 0 and abs(cw / ch - 9 / 16) < 0.01


def test_no_analysis_and_no_subject_fall_back_to_center():
    t = sca.crop_track(None, 0, 10, width=W, height=H)
    assert t["crop_status"] == "center_fallback" and t["fallback_reason"] == "detector_unavailable"
    t2 = sca.crop_track(analysis(lambda t: None), 0, 10)
    assert t2["fallback_reason"] == "no_subject_found"
    cw, _ = sca.crop_window(W, H)
    assert t2["keyframes"][0][1] == (W - cw) // 2 // 2 * 2


def test_steady_subject_single_position_inside_frame():
    tr = sca.crop_track(analysis(lambda t: 1500), 5, 20)
    xs = {x for _, x in tr["keyframes"]}
    assert tr["crop_status"] == "tracked" and len(xs) == 1
    x = xs.pop()
    assert 0 <= x <= W - tr["crop_w"] and abs((x + tr["crop_w"] / 2) - 1500) < 4


def test_edges_are_clamped():
    for cx in (10, W - 5):
        tr = sca.crop_track(analysis(lambda t: cx), 0, 10)
        assert all(0 <= x <= W - tr["crop_w"] for _, x in tr["keyframes"])


def test_single_sample_outliers_do_not_move_the_window():
    tr = sca.crop_track(analysis(lambda t: 1600 if abs(t - 4.0) < 0.1 or abs(t - 9.0) < 0.1 else 600), 0, 15)
    assert len({x for _, x in tr["keyframes"]}) == 1


def test_real_move_holds_then_glides_and_is_deterministic():
    fn = lambda t: 500 if t < 10 else 1400
    a = sca.crop_track(analysis(fn), 0, 20)
    b = sca.crop_track(analysis(fn), 0, 20)
    assert a == b
    xs = [x for _, x in a["keyframes"]]
    assert xs[0] < xs[-1] and len(set(xs)) == 2
    move_start = [k for k in a["keyframes"] if k[1] == xs[0]][-1][0]
    assert move_start >= 10.5  # held ~hold_sec after the subject moved (+median lag)
    times = [k[0] for k in a["keyframes"]]
    assert times == sorted(times) and times[-1] == 20


def test_expression_matches_keyframes():
    tr = sca.crop_track(analysis(lambda t: 500 if t < 10 else 1400), 0, 20)
    expr = sca.x_expression(tr["keyframes"])
    for (t0, x0), (t1, x1) in zip(tr["keyframes"], tr["keyframes"][1:]):
        if t1 > t0:
            mid = (t0 + t1) / 2
            assert abs(eval_expr(expr, mid) - (x0 + (x1 - x0) * (mid - t0) / (t1 - t0))) < 1.0
    assert eval_expr(expr, 0.0) == tr["keyframes"][0][1]
    assert eval_expr(expr, 25.0) == tr["keyframes"][-1][1]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_render_tracked_outputs_1080x1920(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=6",
                    "-f", "lavfi", "-i", "sine=duration=6", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    tr = sca.crop_track(analysis(lambda t: 500 if t < 2 else 1400, t_end=6), 1.0, 4.0)
    out = tmp_path / "out.mp4"
    sca.render_tracked(str(src), str(out), 1.0, 4.0, tr)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height:format=duration",
                                      "-of", "json", str(out)], capture_output=True, text=True).stdout)
    v = [s for s in info["streams"] if s.get("width")][0]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert abs(float(info["format"]["duration"]) - 4.0) < 0.15


def test_virality_tracked_compose_uses_one_analysis_for_all_clips(monkeypatch, tmp_path):
    from modes.virality import runner
    a = analysis(lambda t: 1500, t_end=60)
    p = tmp_path / "faces.json"
    p.write_text(json.dumps(a))
    loads = []
    real = runner._load_face_analysis
    monkeypatch.setattr(runner, "_load_face_analysis", lambda crop: loads.append(1) or real(crop))
    renders = []
    monkeypatch.setattr(sca, "render_tracked", lambda src, out, s, d, tr, **kw: renders.append((s, d, tr["crop_status"])) or out)
    segs = [{"start": 1.0, "duration": 10.0}, {"start": 20.0, "duration": 12.0}, {"start": 40.0, "duration": 8.0}]
    out = runner._compose_tracked("in.mp4", segs, tmp_path, {"mode": "tracked", "analysis_path": str(p), "cache": "hit"})
    assert len(loads) == 1 and len(renders) == 3 and len(out) == 3
    data = json.loads((tmp_path / "selected_segments.json").read_text())
    s0 = data["segments"][0]
    assert s0["crop_status"] == "tracked" and s0["output_aspect"] == "9:16" and s0["crop_cache"] == "hit"
    assert s0["source_width"] == W and s0["crop_detector"].startswith("opencv-haar")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_analyze_faces_headless_on_video_without_people(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from scripts.shared.crop.cropper import analyze_faces
    src = tmp_path / "nobody.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)], check=True)
    a = analyze_faces(str(src), sample_fps=2.0)
    assert a["meta"]["width"] == 1280 and 7 <= a["meta"]["samples"] <= 9
    assert all(s["faces"] == [] for s in a["samples"])
    assert sca.crop_track(a, 0, 4)["fallback_reason"] == "no_subject_found"


@pytest.mark.parametrize("w,h", [(1280, 720), (1920, 1080), (3840, 2160), (1080, 1920), (720, 1280), (1000, 777)])
def test_crop_window_is_exact_9_16(w, h):
    cw, ch = sca.crop_window(w, h)
    assert cw * 16 == ch * 9 and cw % 2 == 0 and ch % 2 == 0 and cw <= w and ch <= h


@pytest.mark.parametrize("content", ["", "{not json", '{"meta": {}, "samples": []}', '{"samples": "x"}'])
def test_bad_cached_analysis_falls_back_to_center(monkeypatch, tmp_path, content):
    from modes.virality import runner
    p = tmp_path / "faces.json"
    p.write_text(content)
    monkeypatch.setattr(runner, "_source_dims", lambda path: (1920, 1080))
    renders = []
    monkeypatch.setattr(sca, "render_tracked", lambda src, out, s, d, tr, **kw: renders.append(tr) or out)
    out = runner._compose_tracked("in.mp4", [{"start": 0, "duration": 5}], tmp_path,
                                  {"mode": "tracked", "analysis_path": str(p)})
    assert out and renders[0]["crop_status"] == "center_fallback"
    seg = json.loads((tmp_path / "selected_segments.json").read_text())["segments"][0]
    assert seg["crop_fallback_reason"] == "analysis_unreadable"
