"""crop-shots-v2: shot-aware 9:16 crop from a per-clip analysis (cuts + YuNet faces)."""
import json
import shutil
import subprocess

import pytest

from scripts.shared.crop.smart_crop_adapter import crop_track_v2, crop_window

W, H = 1920, 1080
CW, _ = crop_window(W, H)                       # 606


def an(cuts=(), faces=None, dur=10.0, fps=10.0):
    """faces(t) -> list of [x, y, w, h, score, mouth]."""
    samples = [{"t": round(i / fps, 3), "faces": faces(i / fps) if faces else []} for i in range(int(dur * fps))]
    return {"meta": {"version": "t", "detector": "opencv-yunet-2023mar", "width": W, "height": H},
            "cuts": list(cuts), "samples": samples}


def x_at(track, t):
    k = track["keyframes"]
    for (t0, x0), (t1, x1) in zip(k, k[1:]):
        if t0 <= t < t1:
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    return k[-1][1]


def face(cx, w=200, score=0.93, mouth=0.0):
    return [int(cx - w / 2), 300, w, int(w * 1.3), score, mouth]


def inside(track, t, cx, w=200):
    x = x_at(track, t)
    return x <= cx - w / 2 and cx + w / 2 <= x + CW


def test_window_jumps_at_a_cut_never_pans_across_it():
    tr = crop_track_v2(an(cuts=[5.0], faces=lambda t: [face(400 if t < 5 else 1500)]), 10.0, width=W, height=H)
    assert inside(tr, 4.9, 400) and inside(tr, 5.0, 1500)
    assert [5.0, x_at(tr, 4.99)] in tr["keyframes"] or any(k[0] == 5.0 for k in tr["keyframes"])
    jump = [k for k in tr["keyframes"] if k[0] == 5.0]
    assert len(jump) == 2 and jump[0][1] != jump[1][1]
    assert [s["target"] for s in tr["shots"]] == ["person", "person"] and tr["cuts"] == [5.0]


def test_steady_shot_is_one_static_position():
    jitter = lambda t: [face(960 + (7 if int(t * 10) % 2 else -7))]   # noqa: E731
    tr = crop_track_v2(an(faces=jitter), 10.0, width=W, height=H)
    assert len({k[1] for k in tr["keyframes"]}) == 1 and inside(tr, 5, 960)


def test_face_is_never_cut_by_the_window_edge():
    walk = lambda t: [face(500 + 100 * t)]                                # noqa: E731
    tr = crop_track_v2(an(faces=walk), 10.0, width=W, height=H)
    late = [t / 10 for t in range(10, 100) if not inside(tr, t / 10, 500 + 10 * t)]
    assert len(late) <= 3                                                # catches up within ~0.3 s


def test_two_faces_too_close_for_a_split_share_one_frame():
    """Owner 2026-11-09 («почему в кадре два одинаковых фрагмента»): when a split half cannot leave the
    other person out, a split shows the same pair twice -> one 9:16 window on the middle of the pair."""
    tr = crop_track_v2(an(faces=lambda t: [face(800, 150), face(1150, 150)]), 10.0, width=W, height=H)
    assert tr["shots"][0]["target"] == "pair" and not tr.get("splits")
    x = x_at(tr, 5)
    assert x <= 800 - 75 and 1150 + 75 <= x + CW              # both faces fully in the one window


def test_two_faces_far_enough_still_split():
    """Owner's rule stays: two faces that a half CAN separate -> split, one person per half."""
    tr = crop_track_v2(an(faces=lambda t: [face(600, 150), face(1300, 150)]), 10.0, width=W, height=H)
    assert tr["shots"][0]["target"] == "split" and tr["splits"] == [{"start": 0.0, "end": 10.0}]
    assert split_inside(tr, "top", 5, 600, 300 + 97, 150, 195) and split_inside(tr, "bottom", 5, 1300, 300 + 97, 150, 195)
    assert tr["split"]["h"] >= 0.48 * H


def split_inside(track, name, t, cx, cy, w, h):
    sp = track["split"]
    x, y = x_at({"keyframes": sp[name]["x"]}, t), x_at({"keyframes": sp[name]["y"]}, t)
    return x <= cx - w / 2 and cx + w / 2 <= x + sp["w"] and y <= cy - h / 2 and cy + h / 2 <= y + sp["h"]


def test_two_people_too_far_apart_get_a_split_screen_left_on_top():
    two = lambda t: [face(500, 90), face(1400, 80)]                                   # noqa: E731
    tr = crop_track_v2(an(cuts=[4.0], faces=lambda t: two(t) if t >= 4 else [face(900)]), 10.0, width=W, height=H)
    assert [s["target"] for s in tr["shots"]] == ["person", "split"]
    sp = tr["split"]
    assert sp["intervals"] == [[4.0, 10.0]] and sp["w"] * 8 == sp["h"] * 9 and sp["w"] % 2 == sp["h"] % 2 == 0
    for t in (4.0, 6.5, 9.9):                       # both faces fully inside their half, all shot long
        assert split_inside(tr, "top", t, 500, 300 + 117 / 2, 90, 117)
        assert split_inside(tr, "bottom", t, 1400, 300 + 104 / 2, 80, 104)
    # no face zoom: as large as possible while each half shows only its own person
    assert sp["h"] == 8 * ((min(H, 900 * 8 / 9) // 8) // 2 * 2) and sp["w"] <= 900 + 9


def test_split_window_size_is_one_per_clip_and_faces_stay_inside():
    near = lambda t: [face(300, 200), face(1600, 200)] if t < 5 else [face(400, 60), face(1500, 60)]  # noqa: E731
    tr = crop_track_v2(an(cuts=[5.0], faces=near), 10.0, width=W, height=H)
    assert [s["target"] for s in tr["shots"]] == ["split", "split"]
    assert split_inside(tr, "top", 2, 300, 300 + 130, 200, 260) and split_inside(tr, "bottom", 7, 1500, 300 + 39, 60, 78)


def test_small_far_extra_face_does_not_split():
    f = lambda t: [face(700, 250), face(1700, 40, 0.82)]                               # noqa: E731
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert tr["shots"][0]["target"] == "person" and "split" not in tr and inside(tr, 5, 700, 250)


def test_low_score_giant_box_is_ignored():
    # blurred foreground shoulder: huge, score ~0.6, sorted first by area
    f = lambda t: [[150, 50, 730, 1000, 0.65, None], face(1100, 270)]    # noqa: E731
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert inside(tr, 5, 1100, 270)


def test_short_faceless_insert_keeps_framing_long_one_centres():
    f = lambda t: [face(400)] if t < 4 or t >= 4.6 else []                # noqa: E731
    tr = crop_track_v2(an(cuts=[4.0, 4.6], faces=f), 10.0, width=W, height=H)
    assert x_at(tr, 4.3) == x_at(tr, 3.9)
    tr2 = crop_track_v2(an(cuts=[4.0], faces=lambda t: [face(400)] if t < 4 else []), 10.0, width=W, height=H)
    assert x_at(tr2, 7) == (W - CW) // 2 // 2 * 2 and tr2["shots"][1]["target"] == "none"


def test_no_analysis_or_no_faces_is_explicit_fallback():
    tr = crop_track_v2(None, 10.0, width=W, height=H)
    assert tr["crop_status"] == "center_fallback" and tr["fallback_reason"] == "detector_unavailable"
    tr = crop_track_v2(an(), 10.0, width=W, height=H)
    assert tr["crop_status"] == "center_fallback" and tr["fallback_reason"] == "no_subject_found"


def test_deterministic_and_even_in_bounds():
    a = an(cuts=[3.3, 7.1], faces=lambda t: [face(300 + 150 * t, 180), face(1700, 100, 0.85)])
    t1 = crop_track_v2(a, 10.0, width=W, height=H)
    t2 = crop_track_v2(json.loads(json.dumps(a)), 10.0, width=W, height=H)
    assert t1 == t2
    assert all(0 <= x <= W - CW and x % 2 == 0 for _, x in t1["keyframes"])
    assert all(b[0] <= a2[0] for b, a2 in zip(t1["keyframes"], t1["keyframes"][1:]))


# ── clip analysis on a real (synthetic) video ──────────────────────────

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_analyze_clip_finds_hard_cuts_only_in_range(tmp_path):
    from scripts.shared.crop import clip_analysis as ca
    if not ca.model_ok()["ok"]:
        pytest.skip("yunet model")
    v = tmp_path / "cuts.mp4"
    # 3 shots: moving test pattern (motion, no cut) → red → blue, cuts at 4 s and 7 s
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4",
                    "-f", "lavfi", "-i", "color=c=red:size=640x360:rate=25:duration=3",
                    "-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=25:duration=3",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", str(v)], check=True)
    a = ca.analyze_clip(str(v), 2.0, 7.0)
    assert [round(c, 1) for c in a["cuts"]] == [2.0, 5.0]               # relative to the clip start
    assert a["meta"]["width"] == 640 and all(s["faces"] == [] for s in a["samples"])
    assert 60 <= len(a["samples"]) <= 72                                # 10 fps over 7 s


def test_model_is_pinned():
    from scripts.shared.crop import clip_analysis as ca
    st = ca.model_ok()
    assert st["ok"], st


def test_runner_caches_clip_analysis(monkeypatch, tmp_path):
    from modes.virality import runner
    from scripts.shared.crop import clip_analysis as ca
    calls = []
    monkeypatch.setattr(ca, "analyze_clip", lambda v, s, d: calls.append(s) or dict(an(dur=d), meta={**an()["meta"], "version": ca.CLIP_ANALYSIS_VERSION, "start": s}))
    crop = {"analysis": "per_clip", "clip_cache_dir": str(tmp_path / "c"), "source_sha256": "ab" * 32}
    a1, c1, e1 = runner._clip_analysis(crop, "v.mp4", 10.0, 5.0)
    a2, c2, _ = runner._clip_analysis(crop, "v.mp4", 10.0, 5.0)
    a3, c3, _ = runner._clip_analysis(crop, "v.mp4", 20.0, 5.0)
    assert (c1, c2, c3) == ("miss", "hit", "miss") and calls == [10.0, 20.0] and e1 is None
    assert not list((tmp_path / "c").glob("*.partial"))


def test_runner_analysis_failure_is_centre_crop_not_a_failed_run(monkeypatch):
    from modes.virality import runner
    from scripts.shared.crop import clip_analysis as ca
    monkeypatch.setattr(ca, "analyze_clip", lambda *a: (_ for _ in ()).throw(RuntimeError("decoder")))
    a, cache, err = runner._clip_analysis({"analysis": "per_clip"}, "v.mp4", 0.0, 5.0)
    assert a is None and err.startswith("analysis_failed")


def test_subtitles_move_to_the_seam_during_split():
    from scripts.shared.subtitles import Cue, SubtitleStyle, to_ass
    cues = [Cue(0.0, 2.0, ["раз"], 1), Cue(4.5, 6.0, ["два"], 1), Cue(9.5, 11.0, ["три"], 1)]
    ass = to_ass(cues, SubtitleStyle(), centered=[(4.0, 10.0)])
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert "\\pos" not in lines[0] and "{\\an5\\pos(540,960)}два" in lines[1] and "\\pos" not in lines[2]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_render_split_screen_is_1080x1920(tmp_path):
    from scripts.shared.crop.smart_crop_adapter import render_tracked
    v = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=4",
                    "-f", "lavfi", "-i", "sine=duration=4", "-shortest", "-pix_fmt", "yuv420p", str(v)], check=True)
    tr = crop_track_v2(an(cuts=[2.0], dur=4.0, faces=lambda t: [face(500, 90), face(1400, 80)] if t >= 2 else [face(900)]),
                       4.0, width=W, height=H)
    out = tmp_path / "o.mp4"
    render_tracked(str(v), str(out), 0.0, 4.0, tr)
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height:format=duration", "-of", "json", str(out)],
                       capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    assert {"width": 1080, "height": 1920} in [{k: s.get(k) for k in ("width", "height")} for s in info["streams"]]
    assert abs(float(info["format"]["duration"]) - 4.0) < 0.15


def test_layout_switches_fast_inside_one_shot_and_ignores_flicker():
    def f(t):
        if 2.0 <= t < 5.0:                                    # second person walks in at 2 s, leaves at 5 s
            return [face(500, 150), face(1400, 150)]
        if 8.5 <= t < 8.7:                                    # a 0.2 s false second face: no flicker
            return [face(500, 150), face(1400, 150)]
        return [face(500, 150)]
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert len(tr["splits"]) == 1
    sp = tr["splits"][0]
    assert 1.5 <= sp["start"] <= 2.1 and 4.9 <= sp["end"] <= 5.5      # reacts within ~0.5 s
    assert inside(tr, 1.0, 500, 150) and inside(tr, 8.6, 500, 150) and inside(tr, 9.5, 500, 150)


def test_absorb_short_runs():
    from scripts.shared.crop.smart_crop_adapter import _absorb_short
    runs = [[0, 0.0, 3.0], ["split", 3.1, 3.3], [0, 3.4, 6.0], ["split", 6.1, 9.0]]
    assert _absorb_short(runs, 0.5) == [[0, 0.0, 6.0], ["split", 6.1, 9.0]]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_fit_layout_keeps_the_whole_frame_with_black_bars(tmp_path):
    import numpy as np
    from scripts.shared.crop.smart_crop_adapter import fit_track, render_tracked
    v = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=white:size=1920x1080:rate=25:duration=1",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    out = tmp_path / "o.mp4"
    render_tracked(str(v), str(out), 0.0, 1.0, fit_track(1920, 1080, 1.0))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(1920, 1080)
    rows = img.mean(axis=1)
    bright = np.where(rows > 128)[0]
    assert abs(bright[0] - (1920 - 608) / 2) <= 4 and abs(bright[-1] - (1920 + 608) / 2) <= 4   # 16:9 band, uncropped
    assert rows[:600].max() < 30 and rows[-600:].max() < 30                                     # black bars
    assert img[960].min() > 200                                                                 # full width kept


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_blur_background_fills_the_screen_and_keeps_the_frame(tmp_path):
    import numpy as np
    from scripts.shared.crop.smart_crop_adapter import fit_track, render_tracked
    v = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=1",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    out = tmp_path / "o.mp4"
    render_tracked(str(v), str(out), 0.0, 1.0, fit_track(1920, 1080, 1.0, "blur"))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(1920, 1080).astype(float)
    assert img[:500].mean() > 15 and img[-500:].mean() > 15          # no black bars: blurred picture
    top_detail = np.abs(np.diff(img[100:400], axis=1)).mean()
    mid_detail = np.abs(np.diff(img[700:1200], axis=1)).mean()
    assert mid_detail > 3 * top_detail                                 # sharp original in the middle, blur around


# ── burned-in text vs subtitles ─────────────────────────────────────────

def test_text_boxes_find_a_lower_third_and_ignore_a_plain_frame():
    import cv2
    import numpy as np
    from scripts.shared.crop.clip_analysis import text_boxes
    img = np.full((1080, 1920, 3), 90, np.uint8)
    cv2.rectangle(img, (60, 790), (1500, 930), (25, 25, 25), -1)
    cv2.putText(img, "ROD ITELI EVANA UEHALI IZ SSSR V 1979", (80, 860), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
    boxes = text_boxes(img)
    assert boxes and all(780 <= y and y + h <= 940 for x, y, w, h in boxes)
    assert text_boxes(np.full((1080, 1920, 3), 90, np.uint8)) == []


def test_text_zones_map_through_the_crop_and_split():
    tr = crop_track_v2(an(cuts=[5.0], faces=lambda t: [face(900)] if t < 5 else [face(500, 90), face(1400, 80)]),
                       10.0, width=W, height=H)
    a = dict(an(), text=[{"t": 1.0, "boxes": [[60, 800, 1400, 120]]}, {"t": 1.5, "boxes": [[60, 800, 1400, 120]]},
                         {"t": 7.0, "boxes": [[60, 800, 1400, 120]]}])
    from scripts.shared.crop.smart_crop_adapter import text_zones
    z = text_zones(a, tr)
    k = 1920 / tr["crop_h"]
    assert z[0][:2] == [0.5, 2.0] and abs(z[0][2] - (800 - tr["y"]) * k) <= 1 and abs(z[0][3] - (920 - tr["y"]) * k) <= 1
    assert len(z) == 2 and z[1][0] == 6.5                   # split shot: mapped into the halves
    fit = text_zones(a, fit_track_for_test())
    assert 656 < fit[0][2] < 1264                            # fit: inside the picture band


def fit_track_for_test():
    from scripts.shared.crop.smart_crop_adapter import fit_track
    return fit_track(W, H, 10.0)


def test_subtitle_moves_above_text_and_back():
    from scripts.shared.subtitles import Cue, SubtitleStyle, to_ass
    cues = [Cue(0.5, 2.0, ["над надписью"], 2), Cue(4.0, 5.0, ["обычно"], 1)]
    ass = to_ass(cues, SubtitleStyle(), avoid=[[0.0, 2.5, 1420, 1640]])
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert "{\\an2\\pos(540,1390)}над надписью" in lines[0] and "pos" not in lines[1]


def test_big_face_close_up_gets_a_wider_square_window():
    tr = crop_track_v2(an(cuts=[5.0], faces=lambda t: [face(900, 420)] if t < 5 else [face(900, 150)]), 10.0,
                       width=W, height=H)
    cu = tr["closeup"]
    assert cu["intervals"] == [[0.0, 5.0]] and cu["size"] == 1080 and tr["shots"][0]["target"] == "closeup"
    x = x_at({"keyframes": cu["x"]}, 2.0)
    assert x <= 900 - 210 and 900 + 210 <= x + cu["size"]          # the whole face inside the square
    assert tr["shots"][1]["target"] == "person"                     # normal shot stays full-screen


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_render_close_up_layout(tmp_path):
    from scripts.shared.crop.smart_crop_adapter import render_tracked
    v = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=3",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    tr = crop_track_v2(an(cuts=[1.5], dur=3.0, faces=lambda t: [face(900, 420)] if t < 1.5 else [face(900, 150)]),
                       3.0, width=W, height=H)
    out = tmp_path / "o.mp4"
    render_tracked(str(v), str(out), 0.0, 3.0, tr)
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True)
    assert p.stdout.strip().splitlines()[0] == "1080,1920"


def test_game_layout_finds_a_steady_webcam_face_and_keeps_the_game_centre():
    from scripts.shared.crop.smart_crop_adapter import game_track
    samples = [{"t": i * 0.1, "faces": [[1650 + (i % 3), 820, 120, 140, 0.95]]} for i in range(100)]
    tr = game_track({"samples": samples, "cuts": []}, 10.0, width=1920, height=1080)
    cx, cy, cw, ch = tr["cam"]
    assert tr["layout"] == "game" and cx <= 1650 and cx + cw >= 1770 and cy <= 820 and cy + ch >= 960
    assert cx + cw <= 1920 and cy + ch <= 1080 and abs(cw / ch - 1080 / 960) < 0.02
    gx, gy, gw, gh = tr["game"]
    assert abs((gx + gw / 2) - 960) <= 2 and abs(gw / gh - 1080 / 960) < 0.02
    moving = [{"t": i * 0.1, "faces": [[100 + i * 15, 400, 120, 140, 0.95]]} for i in range(100)]
    assert game_track({"samples": moving}, 10.0, width=1920, height=1080) is None     # a person in the game
    assert game_track({"samples": [{"t": 0, "faces": []}] * 50}, 5.0, width=1920, height=1080) is None
