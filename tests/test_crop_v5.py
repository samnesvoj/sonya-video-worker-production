"""crop-shots-v5 (owner 2026-11-13, job cd6916ff): faces whole, no false split, calm layouts, no stretching;
the source's own captions blurred; weak editor picks dropped. Real data: tests/data/cd6916ff_*."""
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.shared.crop import crop_v5
from scripts.shared.crop.smart_crop_adapter import crop_track_v2, crop_window, track_graph

DATA = Path(__file__).parent / "data"
W, H = 1920, 1080
CW, _ = crop_window(W, H)


def real(name):
    with gzip.open(DATA / f"cd6916ff_{name}.json.gz", "rt") as f:
        return json.load(f)


def an(cuts=(), faces=None, dur=10.0, fps=10.0):
    samples = [{"t": round(i / fps, 3), "faces": faces(i / fps) if faces else []} for i in range(int(dur * fps))]
    return {"meta": {"version": "t", "detector": "opencv-yunet-2023mar", "width": W, "height": H},
            "cuts": list(cuts), "samples": samples}


def face(cx, w=200, score=0.93, mouth=1.5, y=300):
    return [int(cx - w / 2), y, w, int(w * 1.3), score, mouth]


def val(keys, t):
    for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0) if t1 > t0 else x1
    return keys[-1][1]


def window_at(tr, t):
    """(x, width) of the window that shows the main picture at clip time t."""
    cu = tr.get("closeup") or {}
    if any(a <= t < b for a, b in cu.get("intervals") or []):
        return val(cu["x"], t), cu["size"]
    return val(tr["keyframes"], t), tr["crop_w"]


def layout_runs(tr):
    return tr["layouts"]


@pytest.mark.parametrize("clip", ["clip4", "clip1"])
def test_her_clips_have_no_false_split_and_calm_layouts(clip):
    """Clip 4: a hand-held selfie (one face moving fast) was split in two; clip 1: a silent small bystander
    got half the screen. v5: no split, and no layout shorter than MIN_LAYOUT_SEC (except a tail < it)."""
    a = real(clip)
    d = a["meta"]["duration"]
    tr = crop_track_v2(a, d, width=a["meta"]["width"], height=a["meta"]["height"])
    assert tr["crop_version"] == "crop-shots-v5" and not tr.get("split")
    lay = layout_runs(tr)
    changes = [(p, q) for p, q in zip(lay, lay[1:]) if p[0] != q[0]]
    assert len(changes) <= 1
    for p, q in changes:
        assert p[2] - p[1] >= crop_v5.MIN_LAYOUT_SEC and q[2] - q[1] >= crop_v5.MIN_LAYOUT_SEC


def test_her_selfie_clip_keeps_the_face_whole():
    """Clip 4: the main face is fully inside the shown window in ≥ 97 % of the samples (v3: face cut at 0–7 s)."""
    a = real("clip4")
    Wr = a["meta"]["width"]
    tr = crop_track_v2(a, a["meta"]["duration"], width=Wr, height=a["meta"]["height"])
    total = ok = 0
    for s in a["samples"]:
        fs = [f for f in s["faces"] if f[4] >= 0.8]
        if not fs:
            continue
        f = max(fs, key=lambda b: b[2] * b[3])
        x, w = window_at(tr, s["t"])
        total += 1
        ok += x - 2 <= f[0] and f[0] + f[2] <= x + w + 2
    assert total > 200 and ok / total >= 0.97


def test_fast_moving_single_face_is_one_person_not_a_split():
    """The same face jumping left/right between samples (hand-held camera) → never two people."""
    f = lambda t: [face(500 if int(t * 2) % 2 else 1300, 300)]           # noqa: E731
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert not tr.get("split")


def test_poster_face_does_not_split():
    """A still face that never talks (a poster, a photo on a screen) is not a person to split for."""
    f = lambda t: [face(500, 160), face(1400, 150, mouth=0.0)]            # noqa: E731
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert not tr.get("split")


def test_short_second_face_does_not_split():
    f = lambda t: [face(500, 150), face(1400, 150)] if 3.0 <= t < 4.2 else [face(500, 150)]   # noqa: E731
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert not tr.get("split")


def test_group_shot_is_wide_not_a_split():
    f = lambda t: [face(300, 150), face(900, 150), face(1500, 150)]        # noqa: E731
    tr = crop_track_v2(an(faces=f), 10.0, width=W, height=H)
    assert not tr.get("split") and tr["closeup"]["intervals"] == [[0.0, 10.0]]


def test_short_shot_inherits_the_layout_no_flicker():
    """A 0.8 s shot with a big face between two normal shots keeps the normal layout."""
    f = lambda t: [face(900, 420)] if 4.0 <= t < 4.8 else [face(900, 150)]   # noqa: E731
    tr = crop_track_v2(an(cuts=[4.0, 4.8], faces=f), 10.0, width=W, height=H)
    assert {p[0] for p in tr["layouts"]} == {"single"} and not tr.get("closeup")


def test_moving_face_stays_whole_with_margin():
    walk = lambda t: [face(300 + 130 * t, 220)]                            # noqa: E731
    tr = crop_track_v2(an(faces=walk), 10.0, width=W, height=H)
    for i in range(100):
        t = i / 10
        cx = 300 + 130 * t
        x, w = window_at(tr, t)
        assert x <= cx - 110 and cx + 110 <= x + w, t


def test_windows_never_stretch():
    tr = crop_track_v2(an(cuts=[5.0], faces=lambda t: [face(900, 420)] if t < 5 else [face(600, 150), face(1300, 150)]),
                       10.0, width=W, height=H)
    assert tr["crop_w"] * 16 == tr["crop_h"] * 9                  # 9:16 → 1080x1920
    g = track_graph(tr)
    assert f"crop=w={tr['closeup']['size']}:h={tr['closeup']['size']}" in g and "scale=1080:1080" in g
    assert tr["split"]["w"] * 8 == tr["split"]["h"] * 9 and "scale=1080:960" in g


def test_caption_detector_finds_the_sources_white_subtitles():
    cv2 = pytest.importorskip("cv2")
    from scripts.shared.crop.clip_analysis import caption_boxes_frame
    pos = caption_boxes_frame(cv2.imread(str(DATA / "cd6916ff_caption_pos.jpg")))
    assert pos and all(b[1] > 0.8 * 720 for b in pos) and max(b[2] for b in pos) > 600
    assert caption_boxes_frame(cv2.imread(str(DATA / "cd6916ff_caption_neg.jpg"))) == []


def test_captions_are_blurred_and_not_avoided_by_our_subtitles():
    from scripts.shared.crop.smart_crop_adapter import text_zones
    a = an(faces=lambda t: [face(900, 150)])
    a["text"] = [{"t": t / 2, "boxes": [], "captions": [[500, 900, 900, 50]]} for t in range(2, 7)]
    tr = crop_track_v2(a, 10.0, width=W, height=H)
    cb = tr["caption_blur"]
    assert len(cb) == 1 and cb[0][0] <= 1.0 and cb[0][1] >= 3.0
    x, y, w, h = cb[0][2:]
    assert x <= 500 and y <= 900 and x + w >= 1400 and y + h >= 950
    assert "gblur" in track_graph(tr) and text_zones(a, tr) == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_render_with_caption_blur_and_closeup(tmp_path):
    from scripts.shared.crop.smart_crop_adapter import render_tracked
    v = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=4",
                    "-f", "lavfi", "-i", "sine=duration=4", "-shortest", "-pix_fmt", "yuv420p", str(v)], check=True)
    a = an(dur=4.0, cuts=[2.0], faces=lambda t: [face(900, 420)] if t < 2 else [face(500, 150), face(1400, 150)])
    a["text"] = [{"t": 1.0, "boxes": [], "captions": [[500, 900, 900, 50]]}]
    tr = crop_track_v2(a, 4.0, width=W, height=H)
    assert tr.get("caption_blur") and tr.get("closeup") and tr.get("split")
    out = tmp_path / "o.mp4"
    render_tracked(str(v), str(out), 0.0, 4.0, tr)
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", str(out)],
                       capture_output=True, text=True, check=True)
    assert {"width": 1080, "height": 1920} in [{k: s.get(k) for k in ("width", "height")}
                                               for s in json.loads(p.stdout)["streams"]]


def test_weak_editor_picks_are_dropped_but_five_are_kept(monkeypatch):
    """8 requested, the editor rates only 3 strong → 5 clips (3 strong + 2 best of the rest), not 8."""
    import json as _json
    from scripts.shared.cut_selection import StoryConfig, select_story_clips
    from tests.test_cut_story_v3 import _Res, words_from

    texts = [f"Вопрос номер {i}: почему вы решили переехать в другой город? Потому что там работа и море, "
             f"и мы давно об этом мечтали всей семьёй, это было правильное решение." for i in range(12)]
    raw = words_from([t for t in texts for _ in range(2)], rate=0.45, pause_between=1.2)

    def chat(messages, **kw):
        items = _json.loads(messages[1]["content"].split("Фрагменты:\n", 1)[1].split("\n\nВерни", 1)[0])
        clips = [{"id": x["id"], "score": 88 if i < 3 else 55, "self_contained": True, "qa": True, "hook": "Х",
                  "keywords": [], "reason": "r"} for i, x in enumerate(items)]
        return _Res(_json.dumps({"clips": clips}, ensure_ascii=False))

    segs, meta = select_story_clips(raw, 8, StoryConfig(min_duration=10, editor=True), chat=chat)
    assert meta["selection_method"] == "editor"
    assert len(segs) == 5 and sorted(s["editor_score"] for s in segs)[-3:] == [88, 88, 88]
    assert meta["editor_dropped_weak"] == 3
    monkeypatch.setenv("SONYA_CUT_MIN_CLIPS", "3")
    segs3, _ = select_story_clips(raw, 8, StoryConfig(min_duration=10, editor=True), chat=chat)
    assert len(segs3) == 3


# ── fill_9x16 (owner 2026-11-15: «всегда 9:16 вертикально, как было, только без ошибок») ───────────────────

def _fill(a, d=10.0, **kw):
    return crop_track_v2(a, d, width=a["meta"]["width"], height=a["meta"]["height"], fill=True, **kw)


@pytest.mark.parametrize("clip", ["clip4", "clip1"])
def test_fill_real_clips_one_full_window(clip):
    """Her clips: fill mode never makes a close-up square or a split."""
    a = real(clip)
    tr = _fill(a, a["meta"]["duration"])
    assert tr["crop_version"] == crop_v5.CROP_VERSION_FILL
    assert not tr.get("split") and not tr.get("closeup")
    assert {lay for lay, _, _ in tr["layouts"]} <= {"single"}


def test_fill_big_face_is_not_closeup():
    tr = _fill(an(faces=lambda t: [face(900, 420)]))
    assert not tr.get("closeup") and tr["crop_w"] == CW


@pytest.mark.parametrize("fps", [10.0, 12.0])   # 12 fps: run edges off the 1 ms grid (the window panned 4 s)
def test_fill_two_people_apart_follow_the_talker_with_a_jump(fps):
    """Interviewer left talks 0–5 s, guest right talks 5–10 s, too far apart for one window: no split; the
    window is on the left person, then JUMPS (same time, two x) to the right one; each face whole; no pan."""
    def f(t):
        return [face(400, 160, mouth=3.0 if t < 5 else 0.1), face(1500, 160, mouth=0.1 if t < 5 else 3.0)]
    tr = _fill(an(faces=f, fps=fps))
    assert not tr.get("split") and not tr.get("closeup")
    x1, x2 = val(tr["keyframes"], 2.0), val(tr["keyframes"], 8.0)
    assert x1 <= 400 - 80 and 400 + 80 <= x1 + CW
    assert x2 <= 1500 - 80 and 1500 + 80 <= x2 + CW
    k = tr["keyframes"]
    jumps = [(a, b) for a, b in zip(k, k[1:]) if abs(a[0] - b[0]) < 1e-6 and a[1] != b[1]]
    assert len(jumps) == 1 and 4.0 <= jumps[0][0][0] <= 6.0
    assert all(abs(b[1] - a[1]) <= 8 for a, b in zip(k, k[1:]) if b[0] - a[0] > 1e-6)   # no pan
    assert tr["shots"][0]["target"] == "speaker"


def test_fill_speaker_ping_pong_is_calm():
    """Mouth noise flipping every 0.5 s must not make the window ping-pong: no speaker run < 2.5 s."""
    def f(t):
        odd = int(t * 2) % 2
        return [face(400, 160, mouth=2.0 if odd else 1.0), face(1500, 160, mouth=1.0 if odd else 2.0)]
    tr = _fill(an(faces=f))
    edges = [0.0] + (tr["shots"][0].get("speaker_switch") or []) + [10.0]
    assert all(b - a >= crop_v5.FILL_SPEAKER_MIN_SEC - 1e-6 for a, b in zip(edges, edges[1:]))


def test_fill_pair_that_fits_is_framed_together():
    tr = _fill(an(faces=lambda t: [face(800, 150), face(1150, 150)]))
    assert tr["shots"][0]["target"] == "pair" and not tr.get("split")


def test_fill_off_by_default_is_v5():
    tr = crop_track_v2(an(faces=lambda t: [face(900, 420)]), 10.0, width=W, height=H)
    assert tr["crop_version"] == "crop-shots-v5" and tr.get("closeup")


def test_fill_job_param_sets_the_layout(tmp_path):
    from scripts.shared import cut_profile as cp
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    p = cp.apply_profile("virality", {"crop_mode": "fill_9x16"}, str(v), str(tmp_path), ass_supported=True, editor=False)
    assert p["crop"]["layout"] == "fill_9x16"
    p = cp.apply_profile("virality", {}, str(v), str(tmp_path), ass_supported=True, editor=False)
    assert p["crop"]["layout"] == "smart"
