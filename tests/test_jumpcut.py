"""scripts/shared/jumpcut.py + smart_crop_adapter.render_pieces: pauses cut out, words never clipped,
the SONYA ad spliced in at a pause near 40 %, all in one encode."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from scripts.shared import jumpcut as jc


def _w(spans):
    return [{"word": f"w{i}.", "start": a, "end": b} for i, (a, b) in enumerate(spans)]


def test_long_pause_is_cut_short_pause_kept():
    words = _w([(10.0, 11.0), (11.3, 12.0), (14.0, 15.0)])           # 0.3 s gap kept, 2.0 s gap cut
    ps = jc.pieces(words, 10.0, 15.0, fps=30)
    assert len(ps) == 2
    assert jc.total(ps) == pytest.approx(5.0 - 2.0 + 0.25, abs=0.05)
    # no word is clipped: every word lies inside one piece
    for w in words:
        assert any(a - 1e-3 <= w["start"] and w["end"] <= a + d + 1e-3 for a, d in ps)


def test_loud_gap_is_an_event_and_stays():
    words = _w([(10.0, 11.0), (14.0, 15.0)])
    loud = [0.0] * 20
    loud[12] = 2.5                                                     # a scream / explosion ASR did not catch
    assert len(jc.pieces(words, 10.0, 15.0, loud=loud)) == 1


def test_edges_on_frame_grid_and_remap():
    words = _w([(10.0, 11.0), (13.0, 14.0)])
    ps = jc.pieces(words, 10.0, 14.0, fps=60)
    for a, d in ps:
        assert abs(a * 60 - round(a * 60)) < 0.01 and abs(d * 60 - round(d * 60)) < 0.01
    out = jc.remap_words(words, ps)
    assert out[0]["start"] == 0.0 and out[1]["start"] == pytest.approx(1.25, abs=0.02)
    assert jc.to_output(12.0, ps) is None                              # inside the removed gap


def test_pick_insert_prefers_a_sentence_end_near_40_percent():
    words = [{"word": "раз", "start": 0.0, "end": 3.0}, {"word": "два.", "start": 3.1, "end": 7.8},
             {"word": "три", "start": 8.2, "end": 12.0}, {"word": "четыре", "start": 12.2, "end": 20.0}]
    t = jc.pick_insert(words, [(0.0, 20.0)])
    assert t == pytest.approx(8.0, abs=0.01)                           # after «два.» (40 % = 8 s)
    assert jc.pick_insert([{"word": "x", "start": 0, "end": 20}], [(0.0, 20.0)]) is None


FF = os.environ.get("SONYA_FFMPEG") or shutil.which("ffmpeg")
needs_ff = pytest.mark.skipif(not FF, reason="ffmpeg not installed")


def _probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                        "-of", "json", path], capture_output=True, text=True)
    return json.loads(r.stdout)


@needs_ff
def test_render_pieces_game_layout_with_ad(tmp_path, monkeypatch):
    from scripts.shared.crop import smart_crop_adapter as sca
    monkeypatch.setenv("SONYA_FFMPEG", FF)
    src, ad = tmp_path / "src.mp4", tmp_path / "ad.mp4"
    subprocess.run([FF, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    subprocess.run([FF, "-v", "error", "-f", "lavfi", "-i", "color=c=purple:size=540x960:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=880:duration=2", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(ad)], check=True)
    track = {"layout": "game", "cam": [0, 0, 320, 180], "game": [200, 0, 300, 360]}
    pieces = [(1.0, 3.0), (6.0, 4.0)]                                  # 2 s of «silence» cut out
    rep = {}
    out = tmp_path / "out.mp4"
    sca.render_pieces(str(src), str(out), pieces, track, 1.0, fps=30, insert={"path": str(ad), "at": 2.0, "dur": 2.0,
                      "gain_db": -3.0}, encoder_args=["-c:v", "libx264", "-preset", "ultrafast"], report=rep)
    d = _probe(str(out))
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert float(d["format"]["duration"]) == pytest.approx(7.0 + 2.0, abs=0.15)
    assert rep["render_pieces"] == 3 and rep["render_inputs"] == 4     # piece 1 split at the ad + the ad
    # the ad frame is full screen purple at 2.5 s, the clip resumes after it
    def px(t):
        r = subprocess.run([FF, "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-vf", "scale=1:1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        return tuple(r.stdout[:3])
    r, g, b = px(3.0)
    assert r > 90 and b > 90 and g < 60
    assert px(6.0) != (r, g, b)


def test_sonya_ad_settings(monkeypatch, tmp_path):
    from modes.streamer.runner import sonya_ad_settings
    monkeypatch.delenv("STREAMER_SONYA_AD", raising=False)
    assert sonya_ad_settings({}, {}) is None                           # off by default in production
    assert sonya_ad_settings({}, {"sonyaAd": False}) is None
    ad = sonya_ad_settings({"sonya_ad_asset": "/x/ad.mp4"}, {"sonyaAd": True})
    assert ad == {"path": "/x/ad.mp4", "at": 0.4, "slot": "suspense"}
    assert sonya_ad_settings({"sonya_ad": {"asset": "/y.mp4", "at": 0.9}}, {})["at"] == 0.6   # clamped
    monkeypatch.setenv("STREAMER_SONYA_AD", "true")
    assert sonya_ad_settings({"sonya_ad_asset": "/x/ad.mp4"}, {})["path"] == "/x/ad.mp4"


def test_site_never_inserts_sonya_ad_without_explicit_param(monkeypatch, tmp_path):
    """Owner 2026-11-10 ~05:35: users in production never get the SONYA ad unless a parameter turns it on."""
    from modes.streamer.runner import streamer_crop
    monkeypatch.delenv("STREAMER_SONYA_AD", raising=False)
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    for params in ({}, {"settings": {}}, {"settings": {"subtitleStyle": "classic", "jumpCuts": True}},
                   {"settings": {"sonyaAd": None}}, {"ad_slot": "suspense"}):
        assert "sonya_ad" not in streamer_crop(dict(params, source_sha256="0"), str(src), str(tmp_path))
    on = streamer_crop({"source_sha256": "0", "sonya_ad": True, "sonya_ad_asset": "/a.mp4", "ad_slot": "after_hook"},
                       str(src), str(tmp_path))
    assert on["sonya_ad"]["slot"] == "after_hook"
    assert streamer_crop({"source_sha256": "0", "sonya_ad": True, "sonya_ad_asset": "/a.mp4", "ad_slot": "bogus"},
                         str(src), str(tmp_path))["sonya_ad"]["slot"] == "suspense"


def _ws(spec):
    """[(text, start, end)] -> words"""
    return [{"word": t, "start": a, "end": b} for t, a, b in spec]


SUSP = _ws([("Короче.", 0.0, 0.5), ("Смотри", 0.6, 1.0), ("что", 1.0, 1.3), ("будет.", 1.3, 1.8),
           ("Так,", 3.2, 3.6), ("сейчас,", 3.7, 4.2), ("ну", 4.4, 4.6), ("давай,", 4.6, 5.2), ("давай.", 5.3, 6.0),
           ("Есть!", 7.0, 7.6), ("Ахаха", 7.7, 8.6), ("убил!", 8.7, 9.4), ("Красиво.", 10.0, 10.8),
           ("Вот", 11.5, 11.8), ("так", 11.8, 12.1), ("надо.", 12.1, 12.8), ("Учитесь.", 13.2, 14.0),
           ("Всё.", 16.0, 16.5)])
PS = [(0.0, 20.0)]


def test_ad_slot_suspense_goes_right_before_the_payoff():
    from scripts.shared import jumpcut as jc
    t, slot, how = jc.pick_slot(SUSP, PS, slot="suspense", payoff=7.0)
    assert slot == "suspense" and how == "editor payoff"
    assert 6.0 <= t <= 7.0                                    # after «давай.», before «Есть!» — payoff not eaten
    # no editor mark: the build-up words lead to the same spot
    t2, slot2, how2 = jc.pick_slot(SUSP, PS, slot="suspense")
    assert slot2 == "suspense" and 6.0 <= t2 <= 7.0, (t2, how2)


def test_ad_slot_never_inside_a_word_nor_in_the_hook():
    from scripts.shared import jumpcut as jc
    for slot in jc.AD_SLOTS:
        for kw in ({}, {"payoff": 1.0}, {"punchline": 19.0}, {"payoff": 7.0, "punchline": 13.2}):
            t, _, _ = jc.pick_slot(SUSP, PS, slot=slot, **kw)
            assert t is not None and 3.0 <= t <= 17.0
            assert not any(w["start"] < t < w["end"] for w in SUSP), (slot, kw, t)


def test_ad_slot_after_hook_and_before_punchline():
    from scripts.shared import jumpcut as jc
    t, slot, _ = jc.pick_slot(SUSP, PS, slot="after_hook")
    assert slot == "after_hook" and 3.0 <= t <= 6.5
    t, slot, how = jc.pick_slot(SUSP, PS, slot="before_punchline", punchline=13.2)
    assert slot == "before_punchline" and 12.8 <= t <= 13.2
    t, slot, how = jc.pick_slot(SUSP, PS, slot="before_punchline")
    assert slot == "before_punchline" and how == "last phrase" and t >= 10.0


def test_ad_slot_falls_back_to_mid_pause():
    from scripts.shared import jumpcut as jc
    flat = _ws([(f"слово{i}.", i * 1.0, i * 1.0 + 0.6) for i in range(20)])
    t, slot, how = jc.pick_slot(flat, PS, slot="suspense")
    assert slot == "mid_pause" and "fallback" in how and t is not None
