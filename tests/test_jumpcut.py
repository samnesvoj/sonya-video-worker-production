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
    assert ad == {"path": "/x/ad.mp4", "at": 0.4}
    assert sonya_ad_settings({"sonya_ad": {"asset": "/y.mp4", "at": 0.9}}, {})["at"] == 0.6   # clamped
    monkeypatch.setenv("STREAMER_SONYA_AD", "true")
    assert sonya_ad_settings({"sonya_ad_asset": "/x/ad.mp4"}, {})["path"] == "/x/ad.mp4"
