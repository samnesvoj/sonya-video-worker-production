"""«Качество нарезки»: standard (unchanged) / high = 1440x2560, 24 Mb/s cap, source up to 2160p (scripts/shared/quality.py)."""
import json
import os

import pytest

from scripts.shared import quality as Q
from scripts.shared.crop import smart_crop_adapter as sca
from scripts.shared.crop.smart_crop_adapter import track_graph
from tests.test_stream_templates import BASE as GAME

PERSON = {"layout": "person", "crop_w": 1080, "crop_h": 1920, "y": 0, "keyframes": [[0.0, 300.0], [5.0, 420.0]],
          "closeup": {"size": 800, "y": 200, "x": [[0.0, 500.0]], "intervals": [[1.0, 2.0]]},
          "split": {"w": 900, "h": 800, "intervals": [[3.0, 4.0]],
                    "top": {"x": [[0.0, 10.0]], "y": [[0.0, 20.0]]}, "bottom": {"x": [[0.0, 900.0]], "y": [[0.0, 40.0]]}}}
NVENC = ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "19", "-b:v", "8M", "-maxrate", "8M", "-bufsize", "16M",
         "-pix_fmt", "yuv420p"]


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setenv("SONYA_RENDER_ARGS", json.dumps(NVENC))
    Q._BASE.clear()
    yield
    sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT = 1080, 1920
    Q._BASE.clear()


def viral(q, clip="viral"):
    return {"clipType": clip, "settings": {"quality": q}}


def test_requested_and_resolve():
    assert Q.requested({}) == "standard" and Q.requested(None) == "standard"
    assert Q.requested(viral("high")) == "high" and Q.requested({"quality": "2k"}) == "high"
    assert Q.requested(viral("ultra")) == "standard"
    assert Q.resolve("virality", viral("high")) == "high"
    assert Q.resolve("virality", viral("high", "storytelling")) == "high"
    assert Q.resolve("virality", viral("high", "hooks")) == "high"
    # layouts / overlays drawn in 1080 px: stay standard
    assert Q.resolve("streamer", viral("high", "streamer")) == "standard"
    assert Q.resolve("educational", viral("high", "educational")) == "standard"
    assert Q.resolve("virality", viral("high", "trailer")) == "standard"
    assert Q.resolve("beep", viral("high", "")) == "standard"


def test_off_switch(monkeypatch):
    monkeypatch.setenv("SONYA_QUALITY_HIGH", "off")
    assert Q.resolve("virality", viral("high")) == "standard"


def test_standard_changes_nothing():
    before = track_graph(PERSON), track_graph(GAME)
    assert Q.apply("virality", viral("standard")) == "standard"
    assert (sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT) == (1080, 1920)
    assert json.loads(os.environ["SONYA_RENDER_ARGS"]) == NVENC
    assert (track_graph(PERSON), track_graph(GAME)) == before


def test_high_size_bitrate_and_back():
    assert Q.apply("virality", viral("high")) == "high"
    assert Q.size() == (1440, 2560)
    args = json.loads(os.environ["SONYA_RENDER_ARGS"])
    assert args[args.index("-b:v") + 1] == "24M" and args[args.index("-maxrate") + 1] == "24M"
    assert args[args.index("-bufsize") + 1] == "48M" and args[args.index("-preset") + 1] == "p7"
    g = track_graph(PERSON)
    assert "scale=1440:2560" in g and "scale=1440:1440" in g and "scale=1440:1280" in g
    assert "scale=1080" not in g and ":1920:" not in g.replace("h=1920:", "")
    # the next job of this process is standard again: exactly the old recipe
    assert Q.apply("virality", viral("standard")) == "standard"
    assert Q.size() == (1080, 1920) and json.loads(os.environ["SONYA_RENDER_ARGS"]) == NVENC


def test_high_without_render_args_restores_unset(monkeypatch):
    monkeypatch.delenv("SONYA_RENDER_ARGS")
    monkeypatch.setattr("scripts.shared.cut_profile.encoder_family", lambda ff=None: "cpu")
    Q.apply("virality", viral("high"))
    args = json.loads(os.environ["SONYA_RENDER_ARGS"])
    assert args[:2] == ["-c:v", "libx264"] and args[args.index("-maxrate") + 1] == "24M"
    Q.apply("virality", viral("standard"))
    assert "SONYA_RENDER_ARGS" not in os.environ


def test_game_layout_upscaled_at_2k():
    base = track_graph(GAME)
    sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT = 1440, 2560
    g = track_graph(GAME)
    assert g == f"{base},scale=1440:2560:flags=lanczos,setsar=1"
    assert (sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT) == (1440, 2560)


def test_closeup_top_scales():
    sca.OUTPUT_WIDTH, sca.OUTPUT_HEIGHT = 1440, 2560
    g = track_graph(PERSON)
    assert f"overlay=0:{(2560 - 1440) // 2 - 107}" in g


def test_swap_bitrate_keeps_uncapped():
    assert Q._swap_bitrate(["-b:v", "0", "-cq", "19"], "24M") == ["-b:v", "0", "-cq", "19"]
    assert Q.encoder_args(["-c:v", "libx264", "-crf", "18"], "high")[-4:] == ["-maxrate", "24M", "-bufsize", "48M"]
    assert Q.encoder_args(NVENC, "standard") is NVENC


def test_source_fetch_high_raises_height(monkeypatch, tmp_path):
    from scripts import source_fetch as sf_mod
    from scripts import url_ingest
    seen = {}

    class Stop(Exception):
        pass

    def fake_dl(url, platform, **kw):
        seen.update(kw, env=os.environ.get("SONYA_YTDLP_MAX_HEIGHT"))
        raise Stop()
    monkeypatch.setattr(url_ingest, "assert_safe_url", lambda u: None)
    monkeypatch.setattr(url_ingest, "probe", lambda *a, **k: {"duration": 3000})
    monkeypatch.setattr(url_ingest, "download_video", fake_dl)
    monkeypatch.setattr(sf_mod, "_check_free_space", lambda *a: None)
    monkeypatch.delenv("SONYA_YTDLP_MAX_HEIGHT", raising=False)
    sf_mod._BASE_HEIGHT.clear()
    src = {"url": "https://youtu.be/x", "platform": "youtube", "max_source_sec": 14400}
    with pytest.raises(Stop):
        sf_mod.fetch_source(dict(src, quality="high"), "virality", "k", tmp_path)
    assert seen["max_height"] == 2160 and seen["env"] == "2160"
    seen.clear()
    with pytest.raises(Stop):
        sf_mod.fetch_source(src, "virality", "k", tmp_path)        # standard, 50 min: speedfix 1080p as before
    assert seen["max_height"] == 1080 and seen["env"] == "1440"
