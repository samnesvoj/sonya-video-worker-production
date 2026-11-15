"""Mini ad banner (scripts/shared/ad_banner.py) + worker disk hygiene (scripts/gpu_worker.py)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.shared import ad_banner as ab

ASS_HEAD = """[Script Info]
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sonya,Montserrat Black,82,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,1,0,1,6,0,2,90,90,380,204
Style: Hook,Montserrat Black,62,&H00FAF8F7,&H000000FF,&H59221C1C,&H00000000,0,0,0,0,100,100,0,0,3,22,0,8,90,90,230,204

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def test_kind_of(monkeypatch):
    monkeypatch.delenv("SONYA_AD_BANNER", raising=False)
    assert ab.kind_of({"ad_banner": "mini"}) == "dark"
    assert ab.kind_of({"ad_banner": "MINI_light"}) == "light"
    assert ab.kind_of({"ad_banner": {"kind": "mini"}}) == "dark"
    assert ab.kind_of({}) is None and ab.kind_of(None) is None and ab.kind_of({"ad_banner": "big"}) is None
    monkeypatch.setenv("SONYA_AD_BANNER", "off")
    assert ab.kind_of({"ad_banner": "mini"}) is None


def test_subtitle_bands_default_and_moved():
    ass = ASS_HEAD + "\n".join([
        r"Dialogue: 0,0:00:02.00,0:00:04.00,Sonya,,0,0,0,,ТЫ {\c&H0000E5FF&\fscx112\fscy112}ЗНАЕШЬ{\c&H00FFFFFF&\fscx100\fscy100}",
        r"Dialogue: 0,0:00:04.00,0:00:06.00,Sonya,,0,0,0,,{\an8\pos(540,1590)}НИЖЕ\NТЕКСТА",
        r"Dialogue: 1,0:00:00.00,0:00:03.20,Hook,,0,0,0,,{\fad(150,300)}ЗАГОЛОВОК",
    ])
    b = ab.subtitle_bands(ass, 1920)
    assert len(b) == 3
    t0, t1, top, bot = b[0]
    assert (t0, t1) == (2.0, 4.0) and bot == pytest.approx(1540 + 0, abs=13) and top < 1540 - 82
    assert b[1][2] == pytest.approx(1590) and b[1][3] > 1590 + 2 * 82
    assert b[2][2] == pytest.approx(230)                     # top-aligned hook: from MarginV down


def test_plan_below_subtitles_full_size():
    bands = [(0.0, 20.0, 1430.0, 1540.0)]
    p = ab.plan(1080, 1920, 30.0, bands)
    assert p["status"] == "ok" and p["at"] == 3.0
    assert p["card_w"] == ab.card_limits(1080, 1920)[0]    # ≈ 56 % of the width
    assert 0.5 <= p["card_w"] / 1080 <= 0.6
    assert p["card_y"] >= 1540 + ab.TEXT_GAP * 1920 - 1      # under the text
    assert p["card_y"] + p["card_h"] <= 1920 - ab.BOTTOM_MARGIN * 1920 + 1


def test_plan_shrinks_when_text_is_low_then_skips():
    low = [(0.0, 20.0, 1500.0, 1640.0)]                      # subtitles moved down: less room
    p = ab.plan(1080, 1920, 30.0, low)
    assert p["status"] == "ok" and p["card_w"] < ab.card_limits(1080, 1920)[0]
    assert p["card_y"] >= 1640
    lower = [(0.0, 20.0, 1700.0, 1820.0)]                    # no room at all: never covers the text
    q = ab.plan(1080, 1920, 30.0, lower)
    assert q == {"status": "skipped", "reason": "no_room_below_subtitles"}


def test_plan_moves_to_a_moment_without_low_text_and_avoids_faces():
    bands = [(2.5, 8.0, 1500.0, 1700.0), (8.0, 40.0, 1430.0, 1540.0)]
    p = ab.plan(1080, 1920, 40.0, bands, starts=(3.0, 8.5))
    assert p["status"] == "ok" and p["at"] == 8.5
    faces = [(t / 2, [(400.0, 1450.0, 250.0, 300.0)]) for t in range(0, 40)]   # a face low in the frame, centre
    q = ab.plan(1080, 1920, 30.0, [], faces)
    assert q["status"] == "skipped" and q["reason"] == "face_low_in_frame"
    side = [(t / 2, [(0.0, 1450.0, 150.0, 300.0)]) for t in range(0, 40)]      # beside the card column: fine
    assert ab.plan(1080, 1920, 30.0, [], side)["status"] == "ok"


def test_plan_short_clip_and_landscape():
    assert ab.plan(1080, 1920, 4.0, [])["status"] == "skipped"
    p = ab.plan(1080, 1920, 7.0, [])
    assert p["status"] == "ok" and p["at"] + ab.BANNER_SEC <= 7.0
    land = ab.plan(1920, 1080, 60.0, [])
    assert land["status"] == "ok" and land["card_w"] <= 0.36 * 1080 + 1  # small on 16:9


def test_ensure_asset_verifies_sha(tmp_path, monkeypatch):
    monkeypatch.delenv("SONYA_AD_BANNER_ASSET", raising=False)
    monkeypatch.setenv("SONYA_ASSETS_DIR", str(tmp_path / "assets"))
    data = b"banner-bytes"
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"items": [{"theme": "dark", "file": "b.mov", "s3_key": "worker-v2/assets/ad-mini/b.mov",
                                          "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}]}))
    calls = []

    def dl(key, dst):
        calls.append(key)
        Path(dst).write_bytes(data)
    p = ab.ensure_asset("dark", manifest=man, download=dl)
    assert p.read_bytes() == data and calls == ["worker-v2/assets/ad-mini/b.mov"]
    assert ab.ensure_asset("dark", manifest=man, download=dl) == p and len(calls) == 1   # cached
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"items": [{"theme": "dark", "file": "c.mov", "s3_key": "k", "sha256": "0" * 64, "size": 1}]}))
    with pytest.raises(RuntimeError, match="sha256"):
        ab.ensure_asset("dark", manifest=bad, download=dl)
    assert not list((tmp_path / "assets" / "ad-mini").glob("c.mov*"))


def test_manifest_pins_new_objects_only():
    items = json.loads(ab.MANIFEST.read_text())["items"]
    assert {i["theme"] for i in items} == {"dark", "light"}
    for i in items:
        assert i["s3_key"].startswith("worker-v2/assets/ad-mini/") and i["sha256"][:12] in i["s3_key"]
        assert len(i["sha256"]) == 64 and i["license"] == "own-generated"


def test_apply_to_outputs_off_is_noop_and_drops_ass(tmp_path, monkeypatch):
    clip = tmp_path / "clip_01.mp4"
    clip.write_bytes(b"x")
    ass = tmp_path / "clip_01.ass"
    ass.write_text("x")
    assert ab.apply_to_outputs(str(tmp_path), "virality", {}) == {}
    assert ass.exists() and clip.read_bytes() == b"x"
    assert ab.apply_to_outputs(str(tmp_path), "sonya_gen", {"ad_banner": "mini"}) == {}
    ab.apply_to_outputs(str(tmp_path), "streamer", {}, drop_ass=True)
    assert not ass.exists()


def test_apply_to_outputs_never_raises(tmp_path, monkeypatch):
    clip = tmp_path / "clip_01.mp4"
    clip.write_bytes(b"not a video")
    monkeypatch.setenv("SONYA_AD_BANNER_ASSET", str(tmp_path / "none.mov"))
    rep = ab.apply_to_outputs(str(tmp_path), "virality", {"ad_banner": "mini"})
    assert rep["clip_01.mp4"]["status"] == "failed" and clip.read_bytes() == b"not a video"


def test_overlay_filter_geometry():
    p = {"at": 3.0, "card_w": 604, "card_h": 302, "card_x": 238, "card_y": 1570}
    f = ab.overlay_filter(p, "30/1")
    assert "scale=674:372" in f and "overlay=x=203" in f and "setpts=PTS-STARTPTS+3.000/TB" in f


# ── worker disk hygiene ─────────────────────────────────────────────────────

def test_sweep_stale_tmp_keeps_live_and_own(tmp_path):
    from scripts import gpu_worker as gw
    dead = tmp_path / "sonya_job1_p999999_abc"
    live = tmp_path / f"sonya_job2_p{os.getppid()}_abc"
    own = tmp_path / f"sonya_job3_p{os.getpid()}_abc"
    old_untagged = tmp_path / "sonya_job4_abc"
    new_untagged = tmp_path / "sonya_job5_abc"
    for d in (dead, live, own, old_untagged, new_untagged):
        d.mkdir()
    os.utime(old_untagged, (1, 1))
    assert gw.sweep_stale_tmp(str(tmp_path)) == 2
    assert not dead.exists() and not old_untagged.exists()
    assert live.exists() and own.exists() and new_untagged.exists()


def test_job_tmp_redirect(tmp_path):
    import tempfile
    from scripts import gpu_worker as gw
    before_env, before = os.environ.get("TMPDIR"), tempfile.tempdir
    gw._job_tmp_enter(tmp_path)
    try:
        assert tempfile.gettempdir() == str(tmp_path / "t") and os.environ["TMPDIR"] == str(tmp_path / "t")
    finally:
        gw._job_tmp_exit()
    assert os.environ.get("TMPDIR") == before_env and tempfile.tempdir == before


def test_disk_gate(monkeypatch):
    from scripts import gpu_worker as gw
    monkeypatch.setattr(gw, "_free_bytes", lambda path=None: 2 << 30)
    monkeypatch.setattr(gw, "sweep_stale_tmp", lambda *a, **k: 0)
    monkeypatch.setenv("SONYA_MIN_FREE_GB_TO_CLAIM", "6")
    assert gw._disk_ok_to_claim() is False
    monkeypatch.setenv("SONYA_MIN_FREE_GB_TO_CLAIM", "1")
    assert gw._disk_ok_to_claim() is True
