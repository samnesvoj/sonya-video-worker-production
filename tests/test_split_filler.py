"""split_filler: option parsing, rights gate of the library, segment plan, and a real ffmpeg composition."""
import hashlib
import os
import json
import shutil
import subprocess

import pytest

from scripts.shared import split_filler as sf

FF = shutil.which("ffmpeg")
needs_ff = pytest.mark.skipif(not FF, reason="ffmpeg missing")


def _mk(path, secs, size="320x240", audio=False):
    cmd = [FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s={size}:d={secs}:r=30"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=d={secs}"]
    subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(path)], check=True)


def _lib(tmp_path, rows):
    items = []
    for r in rows:
        f = tmp_path / r["file"]
        if r.pop("make", True):
            _mk(f, 3, audio=r.pop("audio", False))
        r.setdefault("sha256", hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "0")
        items.append(r)
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"items": items}))
    return m


def row(i, kind, **kw):
    return dict({"id": i, "file": f"{i}.mp4", "kind": kind, "license": "own-generated", "watermark": False}, **kw)


def test_normalize():
    assert sf.normalize({}) == {"mode": "none", "filler_id": None, "top_h": 1100}
    assert sf.normalize({"bottom_filler": "ASMR", "filler_id": "x", "filler_top_h": 5000})["top_h"] == 1150
    assert sf.normalize({"bottom_filler": "tiktok"})["mode"] == "none"
    assert sf.normalize({}, {"bottom_filler": "gameplay", "filler_top_h": "999"}) == {"mode": "gameplay", "filler_id": None, "top_h": 1000}
    assert sf.normalize({"filler_top_h": "abc"})["top_h"] == 1100 and sf.normalize({"filler_top_h": 1101})["top_h"] == 1100


def test_library_rights_gate(tmp_path):
    m = _lib(tmp_path, [row("ok", "gameplay"), row("lic", "gameplay", license="unknown"), row("wm", "asmr", watermark=True),
                        row("badsha", "asmr", sha256="deadbeef"), row("gone", "asmr", make=False)])
    assert [e["id"] for e in sf.load_library(str(tmp_path), m)] == ["ok"]
    assert sf.load_library(str(tmp_path), tmp_path / "nope.json") == []


def test_choose_kind_and_determinism(tmp_path):
    lib = sf.load_library(str(tmp_path), _lib(tmp_path, [row("g1", "gameplay"), row("g2", "gameplay"), row("a1", "asmr")]))
    assert sf.choose(lib, "asmr", "g1")["id"] == "a1"            # requested id of the wrong kind is ignored
    assert sf.choose(lib, "gameplay", "g2")["id"] == "g2"
    assert sf.choose(lib, "gameplay", None, seed=3) == sf.choose(lib, "gameplay", None, seed=3)
    assert sf.choose([e for e in lib if e["kind"] == "asmr"], "gameplay") is None


def test_segment_plan():
    long = sf.segment_plan(20.0, 8.0, seed=1)
    assert long["loops"] == 0 and 0 <= long["ss"] <= 12
    short = sf.segment_plan(20.0, 55.0, seed=1)
    assert short["loops"] == 3 and short["ss"] < 20
    assert sf.segment_plan(20.0, 8.0, seed=1) == long


def test_seam_and_top_filter():
    assert sf.seam_intervals(12.34, 1100) == [[0.0, 12.34, 1100]]
    assert "crop=1080:1100:0:" in sf.top_filter(1100)


@needs_ff
@pytest.mark.parametrize("audio_in_filler", [False, True])
def test_apply_end_to_end(tmp_path, audio_in_filler):
    lib = sf.load_library(str(tmp_path), _lib(tmp_path, [row("g1", "gameplay", audio=audio_in_filler)]))
    top = tmp_path / "top.mp4"
    _mk(top, 5, size="1080x1920", audio=True)
    out, rep = tmp_path / "o.mp4", {}
    assert sf.apply(str(top), str(out), {"mode": "gameplay", "filler_id": None, "top_h": 1100}, seed=1, report=rep, library=lib, ffmpeg=FF)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)], capture_output=True, text=True).stdout)
    v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert (v["width"], v["height"]) == (1080, 1920) and len(a) == 1          # only the top clip's audio
    assert abs(float(v["duration"]) - 5) < 0.2
    assert rep["filler_id"] == "g1" and rep["filler_license"] == "own-generated" and rep["filler_loops"] >= 1   # 3 s filler loops to 5 s


@needs_ff
def test_apply_without_usable_filler_returns_false(tmp_path):
    top = tmp_path / "t.mp4"
    _mk(top, 1, size="1080x1920")
    rep = {}
    assert not sf.apply(str(top), str(tmp_path / "o.mp4"), {"mode": "asmr", "filler_id": None, "top_h": 1100}, report=rep, library=[], ffmpeg=FF)
    assert rep["bottom_filler"] == "no_usable_filler"


def test_streamer_crop_wiring(tmp_path):
    from modes.streamer.runner import streamer_crop
    (tmp_path / "x.mp4").write_bytes(b"x")
    c = streamer_crop({"settings": {"bottom_filler": "gameplay"}}, str(tmp_path / "x.mp4"), str(tmp_path))
    assert c["bottom_filler"]["mode"] == "gameplay"
    assert "bottom_filler" not in streamer_crop({}, str(tmp_path / "x.mp4"), str(tmp_path))


def test_shipped_manifest_is_clean():
    rows = json.loads(sf.MANIFEST.read_text())["items"]
    assert sum(r["kind"] == "gameplay" for r in rows) >= 3 and sum(r["kind"] == "asmr" for r in rows) >= 3
    assert all(r["license"] in sf.ALLOWED_LICENSES and not r["watermark"] and not r["audio"] and len(r["sha256"]) == 64
               and r["s3_key"].startswith("worker-v2/assets/fillers/") for r in rows)


def _ass_ff():
    for c in (os.environ.get("SONYA_FFMPEG"), "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", FF):
        if c and shutil.which(c) and " ass " in subprocess.run([c, "-hide_banner", "-filters"], capture_output=True, text=True).stdout:
            return c


@pytest.mark.skipif(not _ass_ff(), reason="ffmpeg with libass missing")
def test_apply_with_subtitles_on_seam(tmp_path):
    lib = sf.load_library(str(tmp_path), _lib(tmp_path, [row("g1", "gameplay")]))
    top = tmp_path / "top.mp4"
    _mk(top, 2, size="1080x1920")
    ass = tmp_path / "s: x's.ass"                    # awkward chars must survive filtergraph escaping
    ass.write_text("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,Alignment\n"
                   "Style: D,Arial,60,&H00FFFFFF,2\n\n[Events]\nFormat: Layer,Start,End,Style,Text\nDialogue: 0,0:00:00.00,0:00:02.00,D,hello\n")
    rep = {}
    assert sf.apply(str(top), str(tmp_path / "o.mp4"), {"mode": "gameplay", "filler_id": None, "top_h": 1100}, ass=str(ass),
                    report=rep, library=lib, ffmpeg=_ass_ff()), rep
