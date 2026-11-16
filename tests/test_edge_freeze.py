"""A shot cut in the first 0.6 s / last 0.45 s is hidden by a still (production 2026-11-15: 10/40 clips flashed
the previous shot at the start, 7/40 the next shot at the end)."""
import shutil
import subprocess

import pytest

from scripts.shared.crop import smart_crop_adapter as sca


def test_edge_freeze_rules(monkeypatch):
    monkeypatch.delenv("SONYA_EDGE_FREEZE", raising=False)
    assert sca.edge_freeze([0.21, 6.05, 30.53], 39.96) == (0.21, 0.0)
    assert sca.edge_freeze([5.0, 39.6], 39.96) == (0.0, 0.36)
    assert sca.edge_freeze([0.4, 20.0, 29.7], 30.0) == (0.4, 0.3)
    assert sca.edge_freeze([0.9, 25.0], 30.0) == (0.0, 0.0)       # a real first shot, not a flash
    assert sca.edge_freeze([], 30.0) == (0.0, 0.0)
    assert sca.edge_freeze(None, 30.0) == (0.0, 0.0)
    monkeypatch.setenv("SONYA_EDGE_FREEZE", "off")
    assert sca.edge_freeze([0.21], 30.0) == (0.0, 0.0)


def test_graph_empty_without_freeze():
    assert sca.edge_freeze_graph(0, 0, 30) == ""
    g = sca.edge_freeze_graph(0.2, 0.3, 30)
    assert "trim=start=0.200:end=29.700" in g and "start_mode=clone" in g and "stop_mode=clone" in g


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_render_hides_head_flash(tmp_path, monkeypatch):
    monkeypatch.delenv("SONYA_EDGE_FREEZE", raising=False)
    src = tmp_path / "src.mp4"
    # 0.2 s red (the previous shot), then 3 s blue; a tone for audio
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=640x360:r=25:d=0.2",
                    "-f", "lavfi", "-i", "color=blue:s=640x360:r=25:d=3", "-f", "lavfi", "-i", "sine=f=440:d=3.2",
                    "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    out = tmp_path / "out.mp4"
    rep = {}
    sca.render_tracked(str(src), str(out), 0.0, 3.2, {"layout": "fit", "cuts": [0.2]},
                       encoder_args=["-c:v", "libx264", "-preset", "ultrafast"], report=rep, freeze_edges=True)
    assert rep["edge_freeze"] == [0.2, 0.0]
    px = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1", "-vf", "crop=8:8:536:956,scale=1:1",
                         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    r, g, b = px[0], px[1], px[2]
    assert b > 150 and r < 80, (r, g, b)        # the first frame is already the blue shot
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                             capture_output=True, text=True).stdout)
    assert abs(d - 3.2) < 0.15
