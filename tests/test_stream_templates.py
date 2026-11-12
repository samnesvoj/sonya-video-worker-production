"""Streamer layout templates 1..5 (scripts/shared/crop/stream_templates)."""
import pytest

from scripts.shared.crop.smart_crop_adapter import track_graph
from scripts.shared.crop.stream_layout import MAX_UPSCALE
from scripts.shared.crop.stream_templates import apply_template, clean_nick, normalize_template, TEMPLATES

BASE = {"layout": "game", "crop_version": "game-v3", "crop_status": "tracked", "source_width": 1920,
        "source_height": 1080, "game": [0, 0, 1920, 1080], "game_masks": [[0, 0, 500, 300]], "game_h": 1080 * 1080 // 1920,
        "top_h": 1312, "cam": [10, 10, 540, 300], "cam_size": [1080, 600], "cam_pos": [0, 356], "cam_src": [8, 8, 544, 304],
        "face": [280.0, 160.0, 90.0, 110.0], "seam_y": 1312}


def test_normalize_template():
    assert [normalize_template(v) for v in (1, "3", "cinema", 9, None, "x", 0)] == [1, 3, 4, 1, 1, 1, 1]


def test_template_1_is_unchanged_graph():
    t = apply_template(BASE, 1)
    assert t["template"] == 1 and track_graph(t) == track_graph(BASE)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_each_template_builds_and_never_upscales_over_2x(n):
    t = apply_template(BASE, n, nick="Buster")
    assert t["template"] == n and t["template_name"] == TEMPLATES[n]
    assert t["cam_upscale"] <= MAX_UPSCALE + 1e-6
    g = track_graph(t)
    assert "overlay" in g or "vstack" in g
    assert 0 < t["seam_y"] < 1920


def test_template_3_is_60_40():
    t = apply_template(BASE, 3)
    assert (t["top_h"], t["game_h"]) == (1152, 768)
    gx, gy, gw, gh = t["game"]
    assert abs(gw / gh - 1080 / 768) < 0.02 and gx > 0           # sides trimmed, masks stay inside the box
    assert all(m[0] + m[2] <= gw and m[1] + m[3] <= gh for m in t["game_masks"])


def test_nick_only_in_5_and_sanitized():
    assert clean_nick("Бустер_1 'x'; [y]%") == "Бустер_1 x y"
    assert "drawtext" in track_graph(apply_template(BASE, 5, nick="Buster"))
    assert "drawtext" not in track_graph(apply_template(BASE, 5))
    assert "drawtext" not in track_graph(apply_template(BASE, 4, nick="Buster"))


def test_other_layouts_untouched():
    person = {"layout": "person", "crop_w": 10}
    assert apply_template(person, 4)["layout"] == "person" and apply_template(None, 4) is None
