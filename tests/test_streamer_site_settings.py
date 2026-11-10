"""Site → worker mapping: settings.streamer_layout (1..5) and settings.moment_types (list)."""
import pytest

from modes.streamer.runner import streamer_crop
from scripts.shared.stream_types import normalize_types


@pytest.mark.parametrize("settings,want", [({"streamer_layout": 3}, 3), ({"streamer_layout": "5"}, 5),
                                            ({}, 1), ({"streamer_layout": 9}, 1)])
def test_streamer_layout_from_settings(tmp_path, settings, want):
    (tmp_path / "x.mp4").write_bytes(b"x")
    crop = streamer_crop({"settings": settings}, str(tmp_path / "x.mp4"), str(tmp_path))
    assert crop["layout_template"] == want


def test_params_layout_wins_over_settings(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"x")
    crop = streamer_crop({"layout_template": 2, "settings": {"streamer_layout": 4}}, str(tmp_path / "x.mp4"), str(tmp_path))
    assert crop["layout_template"] == 2


def test_moment_types_list_and_auto():
    assert normalize_types(["chaos", "philosophy"]) == ["philosophy", "chaos"] or set(normalize_types(["chaos", "philosophy"])) == {"chaos", "philosophy"}
    assert normalize_types(["auto"]) == [] and normalize_types([]) == []
