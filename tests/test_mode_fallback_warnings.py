"""
Cut (virality/hooks), Stories and Educational runners import mode classes
that do not exist in scripts/legacy_gpu (HookMode, StoryMode +
analyze_segments, EducationalMode), so today they always take their
fallback path. These tests pin that the fallback is at least reported in
the result's "warnings" instead of being silent. Heavy calls are
monkeypatched at their source modules (same approach as
test_streamer_runner_split.py).
"""
from __future__ import annotations

import importlib

import pytest

from scripts.shared.crop import smart_crop_adapter
from scripts.shared.enhancers import sonya_enhancer


@pytest.mark.parametrize("mode,key", [
    ("virality", "virality_engine_unavailable"),
    ("stories", "stories_engine_unavailable"),
    ("educational", "educational_engine_unavailable"),
])
def test_engine_fallback_is_reported(monkeypatch, tmp_path, mode, key):
    monkeypatch.setattr(sonya_enhancer, "enrich_video_for_mode",
                        lambda **kw: {"warnings": [], "word_timestamps": [], "crop_hints": {}})
    monkeypatch.setattr(smart_crop_adapter, "compose_vertical_clip", lambda **kw: None)
    runner = importlib.import_module(f"modes.{mode}.runner")
    result = runner.run("in.mp4", str(tmp_path), {"max_clips": 1})
    assert any(w.startswith(key) for w in result["warnings"]), result["warnings"]
    assert result["clips"]  # fallback still produces a clip, as before


@pytest.mark.parametrize("mode,prefix", [
    ("virality", "viral_clip_"), ("stories", "story_clip_"), ("educational", "edu_clip_"),
    ("trailer_film_breaker", "trailer_clip_"), ("streamer", "stream_clip_"),
])
def test_selected_segments_json_records_intervals(monkeypatch, tmp_path, mode, prefix):
    import json
    words = [{"word": "w", "start": float(t), "end": float(t) + 0.4} for t in range(0, 40)]
    monkeypatch.setattr(sonya_enhancer, "enrich_video_for_mode",
                        lambda **kw: {"warnings": [], "word_timestamps": words, "crop_hints": {},
                                      "active_speaker_segments": [{"start": 5, "end": 25}],
                                      "webcam_layout": {"boxes": []}})
    failed = {"n": 0}

    def compose(**kw):
        failed["n"] += 1
        if failed["n"] == 2:
            raise RuntimeError("second clip fails")
    monkeypatch.setattr(smart_crop_adapter, "compose_vertical_clip", compose)
    runner = importlib.import_module(f"modes.{mode}.runner")
    runner.run("in.mp4", str(tmp_path), {"max_clips": 3})
    data = json.loads((tmp_path / "selected_segments.json").read_text(encoding="utf-8"))
    assert data["mode"] == mode and data["segments"]
    for seg in data["segments"]:
        assert 0 <= seg["start"] < seg["end"]
    files = [s["file"] for s in data["segments"]]
    assert files[0].startswith(prefix)
    if len(files) > 1:
        assert files[1] is None  # failed clip is recorded as such, not shifted


def test_write_selected_segments_never_raises(tmp_path):
    from scripts.shared.mode_output import write_selected_segments
    assert write_selected_segments(str(tmp_path / "missing" / "dir"), "x", [{"start": 0}], []) is None
