"""Virality = the Mac mini recipe only: story selection → word-exact boundaries → tracked render.
The old path (hook_mode 0–30 s fallback, Gemini selection, letterbox composer) is gone."""
from __future__ import annotations

import pytest

from modes.virality import runner

RECIPE = {"max_clips": 1, "selection": "story", "story": {"strategy": "qa"},
          "crop": {"mode": "tracked"}}


def _interview():
    # question → long answer, repeated: complete sentences every 8 words
    out, t, i = [], 0.2, 0
    while t < 200:
        end = "?" if i % 40 == 7 else ("." if i % 8 == 7 else "")
        out.append({"word": f"слово{i}{end}", "start": round(t, 3), "end": round(t + 0.35, 3)})
        t += 0.5
        i += 1
    return out


def test_story_clip_is_snapped_to_words_and_rendered_tracked(monkeypatch, tmp_path):
    words = _interview()
    got = {}

    def fake_tracked(video, selected, out_dir, crop, w):
        got.update(selected=selected, crop=crop, words=w)
        return [str(out_dir / "viral_clip_01.mp4")]
    monkeypatch.setattr(runner, "_compose_tracked", fake_tracked)
    monkeypatch.setattr("scripts.shared.audio_features.loudness_per_second",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg in unit tests")))
    clips, selection = runner._select_and_compose("in.mp4", {"word_timestamps": words, "warnings": []},
                                                  dict(RECIPE), tmp_path)
    assert clips == [str(tmp_path / "viral_clip_01.mp4")]
    seg = got["selected"][0]
    spans = [(w["start"], w["end"]) for w in words]
    for t in (seg["start"], seg["start"] + seg["duration"]):
        assert not any(s + 1e-6 < t < e - 1e-6 for s, e in spans)
    assert seg["boundary_version"] == "boundaries-v1"
    assert got["crop"]["mode"] == "tracked" and got["words"] == words


def test_no_words_fails_the_job_instead_of_guessing(tmp_path):
    with pytest.raises(RuntimeError, match="ASR_FAILED"):
        runner._select_and_compose("in.mp4", {"word_timestamps": [], "warnings": ["asr: boom"]},
                                   dict(RECIPE), tmp_path)


def test_no_story_candidates_fails_the_job(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.shared.cut_selection.select_story_clips", lambda *a, **k: ([], {}))
    with pytest.raises(RuntimeError, match="NO_CLIPS"):
        runner._select_and_compose("in.mp4", {"word_timestamps": _interview(), "warnings": []},
                                   dict(RECIPE, story={}), tmp_path)


@pytest.mark.parametrize("params", [
    {"max_clips": 1},                                              # no Cut settings
    {"selection": "story", "crop": {"mode": "legacy"}},            # old composer
    dict(RECIPE, llm="on"),                                        # old Gemini selection
])
def test_anything_but_the_recipe_is_refused(params, tmp_path):
    with pytest.raises(ValueError, match="Mac mini recipe"):
        runner._select_and_compose("in.mp4", {"word_timestamps": _interview()}, params, tmp_path)
