"""
Streamer v2 runner split (modes/streamer/runner.py): analyze() / compose_one() / run().

No ffmpeg, ML models or network: the heavy calls (enrich_video_for_mode = ASR, the stream signals, the
layout survey, the per-clip analysis and the render) are monkeypatched, so the runner's own logic is
exercised for real. v1's yuvelirochka analyzer never existed (ImportError → «first 30 s» fallback); v2
builds moments from the transcript + signals (scripts/shared/stream_moments.py).
"""
from __future__ import annotations

import pytest

from modes.streamer import runner


def _words(n_sent=40, sent_len=8, start=130.0):
    """Sentences of `sent_len` words, 0.4 s per word, 1.5 s pause between sentences."""
    out, t = [], start
    for s in range(n_sent):
        for i in range(sent_len):
            w = f"слово{s}_{i}" + ("!" if i == sent_len - 1 else "")
            out.append({"word": w, "start": round(t, 2), "end": round(t + 0.35, 2)})
            t += 0.4
        t += 1.5
    return out


WORDS = _words()


@pytest.fixture
def patched(monkeypatch, tmp_path):
    calls = {"enrich": 0, "render": [], "analysis": 0}

    def enrich(**kw):
        calls["enrich"] += 1
        return {"word_timestamps": list(WORDS), "warnings": []}

    monkeypatch.setattr("scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode", enrich)
    monkeypatch.setattr(runner, "_source_info", lambda p: {"width": 1920, "height": 1080, "duration": 600.0})
    monkeypatch.setattr(runner, "_layout_survey", lambda *a, **k: {"rects": []})
    from scripts.shared import stream_signals as ss
    n = 600
    loud = [0.0] * n
    for t in range(300, 312):
        loud[t] = 4.0                              # one loud moment at 300 s
    monkeypatch.setattr(ss, "audio_signals", lambda *a, **k: {"loud": loud, "pitch": [0.0] * n})
    monkeypatch.setattr(ss, "visual_signals", lambda *a, **k: {"flash": [0.0] * n, "cut": [0.0] * n})

    def render(src, out, start, dur, track, subtitle_path=None, report=None):
        calls["render"].append({"start": start, "dur": dur, "layout": track.get("layout")})
        if report is not None:
            report.update(render_encoder="cpu", render_s=0.1)
        open(out, "wb").write(b"x")
        return out

    monkeypatch.setattr("scripts.shared.crop.smart_crop_adapter.render_tracked", render)

    def clip_an(crop, src, start, dur):
        calls["analysis"] += 1
        return None, None, "no_model: test"

    monkeypatch.setattr("modes.virality.runner._clip_analysis", clip_an)
    monkeypatch.setattr("scripts.shared.cut_profile.ensure_render_args", lambda *a, **k: None)
    monkeypatch.setattr("scripts.shared.crop.stream_layout.analyze_layout",
                        lambda *a, **k: {"webcam": None, "chat": []})
    return calls


PARAMS = {"source_sha256": "x" * 64, "editor": False, "settings": {"subtitleStyle": "none"}}


def test_analyze_calls_enrichment_exactly_once(patched, tmp_path):
    runner.analyze("input.mp4", str(tmp_path), params=dict(PARAMS))
    assert patched["enrich"] == 1


def test_analyze_returns_moments_with_canonical_contract(patched, tmp_path):
    res = runner.analyze("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=1))
    segs = res["segments"]
    assert len(segs) > 1                            # the whole pool, not only max_clips
    for s in segs:
        assert {"start_sec", "duration_sec", "score", "source"} <= set(s)
        assert isinstance(s["start_sec"], float) and isinstance(s["duration_sec"], float)
        assert 20.0 <= s["duration_sec"] <= 62.5
        assert s["source"].startswith("stream_moment")
    # the loud moment at 300 s is the top pick
    assert segs[0]["start_sec"] <= 300 <= segs[0]["start_sec"] + segs[0]["duration_sec"]
    assert res["crop_hints"]["version"] == runner.STREAMER_VERSION
    assert res["selection"]["chat"]["status"] == "no_chat"


def test_analyze_without_words_fails_clearly(monkeypatch, patched, tmp_path):
    monkeypatch.setattr("scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
                        lambda **kw: {"word_timestamps": [], "warnings": ["asr down"]})
    with pytest.raises(RuntimeError, match="ASR_FAILED"):
        runner.analyze("input.mp4", str(tmp_path), params=dict(PARAMS))


def test_compose_one_does_not_call_enrichment_and_renders_the_segment(patched, tmp_path):
    out = str(tmp_path / "clip.mp4")
    seg = {"start_sec": 4350.0, "duration_sec": 37.0}
    assert runner.compose_one("input.mp4", out, seg, crop_hints={"anchor": "center"}) == out
    assert patched["enrich"] == 0
    assert patched["render"] == [{"start": 4350.0, "dur": 37.0, "layout": "person"}]
    assert seg["render_encoder"] == "cpu" and seg["crop_layout"] == "person"


def test_run_analyses_once_and_renders_top_clips_in_rank_order(patched, tmp_path):
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=2))
    assert patched["enrich"] == 1
    assert len(res["clips"]) == 2 and len(patched["render"]) == 2
    assert patched["render"][0]["start"] <= 300 <= patched["render"][0]["start"] + patched["render"][0]["dur"]
    assert res["mode"] == "streamer" and res["version"] == runner.STREAMER_VERSION and res["beta"] is True
    assert {"clips", "mode", "beta", "selection", "warnings", "webcam_boxes_found", "active_speaker_segs"} <= set(res)


def test_run_reports_per_clip_failure(monkeypatch, patched, tmp_path):
    real = runner.compose_one
    n = {"i": 0}

    def flaky(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(runner, "compose_one", flaky)
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=3))
    assert len(res["clips"]) == 2
    assert any(w.startswith("clip_1_failed") for w in res["warnings"])
