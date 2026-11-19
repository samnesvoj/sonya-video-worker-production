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
    monkeypatch.setenv("SONYA_SUBTITLES_ALLOW_OFF", "1")   # these split tests run without subtitles
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

    def render(src, out, pieces, track, clip_start, subtitle_path=None, report=None, **kw):
        start = pieces[0][0]
        dur = round(pieces[-1][0] + pieces[-1][1] - start, 3)
        calls["render"].append({"start": start, "dur": dur, "layout": track.get("layout")})
        if report is not None:
            report.update(render_encoder="cpu", render_s=0.1)
        open(out, "wb").write(b"x")
        return out

    monkeypatch.setattr("scripts.shared.crop.smart_crop_adapter.render_pieces", render)

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


def test_analyze_no_speech_but_loud_audio_still_gives_clips(monkeypatch, patched, tmp_path):
    """ASR worked but heard no words (a game / music stream): clips from the audio energy, no NO_CLIPS."""
    monkeypatch.setattr("scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
                        lambda **kw: {"word_timestamps": [], "warnings": []})
    res = runner.analyze("input.mp4", str(tmp_path), params=dict(PARAMS))
    assert res["segments"] and any(s["start_sec"] <= 300 <= s["start_sec"] + s["duration_sec"]
                                   for s in res["segments"])
    assert any(w.startswith("asr_no_words") for w in res["warnings"])


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
    # 2026-11-18: a failed clip is rendered again (template 1, then the plain crop) instead of being lost
    assert len(res["clips"]) == 3
    assert any(w.startswith("clip_1_failed") for w in res["warnings"])


def test_run_falls_back_to_template_1_then_plain_crop(monkeypatch, patched, tmp_path):
    real = runner.compose_one
    seen = []

    def picky(video, out, seg, crop_hints=None):
        c = (crop_hints or {}).get("crop") or {}
        seen.append((c.get("layout_template"), c.get("layout")))
        if c.get("layout") != "person":
            raise UnboundLocalError("cannot access local variable 'cam'")
        return real(video, out, seg, crop_hints=crop_hints)

    monkeypatch.setattr(runner, "compose_one", picky)
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=2,
                                                            settings={"streamer_layout": 4}))
    assert len(res["clips"]) == 2
    assert seen[:3] == [(4, seen[0][1]), (1, seen[0][1]), (1, "person")]


def test_run_loses_a_clip_only_when_every_fallback_fails(monkeypatch, patched, tmp_path):
    real = runner.compose_one
    n = {"i": 0}

    def bad_first(video, out, seg, crop_hints=None):
        if out.endswith("_01.mp4"):
            raise RuntimeError("boom")
        return real(video, out, seg, crop_hints=crop_hints)

    monkeypatch.setattr(runner, "compose_one", bad_first)
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=3))
    assert len(res["clips"]) == 2
    assert sum(w.startswith("clip_1_failed") for w in res["warnings"]) == 3


def test_run_exclude_ranges_skips_moments_already_delivered(patched, tmp_path):
    first = runner.run("input.mp4", str(tmp_path / "a"), params=dict(PARAMS, max_clips=1))
    got = patched["render"][0]
    patched["render"].clear()
    res = runner.run("input.mp4", str(tmp_path / "b"),
                     params=dict(PARAMS, max_clips=1, exclude_ranges=[[got["start"], got["start"] + got["dur"]], ["x"]]))
    assert first["clips"] and len(patched["render"]) <= 1
    for r in patched["render"]:
        assert r["start"] >= got["start"] + got["dur"] or r["start"] + r["dur"] <= got["start"]


# ── one layout per job (owner 2026-11-19 «некоторые клипы в другом режиме») ──────────────────────────────────

CAM = [0, 0, 490, 274]


def _cam_on(monkeypatch, has_cam):
    """analyze_layout: the webcam border is seen only in the clips for which has_cam(start) is true."""
    monkeypatch.setattr("scripts.shared.crop.stream_layout.analyze_layout",
                        lambda video, start, *a, **k: {"webcam": {"rect": list(CAM), "by": "lines"} if has_cam(start)
                                                       else None, "chat": []})


def _looks(tmp_path):
    import json
    d = json.loads((tmp_path / "selected_segments.json").read_text())
    segs = d.get("segments", d) if isinstance(d, dict) else d
    return [(s.get("crop_layout"), s.get("layout_template")) for s in segs]


def test_run_clip_without_its_own_webcam_takes_the_jobs_webcam(monkeypatch, patched, tmp_path):
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    starts = []
    _cam_on(monkeypatch, lambda st: starts.append(st) or len(starts) != 2)     # the 2nd clip: no border
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=4, settings={"streamer_layout": 4}))
    assert len(res["clips"]) == 4
    assert set(_looks(tmp_path)) == {("game", 4)}
    assert any(w.startswith("layout_unified") for w in res["warnings"])


def test_run_template_that_fails_for_one_clip_moves_all_clips_to_one_layout(monkeypatch, patched, tmp_path):
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    _cam_on(monkeypatch, lambda st: True)
    real = runner.compose_one
    first = {}

    def picky(video, out, seg, crop_hints=None):
        c = (crop_hints or {}).get("crop") or {}
        first.setdefault("start", seg["start_sec"])
        if seg["start_sec"] == first["start"] and c.get("layout_template") == 4:
            raise RuntimeError("template 4 cannot render this clip")
        return real(video, out, seg, crop_hints=crop_hints)

    monkeypatch.setattr(runner, "compose_one", picky)
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=3, settings={"streamer_layout": 4}))
    assert len(res["clips"]) == 3
    assert set(_looks(tmp_path)) == {("game", 1)}                    # all three, not 2 × template 4 + 1 × template 1


def test_run_lonely_webcam_detection_is_not_forced_on_the_other_clips(monkeypatch, patched, tmp_path):
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    starts = []
    _cam_on(monkeypatch, lambda st: starts.append(st) or len(starts) == 1)     # only the 1st of 6 clips «sees» a cam
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=6, settings={"streamer_layout": 2}))
    assert res["clips"] and set(_looks(tmp_path)) == {("person", None)}


def test_run_uniform_job_is_not_rendered_twice(monkeypatch, patched, tmp_path):
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "1")
    _cam_on(monkeypatch, lambda st: True)
    res = runner.run("input.mp4", str(tmp_path), params=dict(PARAMS, max_clips=3, settings={"streamer_layout": 4}))
    assert len(patched["render"]) == len(res["clips"]) == 3
    assert not any(w.startswith("layout_unified") for w in res["warnings"])


def test_consensus_cam_needs_two_clips_or_the_survey():
    g = lambda r: {"crop_layout": "game", "webcam_rect": r}
    assert runner.consensus_cam([g(CAM), g([2, 0, 488, 274]), {"crop_layout": "person"}], 3) is not None
    assert runner.consensus_cam([g(CAM)] + [{"crop_layout": "person"}] * 7, 8) is None
    assert runner.consensus_cam([g(CAM)] + [{"crop_layout": "person"}] * 7, 8, [{"rect": CAM, "seen": 3}]) == CAM
