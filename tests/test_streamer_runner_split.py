"""
PHASE A runner split (modes/streamer/runner.py): analyze() / compose_one() /
run() compatibility wrapper.

No real GPU, ffmpeg, or ML models are used — the two genuinely heavy calls
(enrich_video_for_mode, compose_vertical_clip) are monkeypatched at their
source module so run()/analyze()/compose_one() are exercised for real, but
nothing downstream of them does real video work.

Note on the yuvelirochka legacy analyzer/clipper: modes/streamer/legacy/
analyzer.py exposes no module-level `analyze` function (only the
ViralMomentAnalyzer class + a differently-named analyze_transcript), and
clipper.py exposes no `clip_segments` (only the VideoClipper class) — so
`from modes.streamer.legacy.analyzer import analyze` / `... clipper import
clip_segments` already raise ImportError today, in production, unmodified.
analyze() catches that and falls back to _fallback_segments() — which is
therefore the actual code path streamer mode runs today, not a test-only
shortcut. These tests exercise exactly that real path rather than
monkeypatching a legacy API surface that doesn't currently exist (see the
PHASE A report's RISKS section for this pre-existing gap).
"""
from __future__ import annotations

from modes.streamer import runner


def _enrichment(*, active_speaker_segments, webcam_boxes, warnings=None):
    return {
        "webcam_layout": {"boxes": webcam_boxes},
        "active_speaker_segments": active_speaker_segments,
        "word_timestamps": [{"word": "hello"}, {"word": "chat"}],
        "crop_hints": {"anchor": "center"},
        "warnings": list(warnings or []),
    }


# Three active-speaker segments -> three _fallback_segments() candidates
# with three DISTINCT scores (2.0 base + 0.1 per nearby webcam box, see
# _fallback_segments in runner.py), so "pick top N by score" is a real
# assertion rather than a coincidence of insertion order:
#   start=70  (C): 0 nearby boxes -> score 2.0  (lowest)
#   start=10  (A): 1 nearby box   -> score 2.1
#   start=40  (B): 3 nearby boxes -> score 2.3  (highest)
_ACTIVE_SPEAKER_SEGS = [
    {"start": 10, "end": 30},   # duration 20 -> A, score 2.1
    {"start": 40, "end": 55},   # duration 15 -> B, score 2.3
    {"start": 70, "end": 100},  # duration 30 -> C, score 2.0
]


def _default_enrichment(**_kwargs):
    # frame/30 must land within 5s of the segment's own start to count as
    # "nearby" (see _fallback_segments). One box near start=10, three near
    # start=40, none near start=70.
    boxes = (
        [{"frame": 10 * 30}] +          # 1 box near A (start=10)  -> +0.1
        [{"frame": 40 * 30}] * 3        # 3 boxes near B (start=40) -> +0.3
    )
    return _enrichment(active_speaker_segments=_ACTIVE_SPEAKER_SEGS, webcam_boxes=boxes)


# ── analyze() ────────────────────────────────────────────────────────────

def test_analyze_calls_enrichment_exactly_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: (calls.append(kw) or _default_enrichment()),
    )

    runner.analyze("input.mp4", str(tmp_path), params={})

    assert len(calls) == 1


def test_analyze_returns_all_candidates_unfiltered(monkeypatch, tmp_path):
    """analyze() must NOT pre-select a top-N subset — that stays run()'s job."""
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )

    result = runner.analyze("input.mp4", str(tmp_path), params={"max_clips": 1})

    assert len(result["segments"]) == 3  # all 3 candidates, despite max_clips=1


def test_analyze_normalizes_to_canonical_segment_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )

    result = runner.analyze("input.mp4", str(tmp_path), params={})

    for seg in result["segments"]:
        assert set(seg.keys()) == {"start_sec", "duration_sec", "score", "source"}
        assert isinstance(seg["start_sec"], float)
        assert isinstance(seg["duration_sec"], float)


def test_analyze_falls_back_and_warns_when_legacy_analyzer_unavailable(monkeypatch, tmp_path):
    """Documents the real current behavior: the yuvelirochka import fails
    (see module docstring), so analyze() always falls back today and
    records why via a warning."""
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )

    result = runner.analyze("input.mp4", str(tmp_path), params={})

    assert any("streamer_analyzer_unavailable" in w for w in result["warnings"])
    assert all(seg["source"] == "speaker_fallback" for seg in result["segments"])


# ── compose_one() ────────────────────────────────────────────────────────

def test_compose_one_does_not_call_enrichment(monkeypatch, tmp_path):
    """B: compose_one() must never re-run enrich_video_for_mode()."""
    def _boom(**_kw):
        raise AssertionError("compose_one() must not call enrich_video_for_mode()")
    monkeypatch.setattr("scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode", _boom)

    captured = {}
    monkeypatch.setattr(
        "scripts.shared.crop.smart_crop_adapter.compose_vertical_clip",
        lambda **kw: (captured.update(kw) or kw["output_path"]),
    )

    segment = {"start_sec": 100.0, "duration_sec": 12.5, "score": 1.0, "source": "analyzer"}
    out = str(tmp_path / "clip.mp4")
    result = runner.compose_one("input.mp4", out, segment, crop_hints={"anchor": "center"})

    assert result == out
    assert captured  # compose_vertical_clip was actually called


def test_compose_one_passes_correct_start_and_duration(monkeypatch, tmp_path):
    """C: compose_one() must translate start_sec/duration_sec -> the
    compose_vertical_clip start_time/duration contract, unchanged values."""
    captured = {}
    monkeypatch.setattr(
        "scripts.shared.crop.smart_crop_adapter.compose_vertical_clip",
        lambda **kw: (captured.update(kw) or kw["output_path"]),
    )

    segment = {"start_sec": 4350.0, "duration_sec": 370.0, "score": 1.0, "source": "analyzer"}
    out = str(tmp_path / "clip.mp4")
    runner.compose_one("input.mp4", out, segment, crop_hints={"x": 1})

    assert captured["input_video_path"] == "input.mp4"
    assert captured["output_path"] == out
    assert captured["start_time"] == 4350.0
    assert captured["duration"] == 370.0
    assert captured["crop_hints"] == {"x": 1}


# ── run() — compatibility wrapper (PRESERVE CURRENT PRODUCTION BEHAVIOR) ──

def test_run_calls_analysis_exactly_once(monkeypatch, tmp_path):
    """A (part 1): the old run() entry point must still call the expensive
    analysis path exactly once, not once per selected clip."""
    calls = []
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: (calls.append(kw) or _default_enrichment()),
    )
    monkeypatch.setattr(
        "scripts.shared.crop.smart_crop_adapter.compose_vertical_clip",
        lambda **kw: kw["output_path"],
    )

    runner.run("input.mp4", str(tmp_path), params={"max_clips": 5})

    assert len(calls) == 1


def test_run_creates_up_to_max_clips_selected_by_score(monkeypatch, tmp_path):
    """A (part 2): run() still picks the top `max_clips` candidates by
    score and composes exactly that many, same as before the split."""
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )
    compose_calls = []
    monkeypatch.setattr(
        "scripts.shared.crop.smart_crop_adapter.compose_vertical_clip",
        lambda **kw: (compose_calls.append(kw) or kw["output_path"]),
    )

    result = runner.run("input.mp4", str(tmp_path), params={"max_clips": 2})

    assert len(result["clips"]) == 2
    assert len(compose_calls) == 2
    # Highest two scores (B=2.3, A=2.1) selected, in that order — C (2.0) dropped.
    assert compose_calls[0]["start_time"] == 40.0   # B
    assert compose_calls[1]["start_time"] == 10.0   # A


def test_run_respects_default_max_clips_of_five(monkeypatch, tmp_path):
    """max_clips defaults to 5 when params omits it — unchanged from before
    the split (mode.yaml's own output.max_clips: 5 is untouched in PHASE A)."""
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )
    monkeypatch.setattr(
        "scripts.shared.crop.smart_crop_adapter.compose_vertical_clip",
        lambda **kw: kw["output_path"],
    )

    result = runner.run("input.mp4", str(tmp_path), params={})

    # Only 3 raw candidates exist in this fixture -- fewer than the
    # default cap of 5, so all 3 get composed.
    assert len(result["clips"]) == 3


def test_run_result_shape_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )
    monkeypatch.setattr(
        "scripts.shared.crop.smart_crop_adapter.compose_vertical_clip",
        lambda **kw: kw["output_path"],
    )

    result = runner.run("input.mp4", str(tmp_path), params={"max_clips": 1})

    assert set(result.keys()) == {
        "clips", "mode", "beta", "webcam_boxes_found", "active_speaker_segs", "warnings",
    }
    assert result["mode"] == "streamer"
    assert result["beta"] is True
    assert result["webcam_boxes_found"] == 4   # 1 + 3, see _default_enrichment
    assert result["active_speaker_segs"] == 3


def test_run_appends_warning_on_per_clip_compose_failure(monkeypatch, tmp_path):
    """A per-clip compose failure must still just warn, same as before the
    split — not abort the whole job."""
    monkeypatch.setattr(
        "scripts.shared.enhancers.sonya_enhancer.enrich_video_for_mode",
        lambda **kw: _default_enrichment(),
    )

    def _flaky_compose(**kw):
        if kw["start_time"] == 40.0:  # the highest-scored segment (B)
            raise RuntimeError("ffmpeg exploded")
        return kw["output_path"]
    monkeypatch.setattr("scripts.shared.crop.smart_crop_adapter.compose_vertical_clip", _flaky_compose)

    result = runner.run("input.mp4", str(tmp_path), params={"max_clips": 3})

    assert len(result["clips"]) == 2  # one of the three failed
    assert any("clip_1_failed" in w for w in result["warnings"])
