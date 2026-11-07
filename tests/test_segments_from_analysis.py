"""
Regression test for scripts.prod_job_store.segments_from_analysis()
preserving per-segment title/description/recommended instead of silently
discarding them (see the NO-GPU local E2E harness pass: fake segments
submitted with real content were coming back as generic "Тема N"
placeholders before this fix).

Pure function, no DB access at all -- no real Postgres needed here, same
as tests/test_job_store_claim.py.
"""
from __future__ import annotations

from scripts.prod_job_store import segments_from_analysis


def test_preserves_title_when_present():
    result = segments_from_analysis({
        "segments": [{"start_sec": 10.0, "duration_sec": 5.0, "title": "Первый сильный момент"}],
    })
    assert result[0]["title"] == "Первый сильный момент"


def test_preserves_description_when_present():
    result = segments_from_analysis({
        "segments": [{"start_sec": 10.0, "duration_sec": 5.0, "description": "Реакция стримера"}],
    })
    assert result[0]["description"] == "Реакция стримера"


def test_preserves_recommended_true_when_present():
    result = segments_from_analysis({
        "segments": [{"start_sec": 10.0, "duration_sec": 5.0, "recommended": True}],
    })
    assert result[0]["recommended"] is True


def test_preserves_recommended_false_when_explicitly_present():
    result = segments_from_analysis({
        "segments": [{"start_sec": 10.0, "duration_sec": 5.0, "recommended": False}],
    })
    assert result[0]["recommended"] is False


def test_falls_back_to_generic_title_when_absent():
    """Today's real analyze() output (modes/streamer/runner.py's
    _fallback_segments()) never includes a title -- this fallback is what
    actually runs in production right now."""
    result = segments_from_analysis({
        "segments": [
            {"start_sec": 10.0, "duration_sec": 5.0},
            {"start_sec": 20.0, "duration_sec": 5.0},
        ],
    })
    assert result[0]["title"] == "Тема 1"
    assert result[1]["title"] == "Тема 2"


def test_falls_back_to_none_description_when_absent():
    result = segments_from_analysis({"segments": [{"start_sec": 10.0, "duration_sec": 5.0}]})
    assert result[0]["description"] is None


def test_falls_back_to_recommended_false_when_absent():
    result = segments_from_analysis({"segments": [{"start_sec": 10.0, "duration_sec": 5.0}]})
    assert result[0]["recommended"] is False


def test_explicit_titles_param_still_overrides_segment_title():
    """Backward compatibility: the older, narrower `titles` positional
    override still wins over a segment's own "title" key, same as before
    this fix -- nothing currently calls it this way, but the precedence
    is a deliberate, documented part of the contract."""
    result = segments_from_analysis(
        {"segments": [{"start_sec": 10.0, "duration_sec": 5.0, "title": "From segment"}]},
        titles=["From titles param"],
    )
    assert result[0]["title"] == "From titles param"


def test_score_and_crop_hints_and_metadata_still_carried_through():
    """Untouched by this fix -- guards against a future edit accidentally
    dropping these while reworking the title/description/recommended
    pass-through."""
    result = segments_from_analysis({
        "segments": [{"start_sec": 10.0, "duration_sec": 5.0, "score": 0.87, "source": "fallback"}],
        "crop_hints": {"box": [0, 0, 1, 1]},
    })
    assert result[0]["score"] == 0.87
    assert result[0]["crop_hints"] == {"box": [0, 0, 1, 1]}
    assert result[0]["metadata"] == {"source": "fallback"}
