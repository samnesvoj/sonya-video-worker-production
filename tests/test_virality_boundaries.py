"""Cut (virality) → boundary engine → renderer gets word-aligned intervals → metadata saved."""
from __future__ import annotations

import json

from scripts.shared.crop import smart_crop_adapter
from scripts.shared.enhancers import sonya_enhancer


def _words():
    # speech every 0.5 s from 0.2 s to ~45 s; sentence end every 8th word
    out, t, i = [], 0.2, 0
    while t < 45:
        out.append({"word": f"w{i}" + ("." if i % 8 == 7 else ""), "start": round(t, 3), "end": round(t + 0.35, 3)})
        t += 0.5
        i += 1
    return out


def test_virality_renders_snapped_interval_and_records_it(monkeypatch, tmp_path):
    words = _words()
    monkeypatch.setattr(sonya_enhancer, "enrich_video_for_mode",
                        lambda **kw: {"warnings": [], "word_timestamps": words, "crop_hints": {}})
    rendered = []
    monkeypatch.setattr(smart_crop_adapter, "compose_vertical_clip",
                        lambda **kw: rendered.append((kw["start_time"], kw["duration"])))
    from modes.virality import runner
    runner.run("in.mp4", str(tmp_path), {"max_clips": 1})

    # with words the semantic interval is an ASR candidate (complete sentences),
    # never the legacy 0–30 s; boundaries-v1 then adds pre/post-roll on real words
    start, dur = rendered[0]
    end = start + dur
    word_spans = [(w["start"], w["end"]) for w in words]
    assert not any(s + 1e-6 < t < e - 1e-6 for s, e in word_spans for t in (start, end))
    data = json.loads((tmp_path / "selected_segments.json").read_text(encoding="utf-8"))
    seg = data["segments"][0]
    assert seg["candidate_id"].startswith("c")
    assert seg["semantic_start"] in {w["start"] for w in words}
    assert seg["semantic_end"] in {w["end"] for w in words}
    assert (seg["semantic_start"], seg["semantic_end"]) != (0.0, 30.0)
    assert seg["boundary_status"] == "snapped" and seg["boundary_version"] == "boundaries-v1"
    assert seg["start"] == seg["render_start"] == round(start, 3)
    assert abs(seg["end"] - end) < 1e-3
    assert "boundary_params" in seg


def test_virality_without_words_keeps_interval(monkeypatch, tmp_path):
    monkeypatch.setattr(sonya_enhancer, "enrich_video_for_mode",
                        lambda **kw: {"warnings": [], "word_timestamps": [], "crop_hints": {}})
    rendered = []
    monkeypatch.setattr(smart_crop_adapter, "compose_vertical_clip",
                        lambda **kw: rendered.append((kw["start_time"], kw["duration"])))
    from modes.virality import runner
    runner.run("in.mp4", str(tmp_path), {"max_clips": 1})
    assert rendered == [(0.0, 30.0)]
    seg = json.loads((tmp_path / "selected_segments.json").read_text())["segments"][0]
    assert seg["boundary_status"] == "no_word_timestamps"
    assert runner.run("in.mp4", str(tmp_path), {"max_clips": 1})["selection"]["selection_status"] == \
        "no_asr_legacy_fallback"
