"""Deterministic boundary engine (scripts/shared/boundaries.py) — synthetic words."""
from __future__ import annotations

import random

import pytest

from scripts.shared.boundaries import (BoundaryConfig, Word, cuts_inside_word, group_sentences,
                                       normalize_words, refine_interval, refine_segments,
                                       resolve_overlaps)


def W(text, s, e):
    return {"word": text, "start": s, "end": e}


# 0.0 "Привет" 0.4 | 0.5 "всем." 0.9 || pause 1.1 || 2.0 "Сегодня" 2.5 | 2.6 "расскажу" 3.2 |
# 3.3 "про" 3.5 | 3.6 "монтаж," 4.1 | 4.2 "это" 4.4 | 4.5 "важно." 5.0 | 5.2 "Дальше" 5.6 | 5.7 "пример" 6.2
SPEECH = [W("Привет", 0.0, 0.4), W("всем.", 0.5, 0.9), W("Сегодня", 2.0, 2.5), W("расскажу", 2.6, 3.2),
          W("про", 3.3, 3.5), W("монтаж,", 3.6, 4.1), W("это", 4.2, 4.4), W("важно.", 4.5, 5.0),
          W("Дальше", 5.2, 5.6), W("пример", 5.7, 6.2)]
CFG = BoundaryConfig(pre_roll=0.15, post_roll=0.3, pause_sec=0.7, max_extend_sec=3.0,
                     min_duration=0.5, max_duration=90, overlap_tolerance=0.2)


def prep(raw=SPEECH, cfg=CFG):
    words, _ = normalize_words(raw)
    return words, group_sentences(words, cfg.pause_sec)


def test_normalize_sorts_drops_malformed_and_deoverlaps():
    raw = [W("b", 1.0, 1.5), W("a", 0.0, 0.5), {"word": "x", "start": None, "end": 1},
           W("", 2, 3), W("neg", -1, 0.2), W("zero", 3, 3), W("inside", 1.1, 1.4), W("over", 1.3, 1.9),
           {"text": "alt", "start": "2.0", "end": "2.2"}, W("nan", float("nan"), 1)]
    words, st = normalize_words(raw)
    assert [w.text for w in words] == ["a", "b", "over", "alt"]
    assert words[2].start == 1.5  # trimmed to previous end
    assert st["dropped_malformed"] == 5 and st["dropped_overlap"] == 1 and st["trimmed_overlap"] == 1
    assert all(words[i].end <= words[i + 1].start for i in range(len(words) - 1))


def test_sentence_grouping_by_punctuation_and_pause():
    words, groups = prep()
    texts = [" ".join(w.text for w in words[a:b + 1]) for a, b in groups]
    assert texts == ["Привет всем.", "Сегодня расскажу про монтаж, это важно.", "Дальше пример"]
    # pause-only boundary (no punctuation)
    w2, g2 = prep([W("раз", 0, 0.3), W("два", 1.5, 1.8)])
    assert g2 == [(0, 0), (1, 1)]


def test_edges_inside_words_include_whole_word_and_extend_to_sentence():
    words, groups = prep()
    r = refine_interval(2.8, 4.0, words, groups, CFG)  # inside "расскажу" … inside "монтаж,"
    assert r.status == "snapped"
    assert "start_was_inside_word" in r.reasons and "end_was_inside_word" in r.reasons
    assert "start_extended_to_sentence" in r.reasons and "end_extended_to_sentence" in r.reasons
    assert words[r.first_word].text == "Сегодня" and words[r.last_word].text == "важно."
    assert not cuts_inside_word(r.start, r.end, words)


def test_pre_post_roll_and_neighbour_limits():
    words, groups = prep()
    cfg = BoundaryConfig(pre_roll=0.15, post_roll=0.3, max_extend_sec=0, min_duration=0.1, pause_sec=0.7)
    r = refine_interval(2.0, 2.5, words, groups, cfg)  # exactly "Сегодня"
    assert r.start == pytest.approx(1.85) and r.end == pytest.approx(2.6)  # post-roll stops at next word
    assert "post_roll_limited_by_next_word" in r.reasons
    r0 = refine_interval(0.0, 0.4, words, groups, cfg)
    assert r0.start == 0.0  # clamped at media start


def test_sentence_extension_respects_max_extend():
    words, groups = prep()
    cfg = BoundaryConfig(max_extend_sec=0.5, min_duration=0.1)
    r = refine_interval(3.3, 3.5, words, groups, cfg)  # "про": sentence start is 1.3 s away
    assert words[r.first_word].text == "про" and "start_extended_to_sentence" not in r.reasons


def test_min_duration_grows_by_whole_words():
    words, groups = prep()
    cfg = BoundaryConfig(min_duration=3.0, max_extend_sec=0)
    r = refine_interval(2.0, 2.5, words, groups, cfg)
    assert r.duration >= 3.0 and "extended_to_min_duration" in r.reasons
    assert not cuts_inside_word(r.start, r.end, words)


def test_max_duration_trims_at_sentence_end():
    words, groups = prep()
    cfg = BoundaryConfig(max_duration=4.0, min_duration=0.5)
    r = refine_interval(0.0, 6.2, words, groups, cfg)
    assert r.duration <= 4.0 + 1e-6
    assert "trimmed_to_max_duration_at_sentence" in r.reasons
    assert words[r.last_word].text.endswith(".")


def test_no_words_keeps_semantic_interval():
    r = refine_segments([{"start": 10, "duration": 30}], [], CFG)[0]
    assert (r["start"], r["duration"]) == (10, 30) and r["boundary_status"] == "no_word_timestamps"


def test_interval_without_speech_is_kept():
    words, groups = prep()
    r = refine_interval(1.0, 1.8, words, groups, CFG)  # inside the pause
    assert r.status == "no_speech_in_interval" and (r.start, r.end) == (1.0, 1.8)


def test_overlap_lower_ranked_is_trimmed_or_dropped():
    words, groups = prep()
    cfg = BoundaryConfig(max_extend_sec=0, min_duration=0.5, overlap_tolerance=0.1)
    a = refine_interval(2.0, 4.1, words, groups, cfg)        # Сегодня … монтаж,
    b = refine_interval(3.3, 6.2, words, groups, cfg)        # про … пример (overlaps a)
    c = refine_interval(2.6, 3.5, words, groups, cfg)        # fully inside a
    out = resolve_overlaps([a, b, c], words, cfg)
    assert out[0].status == "snapped"
    assert out[1].status == "snapped" and "trimmed_for_overlap" in out[1].reasons
    assert out[1].start >= out[0].end - cfg.overlap_tolerance
    assert out[2].status == "dropped_overlap"


def test_deterministic_tie_break_and_input_order_independence():
    shuffled = SPEECH[:]
    random.Random(7).shuffle(shuffled)
    a = refine_segments([{"start": 2.8, "end": 4.0}, {"start": 3.0, "end": 6.0}], SPEECH, CFG)
    b = refine_segments([{"start": 2.8, "end": 4.0}, {"start": 3.0, "end": 6.0}], shuffled, CFG)
    assert a == b
    # equal-length free runs → earliest wins
    words, _ = normalize_words([W("a", 0, 1), W("b", 1, 2), W("c", 2, 3), W("d", 3, 4), W("e", 4, 5)])
    from scripts.shared.boundaries import Refined
    hi = Refined(2, 3, 2.0, 3.0, "snapped", [], 2, 2)
    lo = Refined(0, 5, 0.0, 5.0, "snapped", [], 0, 4)
    cfg = BoundaryConfig(pre_roll=0, post_roll=0, min_duration=1, overlap_tolerance=0)
    out = resolve_overlaps([hi, lo], words, cfg)
    assert (out[1].start, out[1].end) == (0.0, 2.0)


@pytest.mark.parametrize("seed", range(30))
def test_property_never_cuts_inside_a_word(seed):
    rng = random.Random(seed)
    t, raw = 0.0, []
    for i in range(200):
        t += rng.uniform(0.02, 1.2)
        d = rng.uniform(0.1, 0.8)
        raw.append(W(f"w{i}" + ("." if rng.random() < 0.15 else ""), round(t, 3), round(t + d, 3)))
        t += d
    words, _ = normalize_words(raw)
    cfg = BoundaryConfig(min_duration=rng.uniform(1, 10), max_duration=rng.uniform(15, 60))
    segs = [{"start": rng.uniform(0, t), "duration": rng.uniform(1, 40)} for _ in range(5)]
    for seg in refine_segments(segs, raw, cfg):
        if seg["boundary_status"] == "snapped":
            assert not cuts_inside_word(seg["start"], seg["start"] + seg["duration"], words), seg
            assert seg["duration"] <= cfg.max_duration + 1e-3 or "first_word_longer_than_max_duration" in seg["boundary_reasons"]


def test_refine_segments_metadata_contract():
    seg = refine_segments([{"start": 2.8, "duration": 1.2, "score": 3.0, "source": "x"}], SPEECH, CFG)[0]
    for k in ("semantic_start", "semantic_end", "render_start", "render_end", "boundary_status",
              "boundary_version", "boundary_reasons", "boundary_params"):
        assert k in seg
    assert seg["semantic_start"] == 2.8 and seg["semantic_end"] == 4.0
    assert seg["score"] == 3.0 and seg["start"] == seg["render_start"]
