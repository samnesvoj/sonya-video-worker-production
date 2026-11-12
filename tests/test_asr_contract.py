"""The ASR interface contract (docs/ASR_INTERFACE.md): validator, Cut-compatibility metric, offline guarantee, and a compliance test that the
Mac mini backend (scripts/shared/asr) must pass once it lands - skipped until then (needs GIGAAM_WEIGHTS_DIR + GIGAAM_TEST_AUDIO)."""
import os
import socket
from pathlib import Path

import pytest

from scripts.shared import asr_contract as ac

ROOT = Path(__file__).resolve().parents[1]


def good(words=None, **over):
    words = words if words is not None else [{"word": " привет", "start": 0.40, "end": 0.80}, {"word": " мир", "start": 0.84, "end": 1.20}]
    r = {"text": "привет мир", "language": "ru", "backend": "gigaam", "model": "gigaam-v3_e2e_rnnt-ts3",
         "segments": [{"start": 0.0, "end": 1.3, "text": "привет мир", "words": words}], "words": words}
    r.update(over)
    return r


def test_a_conforming_result_has_no_problems():
    assert ac.validate_result(good(), audio_duration=1.5) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.update(words=[{"word": "а", "start": 1.0, "end": 1.0}]), "start < end"),
    (lambda r: r.update(words=[{"word": "а", "start": -0.1, "end": 0.2}]), "start < end"),
    (lambda r: r.update(words=[{"word": " ", "start": 0.0, "end": 0.2}]), "empty"),
    (lambda r: r.update(words=[{"word": "б", "start": 2.0, "end": 2.2}, {"word": "а", "start": 1.0, "end": 1.2}]), "not sorted"),
    (lambda r: r.update(words=[{"word": "а", "start": 0.0, "end": 1.0}, {"word": "б", "start": 0.5, "end": 1.2}]), "overlaps"),
    (lambda r: r.update(words=[{"word": "а", "start": 0.0, "end": 99.0}]), "beyond the audio"),
    (lambda r: r.update(words=[]), "words is empty"),
    (lambda r: r.update(model="gigaam-v3_e2e_rnnt-ts2"), "-150 ms"),
    (lambda r: r.update(model="gigaam-v3_e2e_rnnt"), "model must look like"),
    (lambda r: r.update(model="gigaam-v3_rnnt-ts3"), "model must look like"),
    (lambda r: r.update(backend="whisperx"), "backend"),
    (lambda r: r.pop("segments"), "segments"),
    (lambda r: r.pop("model"), "model"),
    (lambda r: r.update(language=""), "language"),
])
def test_each_violation_is_reported(mutate, needle):
    r = good()
    mutate(r)
    assert any(needle in p for p in ac.validate_result(r, audio_duration=1.5)), ac.validate_result(r, 1.5)


def test_the_fallback_backend_needs_no_gigaam_specific_model_tag():
    assert ac.validate_result(good(backend="faster_whisper_cuda", model="large-v3"), 1.5) == []


def test_backend_selection_defaults_to_gigaam_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("SONYA_ASR_BACKEND", raising=False)
    assert ac.backend_name() == "gigaam" and ac.backend_name({"SONYA_ASR_BACKEND": "faster_whisper_cuda"}) == "faster_whisper_cuda"
    with pytest.raises(ValueError):
        ac.backend_name({"SONYA_ASR_BACKEND": "gigaam-v2"})


def test_expected_files_match_the_manifest_and_mode_yaml_keys():
    import json
    man = {m["key"]: m["local_path"] for m in json.loads((ROOT / "release/models.required.json").read_text())["models"]}
    assert {k: man[k] for k in ac.GIGAAM_FILES} == ac.GIGAAM_FILES
    assert [p.name for p in ac.expected_model_files(Path("/m"))] == ["v3_e2e_rnnt.ckpt", "v3_e2e_rnnt_tokenizer.model"]
    assert ac.expected_model_files(Path("/m"), "faster_whisper_cuda") == []


# ---- Cut compatibility: the metric that decides whether GigaAM words survive Cut's own normalisation
def _grid_words(n=300, frame=ac.FRAME_SEC):
    """RNNT-like words on a 40 ms grid, each at least one frame long, none overlapping."""
    out, f = [], 10
    for i in range(n):
        length = 1 + (i % 6)
        out.append({"word": f"слово{i}", "start": round(f * frame, 3), "end": round((f + length) * frame, 3)})
        f += length + (i % 2)
    return out


def test_well_formed_words_survive_cut_completely_and_make_candidates():
    rep = ac.cut_compat(_grid_words())
    assert rep["ok"] and rep["dropped"] == 0 and rep["kept"] == 300 and rep["sentences"] >= 1


def test_same_frame_words_are_silently_dropped_by_cut_and_the_metric_catches_it():
    """Upstream gigaam gives every word end=(last_frame+1)*shift, so two short words emitted in ONE frame start together: Cut drops the 2nd."""
    words = _grid_words(100)
    f = 5000 * ac.FRAME_SEC
    words += [{"word": "и", "start": round(f, 3), "end": round(f + ac.FRAME_SEC, 3)}, {"word": "я", "start": round(f, 3), "end": round(f + ac.FRAME_SEC, 3)}]
    rep = ac.cut_compat(words)
    assert rep["dropped"] >= 1                                                    # a real word was lost
    rep2 = ac.cut_compat(words, max_dropped_ratio=0.0)
    assert rep2["ok"] is False                                                    # and a strict threshold fails the backend


def test_zero_length_words_are_counted_as_lost():
    words = _grid_words(50) + [{"word": "ой", "start": 90.0, "end": 90.0}]
    assert ac.cut_compat(words)["dropped"] == 1


# ---- offline guarantee
@pytest.fixture
def no_network(monkeypatch):
    def deny(*a, **k):
        raise AssertionError("network access attempted while ASR must be offline")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


def test_the_no_network_fixture_really_blocks_connections(no_network):
    with pytest.raises(AssertionError, match="offline"):
        socket.create_connection(("example.com", 80), timeout=1)


# ---- compliance test for the Mac mini backend (runs where the weights exist)
def test_backend_compliance_offline_with_real_weights(no_network):
    asr = pytest.importorskip("scripts.shared.asr", reason="Mac mini ASR backend not merged yet")
    weights, audio = os.environ.get("GIGAAM_WEIGHTS_DIR"), os.environ.get("GIGAAM_TEST_AUDIO")
    if not (weights and audio and Path(weights).is_dir() and Path(audio).is_file()):
        pytest.skip("set GIGAAM_WEIGHTS_DIR (the models root containing gigaam/) and GIGAAM_TEST_AUDIO")
    import subprocess, json
    dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", audio], capture_output=True, text=True).stdout)["format"]["duration"])
    os.environ.update(SONYA_ASR_BACKEND="gigaam", SONYA_ASR_OFFLINE="1", MODELS_LOCAL_DIR=weights)
    res = asr.transcribe(audio, language="ru")
    assert ac.validate_result(res, dur) == []
    rep = ac.cut_compat(res["words"])
    assert rep["ok"], rep


# ---- the reference post-processing of the Mac mini backend, as a TEST ORACLE (copied from desktop/sonya_desktop/asr/backends.py, owner code).
# It is not the production implementation (that is the Mac mini branch); it proves the contract is satisfiable and that the post-processing is
# exactly what makes upstream RNNT words survive Cut.
def _reference_fix_rnnt_word_times(words, lead=ac.RNNT_START_SHIFT_SEC, min_word=ac.MIN_WORD_SEC):
    prev_end = 0.0
    for i, w in enumerate(words):
        s, e = float(w["start"]), float(w["end"])
        s = max(prev_end, s - lead)
        nxt = float(words[i + 1]["start"]) - lead if i + 1 < len(words) else e + min_word
        if e - s < min_word:
            e = max(e, min(s + min_word, nxt - 0.01))
        w["start"], w["end"] = round(s, 3), round(max(e, s), 3)
        prev_end = w["end"]


def _raw_upstream_like_words():
    """What upstream gigaam frames_to_words yields: start=f*0.04, end=(f_last+1)*0.04, several short words may share one frame."""
    words, f = [], 20
    for i in range(400):
        n = 1 + (i % 5)
        words.append({"word": f" слово{i}", "start": round(f * ac.FRAME_SEC, 3), "end": round((f + n) * ac.FRAME_SEC, 3)})
        if i % 17 == 0:                                                  # two tiny words emitted in the same frame
            words.append({"word": " и", "start": words[-1]["start"], "end": words[-1]["end"]})
        f += n
    return words


def test_reference_postprocessing_saves_normal_words_but_a_same_frame_word_still_dies_in_cut():
    """FINDING for the Mac mini backend: fix_rnnt_word_times rescues ordinary short words, but a word emitted in the SAME 40 ms frame as the
    previous one is clamped to start=prev_end and ends up zero-length -> Cut's normalize_words drops it silently. Real-data loss rate must be
    measured (tools/ci_gigaam_real.py, threshold 1 %); a robust backend should spread same-frame words over the frame instead."""
    raw = _raw_upstream_like_words()
    same_frame = sum(1 for a, b in zip(raw, raw[1:]) if a["start"] == b["start"])
    assert same_frame > 0 and ac.cut_compat(raw)["dropped"] >= same_frame
    fixed = [dict(w) for w in raw]
    _reference_fix_rnnt_word_times(fixed)
    rep = ac.cut_compat(fixed)
    assert rep["dropped"] <= same_frame                                    # nothing but the pathological same-frame words is lost
    no_dups = [dict(w) for w in raw if w["word"] != " и"]
    _reference_fix_rnnt_word_times(no_dups)
    clean = ac.cut_compat(no_dups)
    assert clean["dropped"] == 0 and clean["ok"]                           # ordinary RNNT words all survive after post-processing
    res = good(words=no_dups, text="x", model="gigaam-v3_e2e_rnnt-ts3")
    res["segments"][0]["words"] = no_dups
    assert ac.validate_result(res, no_dups[-1]["end"] + 1) == []
    assert all(f["start"] <= r["start"] + 1e-9 for f, r in zip(no_dups, [w for w in raw if w["word"] != " и"]))     # starts only move EARLIER
    assert all(b["start"] >= a["end"] - 1e-9 for a, b in zip(no_dups, no_dups[1:]))                                  # and never overlap
