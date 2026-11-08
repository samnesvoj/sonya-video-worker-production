"""Worker ASR backend (scripts/shared/asr): GigaAM primary, offline, -150 ms start correction, Cut-safe words.
Pure-Python tests run everywhere; the real-weights test runs where GIGAAM_WEIGHTS_DIR + GIGAAM_TEST_AUDIO exist (Mac mini / CI with S3)."""
import json
import os
import socket
from pathlib import Path

import pytest

from scripts.shared import asr_contract as ac
from scripts.shared.asr import gigaam_backend as g


def test_constants_match_the_contract_and_the_reference():
    assert g.START_LEAD == ac.RNNT_START_SHIFT_SEC == 0.15
    assert g.MIN_WORD == ac.MIN_WORD_SEC
    assert set(ac.GIGAAM_FILES.values()) == {f"gigaam/{f}" for f in g.FILES}
    assert g.MODEL_NAME == "gigaam-v3_e2e_rnnt-ts4"
    import re
    assert int(re.match(ac.GIGAAM_MODEL_RE, g.MODEL_NAME).group(1)) >= ac.MIN_POSTPROC_TAG


def test_starts_move_150_ms_earlier_but_never_before_the_previous_word():
    w = [{"word": " а", "start": 1.00, "end": 1.40}, {"word": " б", "start": 1.45, "end": 1.90}, {"word": " в", "start": 3.00, "end": 3.50}]
    g.fix_rnnt_word_times(w)
    assert w[0]["start"] == 0.85
    assert w[1]["start"] == 1.40            # 1.45 - 0.15 = 1.30 would overlap «а» -> clamped to its end
    assert w[2]["start"] == 2.85


def test_short_words_get_a_minimum_length():
    w = [{"word": " и", "start": 2.00, "end": 2.04}, {"word": " всё", "start": 2.60, "end": 3.0}]
    g.fix_rnnt_word_times(w)
    assert w[0]["end"] - w[0]["start"] >= g.MIN_WORD - 1e-9


def test_punctuation_tokens_are_glued():
    w = [{"word": " прикол.", "start": 1, "end": 2}, {"word": " —", "start": 2, "end": 2.1}, {"word": " да", "start": 2.2, "end": 2.5},
         {"word": " »", "start": 2.5, "end": 2.6}]
    out = g.merge_punct_tokens(w)
    assert [x["word"] for x in out] == [" прикол.", " — да»"]


def test_same_frame_words_survive_cut_after_ts4_split():
    raw = [{"word": " что", "start": 10.0, "end": 10.04}, {"word": " я", "start": 10.0, "end": 10.04}, {"word": " знаю", "start": 10.16, "end": 10.7}]
    w = [dict(x) for x in raw]
    g.fix_rnnt_word_times(w)
    assert any(x["end"] <= x["start"] for x in w)          # the reference ts3 post-processing leaves a zero-length word ...
    assert g.split_zero_length_words(w) == 1               # ... ts4 gives it half of the previous word
    assert all(x["end"] > x["start"] for x in w)
    assert all(b["start"] >= a["end"] - 1e-9 for a, b in zip(w, w[1:]))
    assert ac.cut_compat(w)["dropped"] == 0


def test_missing_weights_raise_and_never_download(tmp_path, monkeypatch):
    def deny(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    (tmp_path / "gigaam").mkdir()
    (tmp_path / "gigaam" / "v3_e2e_rnnt.ckpt").write_bytes(b"x")       # tokenizer missing
    with pytest.raises(g.GigaAMUnavailable, match="missing"):
        g.load_model(tmp_path)


def test_dispatch_default_is_gigaam_and_fallback_is_explicit(monkeypatch, tmp_path):
    from scripts.shared import asr
    calls = []
    monkeypatch.setattr(g, "transcribe", lambda p, progress=None: (_ for _ in ()).throw(g.GigaAMUnavailable("no weights")))
    monkeypatch.setattr(asr, "_whisper", lambda p, lang, b: calls.append((lang, b)) or {"words": [], "backend": b})
    monkeypatch.delenv("SONYA_ASR_BACKEND", raising=False)
    monkeypatch.setenv("SONYA_ASR_FALLBACK", "0")
    with pytest.raises(g.GigaAMUnavailable):
        asr.transcribe("x.mp4")                             # default backend is GigaAM; no silent downgrade
    monkeypatch.setenv("SONYA_ASR_FALLBACK", "1")
    res = asr.transcribe("x.mp4", language="auto")
    assert res["fallback_from"] == "gigaam" and calls[-1][0] == "ru"
    asr.transcribe("x.mp4", language="en")                  # GigaAM is Russian-only -> whisper
    assert calls[-1][0] == "en"
    monkeypatch.setenv("SONYA_ASR_BACKEND", "faster_whisper_cpu")
    asr.transcribe("x.mp4")
    assert calls[-1] == ("ru", "faster_whisper_cpu")


def test_beep_runs_asr_itself_and_fails_loudly_when_it_cannot(tmp_path, monkeypatch):
    from modes.beep import runner
    from scripts.shared import asr
    monkeypatch.setattr(asr, "transcribe", lambda p, language=None: (_ for _ in ()).throw(g.GigaAMUnavailable("no weights")))
    with pytest.raises(RuntimeError, match="BEEP_FAILED: ASR failed"):
        runner.run("in.mp4", str(tmp_path), {})
    monkeypatch.setattr(asr, "transcribe", lambda p, language=None: {"backend": "gigaam", "model": g.MODEL_NAME, "language": "ru", "words": []})
    with pytest.raises(RuntimeError, match="ASR returned no words"):
        runner.run("in.mp4", str(tmp_path), {})
    assert json.loads((tmp_path / "asr_words.json").read_text())["backend"] == "gigaam"


def test_vendored_upstream_is_the_pinned_commit():
    v = Path(g.__file__).resolve().parents[2] / "third_party" / "gigaam"
    assert (v / "LICENSE").read_text().startswith("MIT License")
    assert "7447938d791c4f3e643386ee22c33777004293a5" in (v / "VENDORED.md").read_text()
    import hashlib
    rows = [l.split() for l in (v / "VENDORED.md").read_text().splitlines() if len(l.split()) == 2 and len(l.split()[0]) == 64]
    assert len(rows) == 12
    for sha, name in rows:
        assert hashlib.sha256((v / name).read_bytes()).hexdigest() == sha, name


@pytest.mark.skipif(not (os.environ.get("GIGAAM_WEIGHTS_DIR") and os.environ.get("GIGAAM_REFERENCE_JSON") and os.environ.get("GIGAAM_TEST_AUDIO")),
                    reason="needs GIGAAM_WEIGHTS_DIR, GIGAAM_TEST_AUDIO and the Mac mini reference transcript GIGAAM_REFERENCE_JSON")
def test_same_words_and_times_as_the_mac_mini_reference(monkeypatch):
    def deny(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    res = g.transcribe(os.environ["GIGAAM_TEST_AUDIO"], models_root=Path(os.environ["GIGAAM_WEIGHTS_DIR"]))
    ref = json.loads(Path(os.environ["GIGAAM_REFERENCE_JSON"]).read_text())["words"]
    assert [w["word"] for w in res["words"]] == [w["word"] for w in ref]
    split = {j for i, w in enumerate(ref) if w["end"] <= w["start"] for j in (i - 1, i)}     # ts4 changes only these (ts3 left them zero-length)
    worst = max(max(abs(a["start"] - b["start"]), abs(a["end"] - b["end"])) for i, (a, b) in enumerate(zip(res["words"], ref)) if i not in split)
    assert worst <= float(os.environ.get("GIGAAM_TIME_TOLERANCE_SEC", "0.04"))      # one RNNT frame (fp16 on other GPUs)
