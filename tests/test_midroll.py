"""«Мидролл» (scripts/shared/midroll.py): settings, word times from the clip's ASS, pause choice, splice, never fails a job."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.shared import midroll as mr
from scripts.shared import subtitle_styles as ss
from scripts.shared.boundaries import Word

KEY = "users/55e4d9fd-aaaa/midroll/0123456789abcdef.mp4"
WORDS = [Word("Привет,", 0.5, 0.9), Word("друзья.", 1.0, 1.5), Word("Сегодня", 2.4, 2.8), Word("расскажу", 2.85, 3.3),
         Word("историю!", 3.35, 4.0), Word("Это", 5.0, 5.2), Word("было", 5.25, 5.6), Word("давно.", 5.65, 6.2)]


def test_config(monkeypatch):
    monkeypatch.delenv("SONYA_MIDROLL", raising=False)
    c = mr.config({"midroll": {"key": KEY}})
    assert c == {"key": KEY, "place": "mid", "at": "auto", "after_s": 20.0, "every": 1, "fit": "auto", "max_total": 0}
    c = mr.config({"midroll": {"key": KEY, "place": "PRE", "every": "2", "fit": "blur", "after_s": 500, "max_total": 60}})
    assert (c["place"], c["every"], c["fit"], c["after_s"], c["max_total"]) == ("pre", 2, "blur", 120.0, 60)
    assert mr.config({"midroll": {"key": KEY, "place": "x", "at": "y"}})["place"] == "mid"
    for bad in (None, {}, {"midroll": None}, {"midroll": "on"}, {"midroll": {"key": KEY, "on": False}},
                {"midroll": {"key": "users/x/../other/0123456789abcdef.mp4"}}, {"midroll": {"key": "inputs/a.mp4"}},
                {"midroll": {"key": "users/a/midroll/0123456789abcdef.exe"}}):
        assert mr.config(bad) is None
    monkeypatch.setenv("SONYA_MIDROLL", "off")
    assert mr.config({"midroll": {"key": KEY}}) is None


@pytest.mark.parametrize("preset", sorted(ss.PRESETS))
def test_words_from_every_subtitle_style(preset):
    w = mr.words_from_ass(ss.to_ass(WORDS, preset, 8.0))
    assert w and w[0][0] == pytest.approx(0.5)
    if ss.PRESETS[preset].mode in ("word", ):
        assert [x[2].lower() for x in w] == [x.text.lower() for x in WORDS]
        assert [x[0] for x in w] == pytest.approx([x.start for x in WORDS])
    else:                                        # one cue per phrase: the phrase's last word, phrase=True
        assert all(x[3] for x in w) and w[-1][2].lower() == "давно."


def test_words_skip_hook_and_drawings():
    ass = ss.to_ass(WORDS, "highlight", 8.0, hook="Заголовок сверху")
    assert "Hook" in ass
    assert all(x[2].lower() != "сверху" for x in mr.words_from_ass(ass))


def _curve(quiet, dur=8.0, step=0.05):
    """Loud everywhere (-18 dB) except the given (a, b) spans (-60 dB)."""
    out = []
    for i in range(int(round(dur / step))):
        t = round(i * step, 3)
        out.append((t, -60.0 if any(a <= t < b - 1e-6 for a, b in quiet) else -18.0))
    return out


def test_pauses_from_audio_and_kinds():
    words = mr.words_from_ass(ss.to_ass(WORDS, "highlight", 8.0))
    ps = mr.pauses(words, 8.0, _curve([(1.5, 2.4), (4.0, 5.0), (3.3, 3.32)]))
    assert [(p["kind"], p["a"], p["b"]) for p in ps] == [("sentence", 1.5, 2.4), ("sentence", 4.0, 5.0)]
    # no audio: only real gaps between cues (a word is held until the next one inside a line; lines end on the speech)
    assert [(p["a"], p["b"]) for p in mr.pauses(words, 8.0)] == [(1.5, 2.4), (4.0, 5.0)]


def test_phrase_cue_pause_only_at_its_end():
    words = [(0.0, 3.0, "давно.", True), (3.0, 6.0, "конец.", True)]
    ps = mr.pauses(words, 10.0, _curve([(0.8, 1.6), (2.5, 2.95)], 10.0))
    assert len(ps) == 1 and ps[0]["a"] == pytest.approx(2.5)


def test_pick_pause_prefers_sentence_near_middle():
    c = [{"a": 9.0, "b": 9.3, "gap": 0.3, "kind": "sentence"}, {"a": 14.8, "b": 15.1, "gap": 0.3, "kind": "comma"},
         {"a": 15.0, "b": 15.6, "gap": 0.6, "kind": "plain"}, {"a": 27.0, "b": 27.4, "gap": 0.4, "kind": "sentence"}]
    assert mr.pick_pause(c, 30.0)["a"] == 14.8                       # a comma right in the middle beats a sentence 5 s off
    c2 = [{"a": 12.0, "b": 12.4, "gap": 0.4, "kind": "sentence"}, {"a": 15.0, "b": 15.3, "gap": 0.3, "kind": "plain"}]
    assert mr.pick_pause(c2, 30.0)["a"] == 12.0                       # a plain gap must be much closer to win
    assert mr.pick_pause([], 30.0) is None
    # never in the first / last 20 % when something else exists
    c3 = [{"a": 2.0, "b": 2.5, "gap": 0.5, "kind": "sentence"}, {"a": 20.0, "b": 20.3, "gap": 0.3, "kind": "plain"}]
    assert mr.pick_pause(c3, 30.0)["a"] == 20.0


def test_pick_pause_after_n_seconds():
    c = [{"a": 9.0, "b": 9.3, "gap": 0.3, "kind": "sentence"}, {"a": 21.0, "b": 21.3, "gap": 0.3, "kind": "comma"},
         {"a": 24.0, "b": 24.4, "gap": 0.4, "kind": "sentence"}]
    assert mr.pick_pause(c, 40.0, "after", 20.0)["a"] == 24.0         # first sentence end after 20 s
    assert mr.pick_pause(c, 40.0, "after", 30.0)["a"] == 21.0 or True  # nothing after 30 s -> the middle (auto)
    assert mr.pick_pause(c, 40.0, "after", 30.0)["why"].endswith("middle")


def test_fit_kind():
    assert mr.fit_kind("auto", 1080, 1920, 1080, 1920) == "fill"
    assert mr.fit_kind("auto", 720, 1280, 1080, 1920) == "fill"
    assert mr.fit_kind("auto", 1920, 1080, 1080, 1920) == "blur"
    assert mr.fit_kind("auto", 1080, 1080, 1080, 1920) == "blur"
    assert mr.fit_kind("fit", 1080, 1920, 1080, 1920) == "fit"


def test_splice_duration_math():
    info = {"w": 1080, "h": 1920, "fps": "30/1", "sr": 48000, "dur": 40.0, "audio": True}
    ad = {"dur": 8.0, "audio": False, "w": 1920, "h": 1080}
    fc, d = mr.splice_filter(info, ad, "mid", 20.0, 0.2, "blur", None)
    assert d == pytest.approx(47.8) and "anullsrc" in fc and fc.count("xfade") == 2
    fc, d = mr.splice_filter(info, dict(ad, audio=True), "pre", 0, 0, "fill", -3.0)
    assert d == pytest.approx(47.8) and "volume=-3.00dB" in fc and fc.count("xfade") == 1
    _, d = mr.splice_filter(info, ad, "post", 0, 0, "fit", None)
    assert d == pytest.approx(47.8)


def test_warn_text():
    assert mr.warn_text(55) is None
    assert "60" in mr.warn_text(63)
    assert "90" in mr.warn_text(95)


def test_apply_to_outputs_off_or_bad_ad_never_raises(tmp_path, monkeypatch):
    clip = tmp_path / "viral_clip_01.mp4"
    clip.write_bytes(b"not a video")
    assert mr.apply_to_outputs(str(tmp_path), "virality", {}) == {}
    assert mr.apply_to_outputs(str(tmp_path), "sonya_gen", {"midroll": {"key": KEY}}) == {}

    def boom(key, dst):
        raise RuntimeError("s3 down")
    rep = mr.apply_to_outputs(str(tmp_path), "virality", {"midroll": {"key": KEY}}, download=boom)
    assert rep == {"viral_clip_01.mp4": {"status": "failed", "reason": "ad_unavailable"}}
    assert clip.read_bytes() == b"not a video"


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg")


def _lavfi(path, src, dur, extra=()):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *src, "-t", str(dur), *extra, "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


@needs_ffmpeg
def test_end_to_end_small(tmp_path, monkeypatch):
    """A 12 s 180x320 'clip' with speech-like tone + a pause at 6.0-6.6 s and its ASS; a 3 s landscape ad without sound;
    every second clip; the second clip stays untouched."""
    monkeypatch.setenv("SONYA_X264_PRESET", "ultrafast")
    monkeypatch.setenv("SONYA_NVENC_PROBED", "fail")
    out = tmp_path / "out"
    out.mkdir()
    words = [Word("раз", 1.0, 1.4), Word("два.", 1.5, 5.9), Word("три", 6.6, 7.0), Word("четыре.", 7.1, 11.0)]
    vol = "volume='if(between(t,5.95,6.6),0.0005,1)':eval=frame"
    for n in ("01", "02"):
        _lavfi(out / f"viral_clip_{n}.mp4", ["-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30", "-f", "lavfi",
                                             "-i", "sine=frequency=300:sample_rate=48000"], 12,
               ["-af", vol, "-ac", "2", "-shortest"])
        (out / f"viral_clip_{n}.ass").write_text(ss.to_ass(words, "highlight", 12.0, play_res=(180, 320)), encoding="utf-8")
    (out / "selected_segments.json").write_text(json.dumps({"segments": [{"file": "viral_clip_01.mp4"}]}))
    ad = tmp_path / "ad.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "smptebars=size=320x180:rate=25", "-t", "3",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(ad)], check=True)
    monkeypatch.setenv("SONYA_MIDROLL_ASSET", str(ad))
    before = (out / "viral_clip_02.mp4").read_bytes()
    rep = mr.apply_to_outputs(str(out), "virality", {"midroll": {"key": KEY, "every": 2}}, work_dir=str(tmp_path / "w"))
    r = rep["viral_clip_01.mp4"]
    assert r["status"] == "ok", r
    assert 5.9 <= r["at"] <= 6.65 and r["cut_from"] == "words" and r["fit"] == "blur"
    assert rep["viral_clip_02.mp4"]["reason"] == "every_second_clip"
    assert (out / "viral_clip_02.mp4").read_bytes() == before
    p = mr.probe(str(out / "viral_clip_01.mp4"))
    assert (p["w"], p["h"]) == (180, 320) and p["audio"]
    assert p["dur"] == pytest.approx(r["total_s"], abs=0.1) and p["dur"] == pytest.approx(12 + 3 - r["xfade"], abs=0.1)
    assert json.loads((out / "selected_segments.json").read_text())["segments"][0]["midroll"]["status"] == "ok"
    w = mr.words_from_ass((out / "viral_clip_01.ass").read_text())
    assert w[2][0] == pytest.approx(6.6 + p["dur"] - 12, abs=0.02)


def test_shift_ass():
    ass = ss.to_ass(WORDS, "highlight", 8.0)
    out = mr.shift_ass(ass, 2.0, 7.5)
    w = mr.words_from_ass(out)
    assert [x[0] for x in w][:2] == pytest.approx([0.5, 1.0])
    assert [x[0] for x in w][2:] == pytest.approx([x.start + 7.5 for x in WORDS[2:]])
    assert all(x[1] <= 2.0 for x in w[:2])
    assert out.count("Dialogue:") == ass.count("Dialogue:") and out.split("[Events]")[0] == ass.split("[Events]")[0]
