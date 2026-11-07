"""Burn-in subtitles from ASR words (scripts/shared/subtitles.py)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.shared import subtitles as st
from scripts.shared.crop import smart_crop_adapter as sca

WORDS = [  # absolute source timestamps, Whisper-style leading spaces
    {"word": " Привет", "start": 10.00, "end": 10.30}, {"word": " всем!", "start": 10.30, "end": 10.70},
    {"word": " Сегодня", "start": 11.40, "end": 11.90}, {"word": " я", "start": 11.90, "end": 12.00},
    {"word": " расскажу,", "start": 12.00, "end": 12.60}, {"word": " как", "start": 12.70, "end": 12.90},
    {"word": " монтировать", "start": 12.90, "end": 13.60}, {"word": " короткие", "start": 13.60, "end": 14.10},
    {"word": " ролики", "start": 14.10, "end": 14.50}, {"word": " {из}", "start": 14.50, "end": 14.70},
    {"word": " длинного", "start": 14.70, "end": 15.20}, {"word": " интервью.", "start": 15.20, "end": 15.90},
]
STYLE = st.SubtitleStyle()


def test_absolute_to_clip_local_and_edges_kept():
    # clip starts inside "Привет" and ends inside "интервью." — both must stay
    w = st.clip_words(WORDS, 10.15, 15.5)
    assert w[0].text == "Привет" and w[0].start == 0.0
    assert w[-1].text == "интервью." and w[-1].end == pytest.approx(5.35)
    assert all(0 <= x.start < x.end <= 5.35 + 1e-9 for x in w)


def test_no_words_and_no_intersection(tmp_path):
    assert st.subtitles_for_clip([], 0, 10, tmp_path / "a.ass")["subtitle_status"] == "no_word_timestamps"
    m = st.subtitles_for_clip(WORDS, 100, 10, tmp_path / "b.ass")
    assert m["subtitle_status"] == "no_words_in_interval" and not (tmp_path / "b.ass").exists()


def test_cues_max_two_lines_wrapped_between_words():
    cues = st.build_cues(st.clip_words(WORDS, 9.8, 16.2), STYLE, 6.4)
    texts = [" ".join(c.lines) for c in cues]
    assert " ".join(texts).split() == [w["word"].strip() for w in WORDS]  # every ASR word, in order, nothing added
    for c in cues:
        assert 1 <= len(c.lines) <= STYLE.max_lines
        assert all(len(l) <= STYLE.max_chars_per_line or " " not in l for l in c.lines)


def test_punctuation_and_pause_split_cues():
    cues = st.build_cues(st.clip_words(WORDS, 9.8, 16.2), STYLE, 6.4)
    assert cues[0].lines == ["Привет всем!"]           # sentence end + 0.7 s pause
    assert cues[-1].lines[-1].endswith("интервью.")


def test_timing_monotonic_min_duration_within_clip():
    cues = st.build_cues(st.clip_words(WORDS, 9.8, 16.2), STYLE, 6.4)
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start + 1e-9
    assert all(c.end - c.start >= min(STYLE.min_cue_sec, 0.3) for c in cues)
    assert cues[-1].end <= 6.4


def test_long_word_never_split():
    w = [st.Word("Превысокомногорассмотрительствующий", 0.0, 1.0), st.Word("да", 1.0, 1.2)]
    cues = st.build_cues(w, STYLE, 2.0)
    assert "Превысокомногорассмотрительствующий" in [l for c in cues for l in c.lines]


def test_ass_cyrillic_and_escaping(tmp_path):
    m = st.subtitles_for_clip(WORDS, 9.8, 6.4, tmp_path / "c.ass")
    text = (tmp_path / "c.ass").read_text(encoding="utf-8")
    assert "Привет всем!" in text and "PlayResX: 1080" in text and "PlayResY: 1920" in text
    body = text.split("[Events]")[1]
    assert "{" not in body and "}" not in body  # override braces neutralised
    assert "｛из｝" in body
    assert m["subtitle_cues"] >= 2 and m["subtitle_words"] == len(WORDS)
    assert m["subtitle_first_word"] == "Привет" and m["subtitle_last_word"] == "интервью."
    assert not list(tmp_path.glob("*.partial"))


def test_deterministic(tmp_path):
    st.subtitles_for_clip(WORDS, 9.8, 6.4, tmp_path / "x.ass")
    st.subtitles_for_clip(list(reversed(WORDS)), 9.8, 6.4, tmp_path / "y.ass")
    assert (tmp_path / "x.ass").read_text() == (tmp_path / "y.ass").read_text()


FULL = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"


@pytest.mark.skipif(not os.path.exists(FULL), reason="ffmpeg with libass not installed")
def test_crop_and_subtitles_in_one_render_pass(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x224466:s=1920x1080:r=25:d=7",
                    "-f", "lavfi", "-i", "sine=duration=7", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    ass = tmp_path / "clip.ass"
    st.subtitles_for_clip(WORDS, 9.8, 6.0, ass)
    calls = []
    real_run = subprocess.run
    monkeypatch.setenv("SONYA_FFMPEG", FULL)
    monkeypatch.setattr(sca.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or real_run(cmd, **kw))
    tr = sca.crop_track(None, 0, 6.0, width=1920, height=1080)
    out = tmp_path / "clip.mp4"
    sca.render_tracked(str(src), str(out), 0.5, 6.0, tr, subtitle_path=str(ass))
    assert len(calls) == 1 and calls[0][0] == FULL and any("ass=filename=" in a for a in calls[0])
    info = json.loads(real_run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json",
                                str(out)], capture_output=True, text=True).stdout)
    assert (info["streams"][0]["width"], info["streams"][0]["height"]) == (1080, 1920)
    # a frame inside the first cue (≈0.5 s) has white text pixels; plain background has none
    frame = tmp_path / "f.png"
    real_run(["ffmpeg", "-loglevel", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1", str(frame)], check=True)
    cv2 = pytest.importorskip("cv2")
    img = cv2.imread(str(frame))
    assert (img > 240).all(axis=2).sum() > 500


@pytest.mark.skipif(not os.path.exists(FULL), reason="ffmpeg with libass not installed")
def test_project_path_with_quote_colon_and_space(tmp_path, monkeypatch):
    d = tmp_path / "Мой проект 'Интервью': финал"
    d.mkdir()
    src = d / "src.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=1280x720:r=25:d=3",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)], check=True)
    ass = d / "viral_clip_01.ass"
    st.subtitles_for_clip(WORDS, 9.8, 3.0, ass)
    monkeypatch.setenv("SONYA_FFMPEG", FULL)
    out = d / "viral_clip_01.mp4"
    sca.render_tracked(str(src), str(out), 0, 3.0, sca.crop_track(None, 0, 3, width=1280, height=720),
                       subtitle_path=str(ass))
    assert out.stat().st_size > 0


def test_unsafe_subtitle_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        sca.render_tracked("in.mp4", str(tmp_path / "o.mp4"), 0, 1, sca.crop_track(None, 0, 1, width=640, height=360),
                           subtitle_path=str(tmp_path / "bad'name.ass"))
