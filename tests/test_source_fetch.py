import sys
import types
from pathlib import Path

import pytest

from scripts import source_fetch as sfm
from scripts import url_ingest


@pytest.mark.parametrize("msg,code", [
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", "SOURCE_PRIVATE"),
    ("Video unavailable. This video has been removed by the uploader", "SOURCE_UNAVAILABLE"),
    ("Sign in to confirm your age", "SOURCE_AGE_RESTRICTED"),
    ("boom", "SOURCE_DOWNLOAD_FAILED"),
])
def test_classify(msg, code):
    assert sfm.classify_error(url_ingest.DownloadFailed(msg))[0] == code


def _patch(monkeypatch, tmp_path, duration, uploaded):
    f = tmp_path / "dl" / "v.mp4"
    f.parent.mkdir()
    f.write_bytes(b"x" * 10)
    monkeypatch.setattr(url_ingest, "assert_safe_url", lambda u: None)
    monkeypatch.setattr(url_ingest, "probe", lambda *a, **k: {})
    monkeypatch.setattr(url_ingest, "download_video", lambda *a, **k: (str(f), ".mp4"))
    monkeypatch.setattr(sfm, "_duration", lambda p: duration)
    fake = types.ModuleType("scripts.prod_s3_storage")
    fake.upload_file = lambda p, k, content_type=None: uploaded.append((str(p), k))
    monkeypatch.setitem(sys.modules, "scripts.prod_s3_storage", fake)


def test_ok_uploads(monkeypatch, tmp_path):
    up = []
    _patch(monkeypatch, tmp_path, 60.0, up)
    work = tmp_path / "w"; work.mkdir()
    r = sfm.fetch_source({"url": "https://youtu.be/abc", "platform": "youtube", "max_source_sec": 600}, "virality", "in/k.mp4", work)
    assert up == [(str(work / "input.mp4"), "in/k.mp4")] and Path(r["path"]).exists()


def test_too_long_for_plan(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, 4000.0, [])
    work = tmp_path / "w"; work.mkdir()
    with pytest.raises(sfm.SourceFetchError) as e:
        sfm.fetch_source({"url": "https://youtu.be/abc", "platform": "youtube", "max_source_sec": 1800}, "virality", "k", work)
    assert e.value.code == "SOURCE_TOO_LONG" and "30 минут" in e.value.message


def test_unsupported(tmp_path):
    with pytest.raises(sfm.SourceFetchError) as e:
        sfm.fetch_source({"url": "ftp://x"}, "virality", "k", tmp_path)
    assert e.value.code == "SOURCE_UNSUPPORTED"


def test_trim_first_minutes_instead_of_refusal(monkeypatch, tmp_path):
    up, seen = [], {}
    _patch(monkeypatch, tmp_path, 3660.0, up)          # downloaded section came out 61 min (keyframe rounding)
    monkeypatch.setattr(url_ingest, "probe", lambda *a, **k: seen.setdefault("probe_cap", k["max_duration_sec"]) and {})
    orig = url_ingest.download_video
    monkeypatch.setattr(url_ingest, "download_video", lambda *a, **k: (seen.setdefault("trim", k.get("trim_sec")), orig(*a, **k))[1])
    durs = iter([3660.0, 3600.0])
    monkeypatch.setattr(sfm, "_duration", lambda p: next(durs))
    monkeypatch.setattr(url_ingest, "trim_file", lambda p, s: seen.setdefault("cut", s) and p)
    work = tmp_path / "w"; work.mkdir()
    sfm.fetch_source({"url": "https://youtu.be/abc", "platform": "youtube", "max_source_sec": 3600, "trim_first_sec": 3600}, "virality", "k", work)
    assert seen["trim"] == 3600 and seen["cut"] == 3600 and seen["probe_cap"] > 3600 and up


def test_trim_never_exceeds_cap(monkeypatch, tmp_path):
    seen = {}
    _patch(monkeypatch, tmp_path, 100.0, [])
    monkeypatch.setattr(url_ingest, "download_video", lambda *a, **k: (seen.setdefault("trim", k.get("trim_sec")), (str(tmp_path / "dl" / "v.mp4"), ".mp4"))[1])
    work = tmp_path / "w"; work.mkdir()
    sfm.fetch_source({"url": "https://youtu.be/abc", "platform": "youtube", "max_source_sec": 3600, "trim_first_sec": 99999}, "virality", "k", work)
    assert seen["trim"] == 3600
