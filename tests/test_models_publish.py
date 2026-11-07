import hashlib
import json
import os
import pathlib

import boto3
import pytest
from moto import mock_aws

from tools import models_publish as mp

MAN = mp.load_manifest()
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _populate(tmp_path, only=None, custom=()):
    """Create fake local model files at the exact relative paths; stock hashes cannot match fake bytes, so tests patch the manifest."""
    for m in MAN["models"]:
        if only and m["key"] not in only:
            continue
        p = tmp_path / m["local_path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((m["key"] + ("-CUSTOM" if m["key"] in custom else "")).encode() * 10)


def _fake_manifest(tmp_path, stock_keys=()):
    """Manifest whose public hashes match the fake files for `stock_keys` (and are None for the private model)."""
    man = json.loads(json.dumps(MAN))
    for m in man["models"]:
        if m["key"] in stock_keys:
            m["public_sha256"] = hashlib.sha256((m["key"].encode()) * 10).hexdigest()
    return man


def test_manifest_matches_mode_yaml_exactly_and_covers_both_modes():
    import yaml
    declared = {}
    for mode in ("virality", "beep"):
        for key, spec in yaml.safe_load((ROOT / "modes" / mode / "mode.yaml").read_text())["models"].items():
            declared[key] = (spec["s3_path"], spec["local_path"])
    listed = {m["key"]: (m["s3_path"], m["local_path"]) for m in MAN["models"]}
    assert declared == listed                                  # the doc/manifest cannot drift from what the worker really fetches
    assert {"yolo11n", "yolo11n_pose", "webcam_detector"} <= set(listed) and sum(k.startswith("whisper_base_") for k in listed) == 4
    assert not any("trailer" in k or "trailer" in v[0] for k, v in listed.items())
    assert all(m["used_by"] and m["loaded_by"] and m["source"]["type"] in ("public", "private") for m in MAN["models"])


def test_public_hashes_and_sizes_recorded_for_public_models():
    for m in MAN["models"]:
        if m["source"]["type"] == "public":
            assert len(m["public_sha256"]) == 64 and m["public_size"] > 0 and m["source"]["url"].startswith("https://")
        else:
            assert m["public_sha256"] is None


def test_scan_classifies_matching_custom_and_missing(tmp_path):
    _populate(tmp_path, custom={"yolo11n_pose"})
    man = _fake_manifest(tmp_path, stock_keys={"yolo11n", "whisper_base_config"})
    (tmp_path / "common" / "yolo11n-pose.pt").write_bytes(b"different")
    (tmp_path / "whisper-base" / "model.bin").unlink()
    rows = {r["key"]: r for r in mp.scan(man, [tmp_path], {})}
    assert rows["yolo11n"]["status"] == "matches_public" and rows["yolo11n_pose"]["status"] == "differs_from_public"
    assert rows["whisper_base_model"]["status"] == "missing" and rows["webcam_detector"]["status"] == "private_ok"


def test_finds_models_in_a_foreign_layout_by_name_and_reports_ambiguity(tmp_path):
    src = tmp_path / "old" / "stuff" / "faster-whisper-base-snapshot"
    src.mkdir(parents=True)
    (src / "model.bin").write_bytes(b"w")
    chosen, cands = mp.locate(next(m for m in MAN["models"] if m["key"] == "whisper_base_model"), [tmp_path / "old"], {})
    assert chosen == (src / "model.bin").resolve()
    (tmp_path / "old" / "other" / "whisper-base").mkdir(parents=True)
    (tmp_path / "old" / "other" / "whisper-base" / "model.bin").write_bytes(b"w2")
    chosen, cands = mp.locate(next(m for m in MAN["models"] if m["key"] == "whisper_base_model"), [tmp_path / "old"], {})
    assert chosen is None and len(cands) == 2


def test_lock_refuses_incomplete_or_unexpected_files_and_accepts_explicit_custom(tmp_path):
    _populate(tmp_path)
    man = _fake_manifest(tmp_path, stock_keys={m["key"] for m in MAN["models"] if m.get("public_sha256")})
    (tmp_path / "common" / "yolo11n.pt").write_bytes(b"not the stock file")
    rows = mp.scan(man, [tmp_path], {})
    with pytest.raises(SystemExit, match="differs from the public"):
        mp.make_lock(rows, set())
    lock = mp.make_lock(rows, {"yolo11n"})
    assert set(lock["models"]) == {m["key"] for m in MAN["models"]} and lock["models"]["yolo11n"]["size"] == len(b"not the stock file")
    (tmp_path / "common" / "webcam_detector.pt").unlink()
    with pytest.raises(SystemExit, match="missing"):
        mp.make_lock(mp.scan(man, [tmp_path], {}), {"yolo11n"})


def test_lock_output_is_consumable_by_the_worker_fetcher(tmp_path):
    from scripts.ops import model_fetch
    _populate(tmp_path)
    lock = mp.make_lock(mp.scan(MAN, [tmp_path], {}), {m["key"] for m in MAN["models"] if m.get("public_sha256")})
    items = model_fetch.plan(ROOT, ["virality", "beep"], tmp_path / "dest", lock)
    assert all(i.sha256 and i.size for i in items) and len(items) == 7


@mock_aws
def test_upload_is_dry_run_by_default_requires_confirmation_and_is_idempotent(tmp_path):
    _populate(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket="models-bucket")
    rows = mp.scan(MAN, [tmp_path], {})
    custom = {m["key"] for m in MAN["models"] if m.get("public_sha256")}
    mp.make_lock(rows, custom)
    log = mp.upload(rows, "models-bucket", s3, apply=False)
    assert all(l.startswith("would upload") for l in log) and "Contents" not in s3.list_objects_v2(Bucket="models-bucket")   # nothing uploaded
    with pytest.raises(SystemExit, match="confirm"):
        mp.main(["upload", "--src", str(tmp_path), "--bucket", "models-bucket", "--apply"] + sum([["--accept-custom", k] for k in custom], []))
    log = mp.upload(rows, "models-bucket", s3, apply=True)
    assert len([l for l in log if l.startswith("uploaded")]) == 7
    head = s3.head_object(Bucket="models-bucket", Key="models/common/yolo11n.pt")
    assert head["Metadata"]["sha256"] == next(r["sha256"] for r in rows if r["key"] == "yolo11n")
    assert all(l.startswith("skip") for l in mp.upload(rows, "models-bucket", s3, apply=True))                               # second run: nothing re-uploaded
    (tmp_path / "common" / "yolo11n.pt").write_bytes(b"changed")
    again = mp.scan(MAN, [tmp_path], {})
    assert any(l.startswith("REPLACE") for l in mp.upload(again, "models-bucket", s3, apply=True))


def test_no_secret_value_is_ever_printed(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "AKIA-LEAKCHECK"); monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "SECRET-LEAKCHECK")
    _populate(tmp_path)
    mp.main(["scan", "--src", str(tmp_path)])
    cap = capsys.readouterr()
    assert "LEAKCHECK" not in cap.out + cap.err
