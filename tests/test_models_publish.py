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


CUSTOM_OK = {m["key"] for m in MAN["models"] if m.get("public_sha256") or m.get("upstream_md5")}


def _fake_manifest(tmp_path, stock_keys=()):
    """Manifest whose public hash / upstream md5 / size match the fake files for `stock_keys` (private model: nothing to match)."""
    man = json.loads(json.dumps(MAN))
    for m in man["models"]:
        data = m["key"].encode() * 10
        if m["key"] in stock_keys:
            if m.get("upstream_md5"):
                m["upstream_md5"] = hashlib.md5(data).hexdigest()
            if m.get("public_sha256") is not None or m["source"]["type"] == "public":
                m["public_sha256"] = hashlib.sha256(data).hexdigest() if m.get("public_sha256") else None
            if m.get("public_size"):
                m["public_size"] = len(data)
    return man


def test_manifest_matches_mode_yaml_exactly_and_covers_both_modes():
    import yaml
    declared = {}
    for mode in ("virality", "beep"):
        for key, spec in yaml.safe_load((ROOT / "modes" / mode / "mode.yaml").read_text())["models"].items():
            declared[key] = (spec["s3_path"], spec["local_path"])
    listed = {m["key"]: (m["s3_path"], m["local_path"]) for m in MAN["models"]}
    assert declared == listed                                  # the doc/manifest cannot drift from what the worker really fetches
    assert {"yolo11n", "yolo11n_pose", "webcam_detector", "gigaam_v3_e2e_rnnt_ckpt", "gigaam_v3_e2e_rnnt_tokenizer"} <= set(listed)
    assert sum(k.startswith("whisper_base_") for k in listed) == 4
    assert not any("trailer" in k or "trailer" in v[0] for k, v in listed.items())
    assert all(m["used_by"] and m["loaded_by"] and m["source"]["type"] in ("public", "private") for m in MAN["models"])


def test_public_hashes_and_sizes_recorded_for_public_models():
    for m in MAN["models"]:
        if m["source"]["type"] == "public" and not m["key"].startswith("gigaam"):
            assert len(m["public_sha256"]) == 64 and m["public_size"] > 0 and m["source"]["url"].startswith("https://")
        elif m["key"].startswith("gigaam"):
            assert m["public_sha256"] is None and m["public_size"] > 0                     # CDN cert expired: authoritative sha256 comes from the Mac mini scan
        else:
            assert m["public_sha256"] is None
    assert next(m for m in MAN["models"] if m["key"] == "gigaam_v3_e2e_rnnt_ckpt")["upstream_md5"] == "2730de7545ac43ad256485a462b0a27a"


def test_gigaam_is_primary_and_whisper_is_the_optional_fallback_in_both_modes():
    import yaml
    roles = {m["key"]: m["role"] for m in MAN["models"]}
    assert roles["gigaam_v3_e2e_rnnt_ckpt"] == "primary" and roles["whisper_base_model"] == "fallback"
    for mode in ("virality", "beep"):
        models = yaml.safe_load((ROOT / "modes" / mode / "mode.yaml").read_text())["models"]
        assert models["gigaam_v3_e2e_rnnt_ckpt"]["optional"] is False and models["gigaam_v3_e2e_rnnt_tokenizer"]["optional"] is False
        assert all(v["optional"] is True for k, v in models.items() if k.startswith("whisper_base_"))


def test_scan_classifies_matching_custom_and_missing(tmp_path):
    _populate(tmp_path, custom={"yolo11n_pose"})
    man = _fake_manifest(tmp_path, stock_keys={"yolo11n", "whisper_base_config", "gigaam_v3_e2e_rnnt_ckpt"})
    (tmp_path / "common" / "yolo11n-pose.pt").write_bytes(b"different")
    (tmp_path / "whisper-base" / "model.bin").unlink()
    rows = {r["key"]: r for r in mp.scan(man, [tmp_path], {})}
    assert rows["yolo11n"]["status"] == "matches_public" and rows["yolo11n_pose"]["status"] == "differs_from_public"
    assert rows["whisper_base_model"]["status"] == "missing" and rows["webcam_detector"]["status"] == "private_ok"
    assert rows["gigaam_v3_e2e_rnnt_ckpt"]["status"] == "matches_upstream_md5"          # cross-checked against upstream's own MD5


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
    man = _fake_manifest(tmp_path, stock_keys=CUSTOM_OK)
    (tmp_path / "common" / "yolo11n.pt").write_bytes(b"not the stock file")
    rows = mp.scan(man, [tmp_path], {})
    with pytest.raises(SystemExit, match="differs from the public"):
        mp.make_lock(rows, set())
    lock = mp.make_lock(rows, {"yolo11n"})
    assert set(lock["models"]) == {m["key"] for m in MAN["models"]} and lock["models"]["yolo11n"]["size"] == len(b"not the stock file")
    (tmp_path / "common" / "webcam_detector.pt").unlink()
    with pytest.raises(SystemExit, match="missing"):
        mp.make_lock(mp.scan(man, [tmp_path], {}), {"yolo11n"})


def test_wrong_gigaam_checkpoint_is_rejected_by_upstream_md5_and_size(tmp_path):
    _populate(tmp_path)
    rows = {r["key"]: r for r in mp.scan(MAN, [tmp_path], {})}
    assert rows["gigaam_v3_e2e_rnnt_ckpt"]["status"] == "differs_from_upstream"
    with pytest.raises(SystemExit, match="upstream"):
        mp.make_lock(list(rows.values()), CUSTOM_OK - {"gigaam_v3_e2e_rnnt_ckpt"})


def test_lock_output_is_consumable_by_the_worker_fetcher(tmp_path):
    from scripts.ops import model_fetch
    _populate(tmp_path)
    lock = mp.make_lock(mp.scan(MAN, [tmp_path], {}), CUSTOM_OK)
    items = model_fetch.plan(ROOT, ["virality", "beep"], tmp_path / "dest", lock)
    assert all(i.sha256 and i.size for i in items) and len(items) == 9


@mock_aws
def test_upload_is_dry_run_by_default_requires_confirmation_and_is_idempotent(tmp_path):
    _populate(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket="models-bucket")
    rows = mp.scan(MAN, [tmp_path], {})
    custom = CUSTOM_OK
    mp.make_lock(rows, custom)
    log = mp.upload(rows, "models-bucket", s3, apply=False)
    assert all(l.startswith("would upload") for l in log) and "Contents" not in s3.list_objects_v2(Bucket="models-bucket")   # nothing uploaded
    with pytest.raises(SystemExit, match="confirm"):
        mp.main(["upload", "--src", str(tmp_path), "--bucket", "models-bucket", "--apply"] + sum([["--accept-custom", k] for k in custom], []))
    log = mp.upload(rows, "models-bucket", s3, apply=True)
    assert len([l for l in log if l.startswith("uploaded")]) == 9
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
