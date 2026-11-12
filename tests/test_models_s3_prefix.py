"""MODELS_S3_PREFIX: the new worker keeps its models under its own prefix; default `models/` = unchanged behaviour."""
import hashlib

import boto3
import pytest
from moto import mock_aws

from dispatcher import launcher
from scripts.ops import model_fetch
from scripts.ops.models_prefix import prefix, s3_key
from tools import models_publish as mp

BUCKET = "models-bucket"


@pytest.mark.parametrize("env,want", [({}, "models/common/y.pt"), ({"MODELS_S3_PREFIX": ""}, "models/common/y.pt"),
                                      ({"MODELS_S3_PREFIX": "models/"}, "models/common/y.pt"),
                                      ({"MODELS_S3_PREFIX": "worker-v2/models"}, "worker-v2/models/common/y.pt"),
                                      ({"MODELS_S3_PREFIX": "/worker-v2/models/"}, "worker-v2/models/common/y.pt")])
def test_prefix_mapping(env, want):
    assert s3_key("models/common/y.pt", env) == want
    assert s3_key("outputs/x.mp4", env) == "outputs/x.mp4"          # only model keys are remapped
    assert prefix({}) == "models/"


@mock_aws
def test_publish_and_fetch_use_the_prefix_and_never_touch_old_models(tmp_path, monkeypatch):
    s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="models/common/yolo11n.pt", Body=b"production build")        # the running worker's file
    data = b"new build weights" * 1000
    src = tmp_path / "y.pt"; src.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    row = {"key": "yolo11n", "s3_path": "models/common/yolo11n.pt", "path": str(src), "size": len(data), "sha256": sha}
    monkeypatch.setenv("MODELS_S3_PREFIX", "worker-v2/models/")
    log = mp.upload([row], BUCKET, s3, apply=False)
    assert log == [f"would upload worker-v2/models/common/yolo11n.pt ({len(data)/1e6:.1f} MB)"]     # no REPLACE: old key not looked at
    mp.upload([row], BUCKET, s3, apply=True)
    assert mp.upload([row], BUCKET, s3, apply=True)[0].startswith("skip   worker-v2/models/")
    assert s3.get_object(Bucket=BUCKET, Key="models/common/yolo11n.pt")["Body"].read() == b"production build"
    it = model_fetch.Item("yolo11n", "models/common/yolo11n.pt", tmp_path / "m" / "yolo11n.pt", False, sha, len(data))
    assert [r.status for r in model_fetch.fetch_all([it], client=s3, bucket=BUCKET)] == ["verified"]
    assert it.local.read_bytes() == data


def test_prefix_is_passed_to_the_instance_env():
    assert "MODELS_S3_PREFIX" in launcher.SECRET_PASSTHROUGH
