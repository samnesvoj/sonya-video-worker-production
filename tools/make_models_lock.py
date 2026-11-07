#!/usr/bin/env python3
"""Pin sha256+size of every model of the enabled/opt-in modes. Trusted machine with S3 read access (S3_*, MODELS_S3_BUCKET)."""
import hashlib, json, os, pathlib, sys, tempfile

import boto3, yaml

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scripts.mode_policy import DEFAULT_ENABLED_MODES, OPT_IN_MODES  # noqa: E402

s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name=os.environ.get("S3_REGION"))
models = {}
for mode in (*DEFAULT_ENABLED_MODES, *OPT_IN_MODES):
    cfg = root / "modes" / mode / "mode.yaml"
    if not cfg.is_file():
        continue
    for key, spec in ((yaml.safe_load(cfg.read_text()) or {}).get("models") or {}).items():
        if not spec.get("s3_path"):
            continue
        with tempfile.NamedTemporaryFile() as t:
            s3.download_file(os.environ["MODELS_S3_BUCKET"], spec["s3_path"], t.name)
            data = pathlib.Path(t.name).read_bytes()
            models[key] = {"s3_path": spec["s3_path"], "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
(root / "release" / "models.lock.json").write_text(json.dumps({"schema_version": 1, "models": models}, indent=2) + "\n")
print(f"{len(models)} models pinned")
