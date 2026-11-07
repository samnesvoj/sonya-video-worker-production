"""
model_downloader.py
===================
Downloads required model weights from S3 before a mode runs.
Reads model specs from the mode's mode.yaml.

In mode.yaml, local_path should be a relative path WITHOUT a leading 'models/' prefix:
  local_path: trailer/best.pt        → /opt/sonya/models/trailer/best.pt
  local_path: common/yolo11n.pt      → /opt/sonya/models/common/yolo11n.pt
  local_path: common/face_landmarker.task → /opt/sonya/models/common/face_landmarker.task

s3_path should be the full key in the models bucket:
  s3_path: models/trailer/best.pt
  s3_path: models/common/yolo11n.pt

S3 client always uses S3_ENDPOINT_URL (required for Timeweb / non-AWS S3).
Bucket: MODELS_S3_BUCKET → S3_BUCKET_NAME → S3_BUCKET (fallback chain).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Local model root — override via MODELS_LOCAL_DIR
_DEFAULT_LOCAL_DIR = Path("/opt/sonya/models")
_LOCAL_MODELS_DIR  = Path(os.environ.get("MODELS_LOCAL_DIR", str(_DEFAULT_LOCAL_DIR)))


def _resolve_local_path(spec_local_path: Optional[str], model_key: str) -> Path:
    """
    Resolve local_path from mode.yaml to an absolute filesystem path.

    Rules:
      - If absolute → use as-is
      - If relative and starts with 'models/' → strip the prefix, then join with _LOCAL_MODELS_DIR
        (prevents double 'models/models/...' if user accidentally wrote 'models/trailer/best.pt')
      - Otherwise → join with _LOCAL_MODELS_DIR directly
    """
    raw = spec_local_path or model_key
    p   = Path(raw)

    if p.is_absolute():
        return p

    # Strip redundant 'models/' prefix from relative path
    parts = p.parts
    if parts and parts[0] == "models":
        p = Path(*parts[1:]) if len(parts) > 1 else Path(model_key)

    return _LOCAL_MODELS_DIR / p


def _make_s3_client():
    """Create S3 client with endpoint_url — mandatory for Timeweb/compatible S3."""
    import boto3
    endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip()
    if not endpoint:
        raise RuntimeError(
            "S3_ENDPOINT_URL is required for model download. "
            "Set it to your S3-compatible endpoint (e.g. https://s3.timeweb.cloud)."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION", "ru-1"),
    )


def _get_models_bucket() -> str:
    b = (
        os.environ.get("MODELS_S3_BUCKET")
        or os.environ.get("S3_BUCKET_NAME")
        or os.environ.get("S3_BUCKET", "")
    )
    if not b:
        raise RuntimeError("MODELS_S3_BUCKET (or S3_BUCKET_NAME) env var not set")
    return b


def download_models_for_mode(mode_yaml_path: str | Path) -> bool:
    """
    Download all models declared in mode.yaml.
    Returns True if all required models are available.

    optional=True  → warning on missing, pipeline continues
    optional=False → raises RuntimeError (MODEL_DOWNLOAD_FAILED)
    """
    mode_yaml_path = Path(mode_yaml_path)
    if not mode_yaml_path.exists():
        raise FileNotFoundError(f"mode.yaml not found: {mode_yaml_path}")

    with open(mode_yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", {})
    if not models:
        return True

    try:
        import boto3
        import botocore
        import botocore.exceptions
    except ImportError:
        logger.warning("[model_dl] boto3 not available — skipping model download")
        return True

    bucket = _get_models_bucket()
    s3     = _make_s3_client()
    all_ok = True

    for model_key, spec in models.items():
        s3_path:  Optional[str] = spec.get("s3_path")
        optional: bool          = spec.get("optional", True)
        local_path              = _resolve_local_path(spec.get("local_path"), model_key)

        if local_path.exists():
            logger.info("[model_dl] already present: %s", local_path)
            continue

        if not s3_path:
            if not optional:
                raise RuntimeError(
                    f"MODEL_DOWNLOAD_FAILED: no s3_path for required model {model_key!r}"
                )
            logger.warning("[model_dl] no s3_path for optional model %s — skipping", model_key)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("[model_dl] downloading s3://%s/%s → %s", bucket, s3_path, local_path)
            s3.download_file(bucket, s3_path, str(local_path))
            logger.info("[model_dl] done: %s (%d bytes)", local_path.name, local_path.stat().st_size)
        except botocore.exceptions.ClientError as exc:
            if optional:
                logger.warning("[model_dl] optional model %s not found: %s", model_key, exc)
            else:
                raise RuntimeError(
                    f"MODEL_DOWNLOAD_FAILED: {model_key}: {exc}"
                ) from exc
            all_ok = False

    return all_ok
