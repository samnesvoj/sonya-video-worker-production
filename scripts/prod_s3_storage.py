"""
prod_s3_storage.py
==================
S3 storage layer for SONYA production pipeline.

Key patterns:
  input:  users/{user_id}/jobs/{job_id}/{mode}/input/original{ext}
  output: users/{user_id}/jobs/{job_id}/{mode}/output/{filename}
  debug:  users/{user_id}/jobs/{job_id}/{mode}/debug/{filename}
  model:  models/{relative_path}  (e.g. models/trailer/best.pt)

Env vars:
  S3_BUCKET_NAME     — main bucket (fallback: S3_BUCKET)
  S3_ENDPOINT_URL    — required for Timeweb / non-AWS S3
  S3_ACCESS_KEY_ID
  S3_SECRET_ACCESS_KEY
  S3_REGION          — default: ru-1
  MODELS_S3_BUCKET   — models bucket (fallback: S3_BUCKET_NAME)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_S3_AVAILABLE = False
try:
    import boto3
    import botocore
    import botocore.exceptions
    _S3_AVAILABLE = True
except ImportError:
    pass


# ── Client factory —————————————————————————————————————————————————————————————

def _make_client(bucket_env_keys: tuple[str, ...] = ("S3_BUCKET_NAME", "S3_BUCKET")):
    """Create a fully-configured boto3 S3 client (always with endpoint_url)."""
    if not _S3_AVAILABLE:
        raise RuntimeError("boto3 not installed — pip install boto3")
    endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip()
    if not endpoint:
        raise RuntimeError(
            "S3_ENDPOINT_URL is required. Set it to your S3-compatible endpoint "
            "(e.g. https://s3.timeweb.cloud or https://s3.amazonaws.com)"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION", "ru-1"),
    )


def _bucket() -> str:
    b = os.environ.get("S3_BUCKET_NAME") or os.environ.get("S3_BUCKET", "")
    if not b:
        raise RuntimeError("S3_BUCKET_NAME env var not set")
    return b


def _model_bucket() -> str:
    b = (
        os.environ.get("MODELS_S3_BUCKET")
        or os.environ.get("S3_BUCKET_NAME")
        or os.environ.get("S3_BUCKET", "")
    )
    if not b:
        raise RuntimeError("MODELS_S3_BUCKET env var not set")
    return b


def _client():
    return _make_client()


# ── Key builders ———————————————————————————————————————————————————————————————

def build_input_key(user_id: str, job_id: str, mode: str, ext: str) -> str:
    """
    Build S3 key for job input file.
    Pattern: users/{user_id}/jobs/{job_id}/{mode}/input/original{ext}
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"users/{user_id}/jobs/{job_id}/{mode}/input/original{ext}"


def build_output_key(user_id: str, job_id: str, mode: str, filename: str) -> str:
    """
    Build S3 key for job output file.
    Pattern: users/{user_id}/jobs/{job_id}/{mode}/output/{filename}
    """
    safe_name = Path(filename).name
    return f"users/{user_id}/jobs/{job_id}/{mode}/output/{safe_name}"


def build_debug_key(user_id: str, job_id: str, mode: str, filename: str) -> str:
    """
    Build S3 key for debug/enrichment artifacts.
    Pattern: users/{user_id}/jobs/{job_id}/{mode}/debug/{filename}
    """
    safe_name = Path(filename).name
    return f"users/{user_id}/jobs/{job_id}/{mode}/debug/{safe_name}"


def build_model_key(model_path: str) -> str:
    """
    Build S3 key for a model file.
    Pattern: models/{model_path}  (e.g. models/trailer/best.pt)
    model_path should be relative without leading 'models/' (e.g. trailer/best.pt).
    If already starts with 'models/' — use as-is to avoid double prefix.
    """
    model_path = model_path.lstrip("/")
    if not model_path.startswith("models/"):
        model_path = f"models/{model_path}"
    return model_path


# ── Upload ——————————————————————————————————————————————————————————————————————

def upload_file(
    local_path: str | Path,
    s3_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload a local file to S3. Returns s3_key."""
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    _client().upload_file(
        str(local_path),
        _bucket(),
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    logger.info("[s3] uploaded %s → s3://%s/%s (%d bytes)",
                local_path.name, _bucket(), s3_key, local_path.stat().st_size)
    return s3_key


def upload_bytes(
    data: bytes,
    s3_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes directly to S3. Returns s3_key."""
    import io
    _client().upload_fileobj(
        io.BytesIO(data),
        _bucket(),
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    logger.info("[s3] uploaded bytes → s3://%s/%s (%d bytes)", _bucket(), s3_key, len(data))
    return s3_key


# ── Download ————————————————————————————————————————————————————————————————————

def download_file(s3_key: str, local_path: str | Path) -> Path:
    """Download S3 object to local_path. Returns Path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(_bucket(), s3_key, str(local_path))
    logger.info("[s3] downloaded s3://%s/%s → %s (%d bytes)",
                _bucket(), s3_key, local_path, local_path.stat().st_size)
    return local_path


def download_model(s3_key: str, local_path: str | Path) -> Path:
    """Download a model file from the models bucket."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(_model_bucket(), s3_key, str(local_path))
    logger.info("[s3] model downloaded s3://%s/%s → %s", _model_bucket(), s3_key, local_path)
    return local_path


# ── Presigned URL ———————————————————————————————————————————————————————————————

def generate_presigned_get_url(s3_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL for an S3 object."""
    url = _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": s3_key},
        ExpiresIn=expires_in,
    )
    logger.debug("[s3] presigned_url key=%s expires=%ds", s3_key, expires_in)
    return url


# ── Delete ——————————————————————————————————————————————————————————————————————

def delete_object(s3_key: str, bucket: Optional[str] = None) -> bool:
    """
    Best-effort delete of an S3 object. Never raises -- returns True on a
    confirmed delete (S3 DELETE is idempotent; a missing object is not an
    error), False if the call itself failed (logged as a warning).

    Used to clean up an orphaned upload: POST /api/generation/jobs uploads
    the input file before it knows whether the DB insert will win an
    Idempotency-Key conflict. Callers must only ever pass the s3_key just
    uploaded by the losing request -- never an existing job's
    s3_input_key.
    """
    try:
        b = bucket or _bucket()
        _client().delete_object(Bucket=b, Key=s3_key)
        logger.info("[s3] deleted orphaned object s3://%s/%s", b, s3_key)
        return True
    except Exception as exc:
        logger.warning("[s3] delete_failed key=%s exc=%s", s3_key, exc)
        return False


# ── Existence check —————————————————————————————————————————————————————————————

def object_exists(s3_key: str, bucket: Optional[str] = None) -> bool:
    """Return True if the S3 object exists."""
    b = bucket or _bucket()
    try:
        _client().head_object(Bucket=b, Key=s3_key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


# ── Health check ————————————————————————————————————————————————————————————————

def health_check() -> dict:
    """
    Verify S3 connectivity. Returns dict with status and bucket info.
    Non-raising — returns error details instead.
    """
    try:
        b = _bucket()
        _client().head_bucket(Bucket=b)
        return {"status": "ok", "bucket": b, "endpoint": os.environ.get("S3_ENDPOINT_URL")}
    except Exception as exc:
        logger.error("[s3] health_check_failed: %s", exc)
        return {"status": "error", "error": str(exc)[:200]}
