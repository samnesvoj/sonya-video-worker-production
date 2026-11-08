"""Where the worker's models live in the bucket. Default `models/` = unchanged behaviour.

MODELS_S3_PREFIX lets a new worker build keep its own copies (e.g. `worker-v2/models/`) next to the files the running
production worker uses, so publishing never overwrites them. mode.yaml / models.lock.json keep the logical `models/...` path;
only the physical S3 key is remapped here. Stdlib only (used by tools on machines without the worker deps)."""
from __future__ import annotations

import os
from typing import Mapping, Optional

DEFAULT = "models/"


def prefix(environ: Optional[Mapping[str, str]] = None) -> str:
    p = (environ if environ is not None else os.environ).get("MODELS_S3_PREFIX", "").strip().lstrip("/")
    return (p.rstrip("/") + "/") if p else DEFAULT


def s3_key(s3_path: str, environ: Optional[Mapping[str, str]] = None) -> str:
    """`models/common/x.pt` -> `<prefix>common/x.pt`; keys outside `models/` are left as they are."""
    p = prefix(environ)
    if p == DEFAULT or not s3_path.startswith(DEFAULT):
        return s3_path
    return p + s3_path[len(DEFAULT):]
