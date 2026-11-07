"""
model_paths.py - ONE place that says where model files live.

Production layout (Vast): the code is a pinned release in /opt/sonya/releases/<sha>/ while models are downloaded (and sha256-verified)
into $MODELS_LOCAL_DIR (/opt/sonya/models). Several modules used to look in "<repo root>/models", which in that layout is NOT where the
downloader puts files - an optional model was then silently skipped with only a warning (quality loss without an error).
Rule: $SONYA_MODELS_DIR, else $MODELS_LOCAL_DIR, else the legacy "<repo root>/models" (unchanged behaviour for dev machines).
"""
from __future__ import annotations

import os
from pathlib import Path


def models_root() -> Path:
    for var in ("SONYA_MODELS_DIR", "MODELS_LOCAL_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            return Path(v)
    return Path(__file__).resolve().parents[2] / "models"


def model_file(*parts: str) -> Path:
    return models_root().joinpath(*parts)
