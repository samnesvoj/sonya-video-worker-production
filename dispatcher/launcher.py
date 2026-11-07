"""Offer selection and the create-instance payload (central side). Secrets are assembled here, from the dispatcher's
own environment, and exist only in the create call for that one instance - never in a Template, image, log or Git."""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dispatcher.config import FleetConfig

SECRET_PASSTHROUGH = ("BACKEND_API_URL", "WORKER_SECRET", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
                      "S3_BUCKET_NAME", "S3_BUCKET", "S3_REGION", "MODELS_S3_BUCKET")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


class LaunchError(RuntimeError):
    pass


def validate_release(cfg: FleetConfig) -> None:
    if not SHA_RE.match(cfg.release_sha):
        raise LaunchError("SONYA_RELEASE_SHA must be a full 40-hex commit SHA (never a branch)")
    if not DIGEST_RE.search(cfg.image):
        raise LaunchError("SONYA_WORKER_IMAGE must be pinned by @sha256 digest (never a mutable tag)")


def search_query(cfg: FleetConfig) -> dict:
    return {"verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": cfg.min_vram_gb * 1000}, "disk_space": {"gte": cfg.disk_gb},
            "reliability2": {"gte": cfg.min_reliability}, "inet_down": {"gte": cfg.min_inet_down_mbps},
            "cuda_max_good": {"gte": 12.4}, "dph_total": {"lte": float(cfg.max_dph)}, "type": "on-demand",
            "order": [["dph_total", "asc"]], "limit": 50}


def pick_offer(offers: List[dict], cfg: FleetConfig, banned_machines: set) -> Optional[dict]:
    ok = [o for o in offers if o.get("gpu_name") in cfg.allowed_gpus and o.get("machine_id") not in banned_machines
          and Decimal(str(o.get("dph_total", 99))) <= cfg.max_dph]
    # rank by the FULL hourly cost with our disk (stopped/running storage is billed too), not by the headline dph_total
    cost = lambda o: Decimal(str(o.get("dph_base", o["dph_total"]))) + Decimal(str(o.get("storage_cost", 0))) * cfg.disk_gb / Decimal(720)
    return min(ok, key=cost) if ok else None


def create_payload(cfg: FleetConfig, label: str, github_token: str, environ: Mapping[str, str]) -> Dict[str, Any]:
    validate_release(cfg)
    if not github_token:
        raise LaunchError("no GitHub release token")
    env = {"SONYA_RELEASE_SHA": cfg.release_sha, "SONYA_RELEASE_REPO": cfg.release_repo, "GITHUB_RELEASE_TOKEN": github_token,
           "WORKER_ID": label, "WORKER_BACKEND_MODE": "api", "ENABLED_MODES": ",".join(cfg.pool_modes),
           "SONYA_REQUIRE_MODEL_LOCK": "1", "SONYA_MIN_VRAM_MB": str(cfg.min_vram_gb * 1000 - 4000)}
    missing = [k for k in ("BACKEND_API_URL", "WORKER_SECRET", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_REGION", "MODELS_S3_BUCKET")
               if not environ.get(k)]
    if missing or not (environ.get("S3_BUCKET_NAME") or environ.get("S3_BUCKET")):
        raise LaunchError(f"dispatcher is missing worker env: {', '.join(missing) or 'S3_BUCKET_NAME'}")
    env.update({k: environ[k] for k in SECRET_PASSTHROUGH if environ.get(k)})
    return {"client_id": "me", "image": cfg.image, "env": env, "disk": cfg.disk_gb, "label": label, "runtype": "args"}


def redacted(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Safe-to-log form: env VALUES are never included."""
    out = {k: v for k, v in payload.items() if k != "env"}
    out["env_keys"] = sorted(payload.get("env", {}))
    return out
