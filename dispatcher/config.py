"""
Fleet configuration (central dispatcher side - NEVER shipped in the worker image).
All values come from the dispatcher environment. Defaults are the safe ones:
dry-run, one instance, 60 s idle timeout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Tuple

TRUE = {"1", "true", "yes", "on"}


def _int(env: Dict[str, str], key: str, default: int, lo: int = 0, hi: int = 10**9) -> int:
    try:
        v = int(env.get(key, default))
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}]")
    return v


@dataclass(frozen=True)
class FleetConfig:
    enabled: bool = False                    # SONYA_FLEET_ENABLED - nothing happens unless true
    dry_run: bool = True                     # SONYA_FLEET_DRY_RUN - log intended Vast calls, do not send
    idle_timeout_seconds: int = 60           # IDLE_TIMEOUT_SECONDS
    tick_seconds: int = 10
    boot_grace_seconds: int = 600            # a fresh instance is not 'idle' before it could boot and claim
    max_instances: int = 1
    jobs_per_instance: int = 1               # concurrency of one worker (gpu_worker handles one job at a time)
    stopped_retention_seconds: int = 1800    # stopped longer than this -> DESTROY (storage keeps billing; see docs/COSTS.md)
    resume_timeout_seconds: int = 90         # 'scheduling' longer than this -> GPU likely taken (Vast docs: ~30 s)
    max_resume_failures: int = 2
    create_timeout_seconds: int = 900        # create -> running must happen within this
    max_creates_per_hour: int = 6            # runaway-cost guard
    max_running_hours: int = 24              # hard cap for a single continuous run
    pool_modes: Tuple[str, ...] = ("virality", "beep")
    label_prefix: str = "sonya-pool-"
    # Offer filter
    min_vram_gb: int = 24
    disk_gb: int = 80
    max_dph: Decimal = Decimal("0.60")       # USD/h price cap per offer
    allowed_gpus: Tuple[str, ...] = ("RTX 3090", "RTX 4090", "RTX A5000", "RTX A6000", "L4", "A10", "L40", "L40S")
    min_reliability: float = 0.98
    min_inet_down_mbps: int = 200
    # Release identity (passed to the worker as env; the token is minted per instance)
    image: str = ""
    release_sha: str = ""
    release_repo: str = "samnesvoj/sonya-video-worker-production"

    @classmethod
    def from_env(cls, env: Dict[str, str] | None = None) -> "FleetConfig":
        e = dict(os.environ if env is None else env)
        idle = _int(e, "IDLE_TIMEOUT_SECONDS", 60, lo=10, hi=86400)
        cfg = cls(
            enabled=e.get("SONYA_FLEET_ENABLED", "").lower() in TRUE,
            dry_run=e.get("SONYA_FLEET_DRY_RUN", "true").lower() in TRUE,
            idle_timeout_seconds=idle,
            tick_seconds=_int(e, "SONYA_FLEET_TICK_SECONDS", 10, lo=2, hi=300),
            boot_grace_seconds=_int(e, "SONYA_BOOT_GRACE_SECONDS", 600, lo=60, hi=3600),
            max_instances=_int(e, "SONYA_FLEET_MAX_INSTANCES", 1, lo=1, hi=8),
            stopped_retention_seconds=_int(e, "SONYA_STOPPED_RETENTION_SECONDS", 1800, lo=0, hi=7 * 86400),
            resume_timeout_seconds=_int(e, "SONYA_RESUME_TIMEOUT_SECONDS", 90, lo=30, hi=900),
            max_resume_failures=_int(e, "SONYA_MAX_RESUME_FAILURES", 2, lo=1, hi=10),
            create_timeout_seconds=_int(e, "SONYA_CREATE_TIMEOUT_SECONDS", 900, lo=120, hi=3600),
            max_creates_per_hour=_int(e, "SONYA_MAX_CREATES_PER_HOUR", 6, lo=1, hi=60),
            pool_modes=tuple(m.strip() for m in e.get("SONYA_POOL_MODES", "virality,beep").split(",") if m.strip()),
            disk_gb=_int(e, "SONYA_DISK_GB", 80, lo=20, hi=500),
            max_dph=Decimal(e.get("SONYA_MAX_DPH", "0.60")),
            image=e.get("SONYA_WORKER_IMAGE", ""),
            release_sha=e.get("SONYA_RELEASE_SHA", ""),
            release_repo=e.get("SONYA_RELEASE_REPO", "samnesvoj/sonya-video-worker-production"),
        )
        if any(m.startswith("trailer") for m in cfg.pool_modes):
            raise ValueError("SONYA_POOL_MODES must not contain Trailer modes")
        return cfg
