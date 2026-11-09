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
    max_instances: int = 3                   # owner 2026-11-08: 1 warm + extra instances created per load (load test, SPEED_REPORT.md)
    jobs_per_instance: int = 2               # = SONYA_WORKER_CONCURRENCY in the worker (N worker processes, one job each)
    # Warm server (owner decision 2026-11-08): exactly `warm_instances` STOPPED instance(s) of the default class are kept
    # (image + release + models on disk, resume ~10-40 s). Every other instance is a BURST one: created for the load and
    # DESTROYED (not stopped) when idle. A warm that cannot resume (host busy) is destroyed and the next instance becomes warm.
    warm_instances: int = 1                  # SONYA_WARM_INSTANCES (0 = old behaviour: every instance is burst-like)
    warm_prime: bool = True                  # SONYA_WARM_PRIME - no warm at all (first start / lost / stale) -> create one, let it boot, stop it
    stopped_retention_seconds: int = 1800    # a stopped NON-warm instance older than this -> DESTROY (storage keeps billing)
    resume_timeout_seconds: int = 90         # 'scheduling' longer than this -> GPU likely taken (Vast docs: ~30 s)
    max_resume_failures: int = 1             # one failed wake-up of the warm -> replace it (the user should not wait twice)
    create_timeout_seconds: int = 900        # create -> running must happen within this
    host_error_grace_seconds: int = 60       # Vast reports a docker/GPU start error this long -> drop the host now
    stall_seconds: int = 900                 # jobs wait, worker is up but claims nothing for this long -> replace the instance
    max_creates_per_hour: int = 6            # runaway-cost guard
    max_running_hours: int = 24              # hard cap for a single continuous run
    pool_modes: Tuple[str, ...] = ("virality", "beep")
    label_prefix: str = "sonya-pool-"
    # Offer filter (owner decision 2026-11-08: «RTX 3060 (12 ГБ) с процессором Ryzen 7 5800X» = the default class;
    # see dispatcher/launcher.py pick_offer for the full rule)
    min_vram_gb: int = 12
    disk_gb: int = 40                        # image (~11 GB unpacked) + models (0.6 GB) + temp for 2-4 parallel jobs; SPEED_REPORT.md
    max_dph: Decimal = Decimal("0.18")       # USD/h price cap per offer (GPU + our disk), fallback classes included
    default_gpus: Tuple[str, ...] = ("RTX 3060",)
    fallback_gpus: Tuple[str, ...] = ("RTX 4060 Ti", "RTX 4070", "RTX A4000", "RTX 3080", "RTX 3080 Ti", "RTX 4070 Ti",
                                      "RTX 4070 SUPER", "RTX 3090", "RTX 5060 Ti", "RTX 5070")
    min_cpu_cores: float = 8.0               # effective cores of OUR slice (Vast cpu_cores_effective)
    min_cpu_ram_gb: int = 16
    max_storage_gb_month: Decimal = Decimal("0.25")  # USD per GB-month: the warm's pause cost = disk_gb x this
    min_reliability: float = 0.98
    min_inet_down_mbps: int = 800
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
            max_instances=_int(e, "SONYA_FLEET_MAX_INSTANCES", 3, lo=1, hi=8),
            jobs_per_instance=_int(e, "SONYA_JOBS_PER_INSTANCE", 2, lo=1, hi=4),
            warm_instances=_int(e, "SONYA_WARM_INSTANCES", 1, lo=0, hi=1),
            warm_prime=e.get("SONYA_WARM_PRIME", "true").lower() in TRUE,
            stopped_retention_seconds=_int(e, "SONYA_STOPPED_RETENTION_SECONDS", 1800, lo=0, hi=7 * 86400),
            resume_timeout_seconds=_int(e, "SONYA_RESUME_TIMEOUT_SECONDS", 90, lo=30, hi=900),
            max_resume_failures=_int(e, "SONYA_MAX_RESUME_FAILURES", 1, lo=1, hi=10),
            create_timeout_seconds=_int(e, "SONYA_CREATE_TIMEOUT_SECONDS", 900, lo=120, hi=3600),
            host_error_grace_seconds=_int(e, "SONYA_HOST_ERROR_GRACE_SECONDS", 60, lo=20, hi=900),
            stall_seconds=_int(e, "SONYA_STALL_SECONDS", 900, lo=300, hi=7200),
            max_creates_per_hour=_int(e, "SONYA_MAX_CREATES_PER_HOUR", 6, lo=1, hi=60),
            pool_modes=tuple(m.strip() for m in e.get("SONYA_POOL_MODES", "virality,beep").split(",") if m.strip()),
            disk_gb=_int(e, "SONYA_DISK_GB", 40, lo=20, hi=500),
            max_dph=Decimal(e.get("SONYA_MAX_DPH", "0.18")),
            max_storage_gb_month=Decimal(e.get("SONYA_MAX_STORAGE_GB_MONTH", "0.25")),
            min_inet_down_mbps=_int(e, "SONYA_MIN_INET_DOWN_MBPS", 800, lo=100, hi=100000),
            image=e.get("SONYA_WORKER_IMAGE", ""),
            release_sha=e.get("SONYA_RELEASE_SHA", ""),
            release_repo=e.get("SONYA_RELEASE_REPO", "samnesvoj/sonya-video-worker-production"),
        )
        if any(m.startswith("trailer") for m in cfg.pool_modes):
            raise ValueError("SONYA_POOL_MODES must not contain Trailer modes")
        return cfg
