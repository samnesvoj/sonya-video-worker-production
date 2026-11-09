"""Offer selection and the create-instance payload (central side). Secrets are assembled here, from the dispatcher's
own environment, and exist only in the create call for that one instance - never in a Template, image, log or Git."""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from dispatcher.config import FleetConfig

SECRET_PASSTHROUGH = ("BACKEND_API_URL", "WORKER_SECRET", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
                      "S3_BUCKET_NAME", "S3_BUCKET", "S3_REGION", "MODELS_S3_BUCKET", "MODELS_S3_PREFIX",
                      # optional: the Cut LLM editor (Gemini via Timeweb, OpenRouter fallback); without them the
                      # worker keeps the local ranking (scripts/shared/cut_profile.py)
                      "TIMEWEB_GEMINI_API_KEY", "TIMEWEB_GEMINI_BASE_URL", "TIMEWEB_GEMINI_MODEL",
                      "SONYA_LLM_TIMEWEB_ENABLED", "OPENROUTER_API_KEY", "SONYA_LLM_OPENROUTER_MODEL",
                      # load tests only (not secrets): resource sampler, clips rendered in parallel inside one job
                      "SONYA_RES_SAMPLER", "SONYA_CLIP_WORKERS")
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
            "gpu_name": {"in": list(cfg.default_gpus + cfg.fallback_gpus)},
            "gpu_ram": {"gte": cfg.min_vram_gb * 1000 - 500}, "disk_space": {"gte": cfg.disk_gb},
            "cpu_cores_effective": {"gte": cfg.min_cpu_cores}, "cpu_ram": {"gte": max(cfg.min_cpu_ram_gb, 4 * cfg.jobs_per_instance + 4) * 1000},   # ~3.5 GB RAM per parallel video
            "reliability2": {"gte": cfg.min_reliability}, "inet_down": {"gte": cfg.min_inet_down_mbps},
            "cuda_max_good": {"gte": 12.6}, "dph_total": {"lte": float(cfg.max_dph)}, "type": "on-demand",
            "order": [["dph_total", "asc"]], "limit": 200}


# ── server choice (owner 2026-11-08: «процессор под стать Mac mini, норм видеокарта, недорого и сразу с хорошим интернетом»;
#    decision: «бери RTX 3060 (12 ГБ) с процессором Ryzen 7 5800X») ────────────────────────────────────────────────────
# The Cut recipe is CPU-heavy (face tracking, ffmpeg filters, subtitles); the GPU does ASR + NVENC. So a modern fast CPU
# matters as much as the GPU. Vast's cpu_ghz is sometimes the base clock (5950X shows 3.4), so the CPU is judged by its
# FAMILY (all of these boost >= 4.0 GHz) + effective cores, not by cpu_ghz alone.
MODERN_CPU = re.compile(
    r"Ryzen\s+[3579]\s+[5789]\d{3}"          # Ryzen 5000 / 7000 / 8000 / 9000 desktop (5800X, 5700X3D, 5950X, 7950X3D ...)
    r"|Threadripper\s+(?:PRO\s+)?[579]\d{3}"  # Threadripper 5000 / 7000 / 9000
    r"|1[2-5]th Gen Core"                      # Intel 12th-15th gen (Vast: "12th Gen Core™ i7-12700")
    r"|Core™?\s+i[3579]-1[2-4]\d{3}"           # same, other spelling ("Core™ i7-14700")
    r"|Core™?\s+Ultra", re.I)
EUROPE = {"AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "UK", "GR", "HR", "HU",
          "IE", "IS", "IT", "LI", "LT", "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK",
          "UA", "XK"}
# Measured job time (s, claim -> done) for kishlak_15m virality in Europe, load test 2026-11-09 (SPEED_REPORT.md):
# A4000 + i7-14700 (ES) 103-108, 4060 Ti + 5950X (DE) 109; run 4 RTX 3090 (CPU encode) 179. 3060 assumed like the A4000.
# Only the RATIO matters: score = estimated job time x hourly price ("время обработки × цена").
JOB_SECONDS = {"RTX 3060": 110.0, "RTX A4000": 105.0, "RTX 4060 Ti": 109.0, "RTX 3090": 179.0}


def cpu_modern(o: Mapping[str, Any]) -> bool:
    return bool(MODERN_CPU.search(str(o.get("cpu_name") or "")))


def in_europe(o: Mapping[str, Any]) -> bool:
    geo = str(o.get("geolocation") or "")
    return geo.rsplit(",", 1)[-1].strip().upper() in EUROPE


def hourly_cost(o: Mapping[str, Any], cfg: FleetConfig) -> Decimal:
    """GPU + our disk per hour (stopped/running storage is billed too), not the headline dph_total."""
    return Decimal(str(o.get("dph_base", o.get("dph_total", 99)))) + Decimal(str(o.get("storage_cost", 0))) * cfg.disk_gb / Decimal(720)


def offer_class(o: Mapping[str, Any], cfg: FleetConfig) -> Optional[int]:
    """0 = default class (RTX 3060 + modern CPU), 1 = fallback GPU + modern CPU, None = not acceptable."""
    if not cpu_modern(o) or float(o.get("cpu_cores_effective") or 0) < cfg.min_cpu_cores:
        return None
    if o.get("gpu_name") in cfg.default_gpus:
        return 0
    if o.get("gpu_name") in cfg.fallback_gpus:
        return 1
    return None


def warm_eligible(o: Mapping[str, Any], cfg: FleetConfig) -> bool:
    """May this host be the warm (paused) server: default class IN EUROPE and cheap storage (pause cost = disk x storage).
    Europe is required: the same RTX 3060 in Vietnam took 260 s per video vs ~105 s in Europe (S3 is in Russia)."""
    cls = offer_class(o, cfg)
    return (cls == 0 or (cls == 1 and cfg.warm_fallback)) and in_europe(o) and Decimal(str(o.get("storage_cost") or 0)) <= cfg.max_storage_gb_month


def offer_rank(o: Mapping[str, Any], cfg: FleetConfig, known_machines: Iterable[int] = ()) -> Tuple:
    est = JOB_SECONDS.get(str(o.get("gpu_name")), JOB_SECONDS["RTX 3090"])
    # Europe FIRST (load test 2026-11-09): our S3 + API are in Russia; from Vietnam every upload/status call cost seconds
    # (260 s per video, 2.5x Europe) and the image pull took 289 s vs 117 s in Germany. Then the class, then the cache.
    return (not in_europe(o),
            offer_class(o, cfg),
            o.get("machine_id") not in set(known_machines),   # booted there before -> the image layers are likely cached
            Decimal(str(est)) * hourly_cost(o, cfg),
            -float(o.get("inet_down") or 0))


def pick_offer(offers: List[dict], cfg: FleetConfig, banned_machines: set, known_machines: Iterable[int] = (),
               *, warm: bool = False, exclude: Iterable[int] = ()) -> Optional[dict]:
    """Best acceptable offer: default class first, then (cached image, Europe, time x price, faster internet).
    warm=True: only hosts that may hold the paused warm server (default class, storage <= max_storage_gb_month)."""
    skip = set(exclude)
    ok = [o for o in offers if offer_class(o, cfg) is not None and o.get("machine_id") not in banned_machines
          and o.get("id") not in skip and o.get("machine_id") not in skip
          and Decimal(str(o.get("dph_total", 99))) <= cfg.max_dph
          and float(o.get("inet_down") or 0) >= cfg.min_inet_down_mbps
          and float(o.get("reliability2", o.get("reliability", 1)) or 0) >= cfg.min_reliability
          and (not warm or warm_eligible(o, cfg))]
    known = set(known_machines)
    return min(ok, key=lambda o: offer_rank(o, cfg, known)) if ok else None


def create_payload(cfg: FleetConfig, label: str, github_token: str, environ: Mapping[str, str]) -> Dict[str, Any]:
    validate_release(cfg)
    if not github_token:
        raise LaunchError("no GitHub release token")
    env = {"SONYA_RELEASE_SHA": cfg.release_sha, "SONYA_RELEASE_REPO": cfg.release_repo, "GITHUB_RELEASE_TOKEN": github_token,
           "WORKER_ID": label, "WORKER_BACKEND_MODE": "api", "ENABLED_MODES": ",".join(cfg.pool_modes),
           "SONYA_REQUIRE_MODEL_LOCK": "1", "SONYA_MIN_VRAM_MB": str(cfg.min_vram_gb * 1000 - 4000),
           "SONYA_WORKER_CONCURRENCY": str(cfg.jobs_per_instance),
           # NVENC: the NVIDIA runtime mounts libnvidia-encode only with the "video" capability; without it the worker
           # fell back to libx264 on Vast (run 4: "render encoder: cpu @ 6M", 179 s vs 108 s on the Mac mini)
           "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video"}
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
