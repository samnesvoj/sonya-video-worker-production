"""
prod_preflight_check.py
=======================
Pre-flight environment check before starting the API or worker.
Exits with code 1 if any required env var is missing.

Usage
-----
  python prod_preflight_check.py --role worker    (recommended)
  python prod_preflight_check.py --role backend
  python prod_preflight_check.py worker           (legacy positional)
  python prod_preflight_check.py backend          (legacy positional)

Worker api-mode (WORKER_BACKEND_MODE=api)
-----------------------------------------
External GPU workers (vast.ai) communicate via BACKEND_API_URL, not PostgreSQL.
DATABASE_URL is NOT checked — the instance cannot reach the private DB server.

S3 bucket aliasing
------------------
S3_BUCKET and S3_BUCKET_NAME are treated as aliases.  Either one satisfies
the bucket requirement — the check passes if at least one is set.
"""
from __future__ import annotations

import os
import sys

# ── Required env var groups ────────────────────────────────────────────────────

# Backend (VPS API server) — requires DATABASE_URL
# NOTE: SMTP_* vars are intentionally NOT required here — the API must still
# boot without them; /api/auth/request-code fails clearly with 503
# email_not_configured at request time instead (see scripts/email_sender.py).
REQUIRED_BACKEND: list[str] = [
    "DATABASE_URL",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "WORKER_SECRET",
    "AUTH_SECRET",
    "CORS_ORIGINS",
]

# Worker in db-mode (legacy, VPS-internal worker with DB access)
REQUIRED_WORKER_DB: list[str] = [
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
]

# Worker in api-mode (external GPU, e.g. vast.ai) — NO DATABASE_URL
REQUIRED_WORKER_API: list[str] = [
    "WORKER_SECRET",
    "BACKEND_API_URL",
    "S3_ENDPOINT_URL",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _s3_bucket_present() -> bool:
    """Accept either S3_BUCKET or S3_BUCKET_NAME (they are aliases)."""
    return bool(os.environ.get("S3_BUCKET") or os.environ.get("S3_BUCKET_NAME"))


def _models_bucket_present() -> bool:
    """MODELS_S3_BUCKET falls back to S3_BUCKET_NAME if not set."""
    return bool(
        os.environ.get("MODELS_S3_BUCKET")
        or os.environ.get("S3_BUCKET_NAME")
        or os.environ.get("S3_BUCKET")
    )


def check(var_list: list[str], label: str) -> bool:
    missing = [v for v in var_list if not os.environ.get(v)]
    if missing:
        print(f"[PREFLIGHT] {label} — MISSING env vars: {missing}", file=sys.stderr)
        return False
    print(f"[PREFLIGHT] {label} — OK")
    return True


def _parse_role() -> str:
    """
    Parse --role ROLE (preferred) or positional ROLE (legacy).
    Returns 'worker', 'backend', or 'backend' (default).
    """
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--role" and i + 1 < len(args):
            return args[i + 1]
        if arg in ("backend", "worker"):
            return arg
    return "backend"


def _detector_preflight() -> None:
    """Face detector (OpenCV Haar) check. Never fails the worker: Cut keeps
    working with a centered crop, but the problem is logged loudly."""
    try:
        from scripts.shared.crop.detector_check import face_detector_preflight
    except Exception as exc:  # pragma: no cover - import path problems
        print(f"[PREFLIGHT] worker face detector — check unavailable: {exc}", file=sys.stderr)
        return
    res = face_detector_preflight()
    if res["ok"]:
        print(f"[PREFLIGHT] worker face detector — OK (OpenCV {res['opencv_version']}, Haar cascade loaded)")
    else:
        print(f"[PREFLIGHT] worker face detector — UNAVAILABLE: {res['reason']} "
              "(crops fall back to center)", file=sys.stderr)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    role        = _parse_role()
    worker_mode = os.environ.get("WORKER_BACKEND_MODE", "db").lower()
    ok          = True

    if role == "worker":
        if worker_mode == "api":
            # ── External GPU worker (vast.ai) — api-only mode ─────────────────
            # DATABASE_URL is NOT required — instance cannot reach private Postgres.
            # All job ops go through BACKEND_API_URL with WORKER_SECRET.
            print("[PREFLIGHT] worker-api mode (WORKER_BACKEND_MODE=api) — no DATABASE_URL required")
            ok = check(REQUIRED_WORKER_API, "worker-api")

            if not _s3_bucket_present():
                print(
                    "[PREFLIGHT] worker-api — MISSING env vars: [S3_BUCKET or S3_BUCKET_NAME]",
                    file=sys.stderr,
                )
                ok = False
            else:
                print("[PREFLIGHT] worker-api S3 bucket (S3_BUCKET / S3_BUCKET_NAME) — OK")

            if not _models_bucket_present():
                print(
                    "[PREFLIGHT] worker-api — MISSING env vars: [MODELS_S3_BUCKET or S3_BUCKET_NAME]",
                    file=sys.stderr,
                )
                ok = False
            else:
                print("[PREFLIGHT] worker-api MODELS_S3_BUCKET (or S3_BUCKET_NAME fallback) — OK")
        else:
            # ── VPS-internal db-mode worker ───────────────────────────────────
            ok = check(REQUIRED_WORKER_DB, "worker-db")
            if not _s3_bucket_present():
                print(
                    "[PREFLIGHT] worker-db — MISSING env vars: [S3_BUCKET or S3_BUCKET_NAME]",
                    file=sys.stderr,
                )
                ok = False
            else:
                print("[PREFLIGHT] worker-db S3 bucket — OK")
        _detector_preflight()
    else:
        # ── Backend API server ────────────────────────────────────────────────
        ok = check(REQUIRED_BACKEND, "backend")
        if not _s3_bucket_present():
            print(
                "[PREFLIGHT] backend — MISSING env vars: [S3_BUCKET or S3_BUCKET_NAME]",
                file=sys.stderr,
            )
            ok = False
        else:
            print("[PREFLIGHT] backend S3 bucket — OK")

        # Paid plans have a max source length, measured with ffprobe on the
        # API server (scripts/entitlements.py). Without it every paid
        # upload is refused (503) -- free uploads still work.
        import shutil
        if shutil.which(os.environ.get("FFPROBE_BIN", "ffprobe")):
            print("[PREFLIGHT] backend ffprobe — OK")
        else:
            print("[PREFLIGHT] backend — MISSING binary: ffprobe (apt-get install -y ffmpeg); "
                  "paid-plan jobs would be refused", file=sys.stderr)
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
