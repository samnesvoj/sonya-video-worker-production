#!/usr/bin/env bash
# worker_entrypoint.sh — SONYA Docker GPU worker entrypoint
#
# Runs inside the container on a vast.ai ephemeral GPU instance.
#
# Production path:
#   VPS dispatcher → vast.ai direct image → THIS entrypoint
#   → prod_preflight_check → model_downloader → gpu_worker
#   → backend worker API (BACKEND_API_URL) → S3 → shutdown/destroy
#
# Secrets are injected by vast.ai via the `env` dict in the create payload
# (sent over HTTPS to vast.ai API, never embedded in a startup script).
# DATABASE_URL is NOT required — WORKER_BACKEND_MODE=api.
#
# Required env vars:
#   BACKEND_API_URL     https://sonya-e.com/api/worker
#   WORKER_SECRET       HMAC secret for worker API calls
#   S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
#   S3_BUCKET_NAME, S3_REGION, MODELS_S3_BUCKET
#   JOB_ID              UUID of the job to process — REQUIRED ONLY when
#                        WORKER_LOOP is not "true" (single-job mode)
#   MODE                mode name (default: trailer_film_breaker) — only used
#                        in single-job mode; ignored when WORKER_LOOP=true
#                        (mode is decided per claimed job)
#
# Worker mode (WORKER_LOOP env var):
#   WORKER_LOOP=false (default) — single-job / ephemeral mode. Vast launches
#     one instance per job with JOB_ID set; the entrypoint downloads models
#     for MODE, runs gpu_worker.py --once --job-id, then the container exits
#     and Vast destroys/recycles the instance.
#   WORKER_LOOP=true — persistent mode. Vast (or any host) starts the
#     container once with NO JOB_ID; the entrypoint skips model_downloader
#     (mode varies per job) and execs gpu_worker.py --poll, which claims jobs
#     from the backend queue itself and downloads models per claimed job.
#     Extra vars for this mode:
#       WORKER_IDLE_SLEEP_SEC  seconds between poll attempts when idle
#                              (default: 15; bridged to gpu_worker.py's own
#                              WORKER_POLL_INTERVAL)
#       WORKER_ID              worker identifier reported to the backend
#                              (default: gpu-persistent-$HOSTNAME)
#
# Debug-safe mode (diagnosing Vast "Retrying in 1 second" loops):
#   VAST_DEBUG_SLEEP_ON_FAIL=true → on any failure, print diagnostics and
#   sleep 900s instead of exiting immediately, so the log can be read on
#   the vast.ai console before the instance is destroyed/retried.
#   Default: false (production — exits immediately with the original code).
#   Turn OFF again once the failure has been diagnosed.
#
# Early backend heartbeat (worker_status()):
#   Posts POST /api/worker/jobs/{JOB_ID}/status to BACKEND_API_URL at each
#   stage (worker_started → preflight_running → model_downloading →
#   mode_running) so the backend sees activity even if preflight or model
#   download hangs/fails before gpu_worker.py makes its own first call.
#   Best-effort only: failures are logged and swallowed (`|| true`), never
#   fail the entrypoint, and never print secret values. It is job-scoped:
#   when JOB_ID is empty (persistent mode, before a job is claimed) it logs
#   a skip line instead of calling the API.

set -uo pipefail

LOG_DIR="/var/log/sonya"
LOG_FILE="${LOG_DIR}/gpu_worker_container.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

VAST_DEBUG_SLEEP_ON_FAIL="${VAST_DEBUG_SLEEP_ON_FAIL:-false}"

# ── Worker mode config ─────────────────────────────────────────────────────────
WORKER_LOOP="${WORKER_LOOP:-false}"
WORKER_IDLE_SLEEP_SEC="${WORKER_IDLE_SLEEP_SEC:-15}"
WORKER_ID="${WORKER_ID:-gpu-persistent-${HOSTNAME:-worker}}"
# Bridge to gpu_worker.py's own poll-interval env var so WORKER_IDLE_SLEEP_SEC
# actually controls the idle sleep between poll attempts in --poll mode.
export WORKER_POLL_INTERVAL="${WORKER_POLL_INTERVAL:-${WORKER_IDLE_SLEEP_SEC}}"

_present() { [[ -n "${1:-}" ]] && echo "yes" || echo "no"; }
_pyver()   { python --version 2>&1 || python3 --version 2>&1 || echo "python not found"; }

# ── Early startup banner — FIRST lines printed, before any validation/logic ───
# so the banner is captured even if the container dies almost instantly.
echo "=== SONYA GPU worker container start ==="
date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date
pwd
whoami 2>/dev/null || id -un 2>/dev/null || echo "unknown"
_pyver
echo "WORKER_BACKEND_MODE=${WORKER_BACKEND_MODE:-api}"
echo "BACKEND_API_URL=${BACKEND_API_URL:-<not set>}"
echo "JOB_ID=${JOB_ID:-<not set>}"
echo "WORKER_LOOP=${WORKER_LOOP}"
echo "WORKER_ID=${WORKER_ID}"
echo "S3_BUCKET present: $(_present "${S3_BUCKET:-}")"
echo "S3_BUCKET_NAME present: $(_present "${S3_BUCKET_NAME:-}")"
echo "WORKER_SECRET present: $(_present "${WORKER_SECRET:-}")"
echo "VAST_DEBUG_SLEEP_ON_FAIL=${VAST_DEBUG_SLEEP_ON_FAIL}"
echo "=== end startup banner ==="

# ── Error trap — runs on ANY failing command from this point on ───────────────
# Prints diagnostics (never secret values — only env var NAMES) and, when
# VAST_DEBUG_SLEEP_ON_FAIL=true, sleeps instead of exiting so a human can
# read the log before Vast retries/destroys the instance.
_on_error() {
    local exit_code=$?
    local line_no="${1:-?}"
    echo "[ENTRYPOINT_ERROR] line=${line_no} exit_code=${exit_code}"
    date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date
    pwd
    whoami 2>/dev/null || id -un 2>/dev/null || echo "unknown"
    _pyver
    echo "[ENTRYPOINT_ERROR] env var names present (values are NEVER printed):"
    env | cut -d= -f1 | sort
    if [[ "${VAST_DEBUG_SLEEP_ON_FAIL,,}" == "true" ]]; then
        echo "[ENTRYPOINT_ERROR] VAST_DEBUG_SLEEP_ON_FAIL=true — sleeping 900s so the log can be inspected before the instance is retried/destroyed."
        sleep 900
    fi
    exit "${exit_code}"
}
trap '_on_error "${LINENO}"' ERR

set -e

ts()   { date '+%Y-%m-%dT%H:%M:%S%z'; }
log()  { echo "[$(ts)] [INFO]  $*"; }
warn() { echo "[$(ts)] [WARN]  $*"; }
fail() { echo "[$(ts)] [ERROR] $*"; exit 1; }

log "=== SONYA GPU worker container start (detailed) ==="
log "JOB_ID=${JOB_ID:-<not set>}"
log "MODE=${MODE:-trailer_film_breaker}"
log "BACKEND_API_URL=${BACKEND_API_URL:-<not set>}"
log "WORKER_BACKEND_MODE=${WORKER_BACKEND_MODE:-api}"
log "WORKER_LOOP=${WORKER_LOOP} WORKER_ID=${WORKER_ID} WORKER_IDLE_SLEEP_SEC=${WORKER_IDLE_SLEEP_SEC}"

# ── Required env validation ────────────────────────────────────────────────────
# JOB_ID is only required in single-job mode (WORKER_LOOP != true). Persistent
# workers (WORKER_LOOP=true) have no JOB_ID at container startup — they claim
# jobs themselves via gpu_worker.py --poll. All other vars are always required.
if [[ "${WORKER_LOOP,,}" != "true" ]]; then
    : "${JOB_ID:?JOB_ID env var is required when WORKER_LOOP is not true}"
fi
JOB_ID="${JOB_ID:-}"
: "${BACKEND_API_URL:?BACKEND_API_URL env var is required}"
: "${WORKER_SECRET:?WORKER_SECRET env var is required}"
: "${S3_ENDPOINT_URL:?S3_ENDPOINT_URL env var is required}"
: "${S3_ACCESS_KEY_ID:?S3_ACCESS_KEY_ID env var is required}"
: "${S3_SECRET_ACCESS_KEY:?S3_SECRET_ACCESS_KEY env var is required}"
: "${S3_BUCKET_NAME:?S3_BUCKET_NAME env var is required}"
: "${S3_REGION:?S3_REGION env var is required}"
: "${MODELS_S3_BUCKET:?MODELS_S3_BUCKET env var is required}"

# ── Early backend heartbeat ────────────────────────────────────────────────────
# Posts a best-effort status update to the backend worker API as soon as the
# entrypoint reaches each stage. This is diagnostic only: if preflight or
# model download hangs/fails before gpu_worker.py ever starts, the backend
# still sees SOMETHING (previously it saw nothing until gpu_worker.py itself
# made its first call). Never fails the entrypoint (`|| true` + internal
# try/except) and never prints secret values — only the HTTP status code and
# the status string being reported.
worker_status() {
    local status="$1"
    local progress="${2:-0}"
    local message="${3:-}"
    if [[ -z "${JOB_ID:-}" ]]; then
        echo "[worker_entrypoint] status_callback_skip reason=no_job_id status=${status}"
        return 0
    fi
    python - <<PY || true
import json, os, urllib.request
base = os.environ["BACKEND_API_URL"].rstrip("/")
job_id = os.environ["JOB_ID"]
secret = os.environ["WORKER_SECRET"]
payload = json.dumps({
    "status": "$status",
    "progress": int("$progress"),
    "message": "$message",
}).encode("utf-8")
req = urllib.request.Request(
    f"{base}/api/worker/jobs/{job_id}/status",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secret}",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print("[worker_entrypoint] status_callback", r.status, "$status")
except Exception as e:
    print("[worker_entrypoint] status_callback_failed", "$status", repr(e))
PY
}

worker_status "worker_started" 1 "entrypoint started"

MODE="${MODE:-trailer_film_breaker}"
WORKER_BACKEND_MODE="${WORKER_BACKEND_MODE:-api}"
WORKDIR="/opt/sonya"

# ── S3 bucket alias (S3_BUCKET and S3_BUCKET_NAME are interchangeable) ────────
# The Vast payload sends S3_BUCKET_NAME; some internal tools expect S3_BUCKET.
# Keep both in sync so preflight and workers see the same value.
S3_BUCKET_NAME="${S3_BUCKET_NAME:-${S3_BUCKET:-}}"
S3_BUCKET="${S3_BUCKET:-${S3_BUCKET_NAME:-}}"
MODELS_S3_BUCKET="${MODELS_S3_BUCKET:-${S3_BUCKET_NAME:-}}"

# ── Write .env.local (worker reads this for runtime config) ───────────────────
ENV_LOCAL="${WORKDIR}/.env.local"
log "Writing ${ENV_LOCAL}..."
{
    echo "WORKER_BACKEND_MODE=${WORKER_BACKEND_MODE}"
    echo "BACKEND_API_URL=${BACKEND_API_URL}"
    echo "S3_ENDPOINT_URL=${S3_ENDPOINT_URL}"
    echo "S3_ACCESS_KEY_ID=${S3_ACCESS_KEY_ID}"
    echo "S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}"
    echo "S3_BUCKET_NAME=${S3_BUCKET_NAME}"
    echo "S3_BUCKET=${S3_BUCKET}"            # alias for tools that use S3_BUCKET
    echo "S3_REGION=${S3_REGION}"
    echo "MODELS_S3_BUCKET=${MODELS_S3_BUCKET}"
    echo "WORKER_SECRET=${WORKER_SECRET}"
    echo "WORKER_ID=${WORKER_ID}"
    echo "AUTO_GPU_TRIGGER_ENABLED=false"
    # Optional keys (injected only when set)
    [[ -n "${OPENROUTER_API_KEY:-}" ]]  && echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}"
    [[ -n "${GEMINI_API_KEY:-}" ]]       && echo "GEMINI_API_KEY=${GEMINI_API_KEY}"
    [[ -n "${ELEVENLABS_API_KEY:-}" ]]   && echo "ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}"
    [[ -n "${ELEVENLABS_VOICE_ID:-}" ]]  && echo "ELEVENLABS_VOICE_ID=${ELEVENLABS_VOICE_ID}"
    # DATABASE_URL intentionally omitted — api mode does not need it
} > "${ENV_LOCAL}"
chmod 600 "${ENV_LOCAL}"
log ".env.local written (mode 600)."

# ── Pre-flight check ───────────────────────────────────────────────────────────
worker_status "preflight_running" 5 "preflight started"
log "Running pre-flight check (worker role)..."
python "${WORKDIR}/scripts/prod_preflight_check.py" --role worker \
    || fail "Pre-flight check failed."
log "Pre-flight check passed."

# ── Reproducibility manifest (best-effort, never fails the worker) ─────────────
# commit / image digest / Python, Torch, CUDA, FFmpeg, OpenCV / deps hash /
# stage versions / model file ids / non-secret env flags.
MANIFEST_FILE="${SONYA_MANIFEST_FILE:-/tmp/sonya_worker_manifest.json}"
if python -m scripts.shared.worker_manifest --out "${MANIFEST_FILE}" >/dev/null 2>&1; then
    log "Worker manifest written: ${MANIFEST_FILE} (commit=${SONYA_GIT_SHA:-unknown}, image=${SONYA_IMAGE:-unset})"
else
    log "Worker manifest skipped (non-fatal)."
fi

if [[ "${WORKER_LOOP,,}" == "true" ]]; then
    # ── Persistent poll mode ─────────────────────────────────────────────────
    # Mode is only known once a job is claimed, so model_downloader must NOT
    # run here — gpu_worker.py already downloads models per claimed job
    # (ensure_models_for_mode) before running each mode, exactly as in
    # --once mode. No JOB_ID exists yet, so worker_status() above only ever
    # logged a status_callback_skip for this instance until now.
    log "WORKER_LOOP=true — starting persistent poll mode (worker_id=${WORKER_ID}, idle_sleep=${WORKER_IDLE_SLEEP_SEC}s)..."
    exec python "${WORKDIR}/scripts/gpu_worker.py" --poll --worker-id "${WORKER_ID}"
else
    # ── Single-job mode (legacy Vast direct-image launch) ────────────────────
    : "${JOB_ID:?JOB_ID env var is required when WORKER_LOOP is not true}"

    # ── Model download ───────────────────────────────────────────────────────
    worker_status "model_downloading" 15 "model download started"
    log "Downloading models for MODE=${MODE}..."
    python "${WORKDIR}/scripts/model_downloader.py" --mode "${MODE}" \
        || fail "Model download failed for MODE=${MODE}."
    log "Models ready."

    # ── Run worker (exactly once) ────────────────────────────────────────────
    worker_status "mode_running" 30 "gpu_worker starting"
    log "Starting gpu_worker.py --once --job-id ${JOB_ID}..."
    exec python "${WORKDIR}/scripts/gpu_worker.py" \
        --once \
        --job-id "${JOB_ID}"
fi
