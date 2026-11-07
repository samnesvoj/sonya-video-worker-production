#!/usr/bin/env bash
# Runs from the VERIFIED release: preflight -> readiness gate -> health -> gpu_worker --poll (graceful SIGTERM).
set -Eeuo pipefail
umask 077
log() { printf '[run_worker] %s\n' "$*" >&2; }

REL="${SONYA_RELEASE_DIR:?start via the base-image entrypoint}"
STATE_DIR="${SONYA_STATE_DIR:-/run/sonya}"
BOOT="${SONYA_BOOTSTRAP_DIR:-/opt/sonya-bootstrap}"
export SONYA_READY_FILE="${SONYA_READY_FILE:-${STATE_DIR}/ready}"
export SONYA_WORKER_PID_FILE="${STATE_DIR}/worker.pid"
export WORKER_BACKEND_MODE="${WORKER_BACKEND_MODE:-api}"
export MODELS_LOCAL_DIR="${MODELS_LOCAL_DIR:-/opt/sonya/models}"
export TMPDIR="${TMPDIR:-/work/tmp}"
export WORKER_POLL_INTERVAL="${WORKER_POLL_INTERVAL:-${WORKER_IDLE_SLEEP_SEC:-15}}"
export SONYA_REQUIRE_READY=1
export WORKER_ID="${WORKER_ID:-vast-${VAST_CONTAINERLABEL:-${HOSTNAME:-worker}}}"

mkdir -p "${TMPDIR}" "${MODELS_LOCAL_DIR}"
rm -f "${SONYA_READY_FILE}" "${SONYA_WORKER_PID_FILE}"
wipe() { find "${TMPDIR}" -mindepth 1 -maxdepth 1 -name 'sonya_*' -exec rm -rf {} + 2>/dev/null || true; }
wipe   # never keep user video from a crashed previous run

cd "${REL}"
log "preflight (release $(python3 -c 'import json;print(json.load(open("release/manifest.json"))["version"])'))"
python3 scripts/prod_preflight_check.py --role worker >/dev/null || { log "preflight failed"; exit 3; }
python3 -m scripts.ops.readiness || { log "readiness failed - worker NOT started"; exit 4; }

python3 "${BOOT}/healthd.py" &
HEALTH_PID=$!
python3 scripts/gpu_worker.py --poll --worker-id "${WORKER_ID}" &
WORKER_PID=$!
echo "${WORKER_PID}" > "${SONYA_WORKER_PID_FILE}"
log "worker pid=${WORKER_PID}; health on 127.0.0.1:${SONYA_HEALTH_PORT:-8099}"

_term() {
  log "SIGTERM: worker requeues its job and exits"
  kill -TERM "${WORKER_PID}" 2>/dev/null || true
  wait "${WORKER_PID}" 2>/dev/null || true
  kill "${HEALTH_PID}" 2>/dev/null || true
  rm -f "${SONYA_READY_FILE}"; wipe
  exit 143
}
trap _term TERM INT
set +e; wait "${WORKER_PID}"; RC=$?; set -e
kill "${HEALTH_PID}" 2>/dev/null || true
rm -f "${SONYA_READY_FILE}"
log "worker exited rc=${RC}"
exit "${RC}"
