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
export WORKER_POLL_INTERVAL="${WORKER_POLL_INTERVAL:-${WORKER_IDLE_SLEEP_SEC:-5}}"   # 5 s: a job arriving while the worker is idle waits <= 5 s (was 15 s)
export SONYA_REQUIRE_READY=1
# ASR (owner decision 2026-11-07): GigaAM v3 primary, faster-whisper fallback; models come ONLY from our S3 (sha256-verified), never downloaded.
export SONYA_ASR_BACKEND="${SONYA_ASR_BACKEND:-gigaam}"
export SONYA_ASR_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # not overridable from the Template: no runtime downloads
export SONYA_REQUIRE_ASR="${SONYA_REQUIRE_ASR:-1}"
# Owner decisions 2026-11-08 for the first Vast test (match the Mac mini reference): no Gemini layout, no webcam detector unless set explicitly.
export GEMINI_LAYOUT_ENABLED="${GEMINI_LAYOUT_ENABLED:-false}"
export WEBCAM_DETECTOR_ENABLED="${WEBCAM_DETECTOR_ENABLED:-false}"
export WORKER_ID="${WORKER_ID:-vast-${VAST_CONTAINERLABEL:-${HOSTNAME:-worker}}}"

CONC="${SONYA_WORKER_CONCURRENCY:-1}"
case "${CONC}" in 1|2|3|4) ;; *) log "SONYA_WORKER_CONCURRENCY must be 1..4 (got ${CONC})"; exit 3 ;; esac
mkdir -p "${TMPDIR}" "${MODELS_LOCAL_DIR}"
rm -f "${SONYA_READY_FILE}" "${SONYA_WORKER_PID_FILE}"
wipe() { find "${TMPDIR}" -mindepth 1 -maxdepth 1 -name 'sonya_*' -exec rm -rf {} + 2>/dev/null || true; }
wipe   # never keep user video from a crashed previous run

cd "${REL}"
export SONYA_TIMELINE_FILE="${SONYA_TIMELINE_FILE:-${STATE_DIR}/timeline.jsonl}"
mark() { python3 -m scripts.ops.timeline mark "$1" || true; }
mark run_worker_start
log "preflight (release $(python3 -c 'import json;print(json.load(open("release/manifest.json"))["version"])'))"
python3 scripts/prod_preflight_check.py --role worker >/dev/null || { log "preflight failed"; exit 3; }
mark preflight_done
python3 -m scripts.ops.readiness || { log "readiness failed - worker NOT started"; python3 -m scripts.ops.timeline upload || true; exit 4; }

python3 "${BOOT}/healthd.py" &
HEALTH_PID=$!
# Load tests only (SONYA_RES_SAMPLER=1): VRAM/RAM/CPU samples -> S3 fast-start-test/res/ (scripts/ops/res_sampler.py)
[ "${SONYA_RES_SAMPLER:-0}" = "1" ] && { python3 -m scripts.ops.res_sampler & }
# Parallel jobs on one server (dispatcher jobs_per_instance): N independent worker PROCESSES under the same WORKER_ID,
# each claims and runs ONE job at a time (own memory, own CUDA context; no shared state between jobs). The production
# claim API has no per-worker limit, so N processes = N jobs. Default 1 = exactly the old behaviour.
WORKER_PIDS=()
for n in $(seq 1 "${CONC}"); do
  SONYA_WORKER_SLOT="${n}" python3 scripts/gpu_worker.py --poll --worker-id "${WORKER_ID}" &
  WORKER_PIDS+=("$!")
  [ "${n}" -lt "${CONC}" ] && sleep 2       # stagger the prewarm / CUDA init
done
WORKER_PID="${WORKER_PIDS[0]}"
echo "${WORKER_PID}" > "${SONYA_WORKER_PID_FILE}"
mark worker_started
(python3 -m scripts.ops.timeline upload || true) &   # best-effort: lets the central dispatcher see the start-up breakdown
log "worker pids=${WORKER_PIDS[*]} (concurrency ${CONC}); health on 127.0.0.1:${SONYA_HEALTH_PORT:-8099}"

_term() {
  log "SIGTERM: workers requeue their jobs and exit"
  kill -TERM "${WORKER_PIDS[@]}" 2>/dev/null || true
  wait "${WORKER_PIDS[@]}" 2>/dev/null || true
  kill "${HEALTH_PID}" 2>/dev/null || true
  rm -f "${SONYA_READY_FILE}"; wipe
  exit 143
}
trap _term TERM INT
# One worker process dying = the server is unhealthy: stop the others (they requeue their jobs) and exit.
set +e; wait -n "${WORKER_PIDS[@]}"; RC=$?; set -e
kill -TERM "${WORKER_PIDS[@]}" 2>/dev/null || true
wait "${WORKER_PIDS[@]}" 2>/dev/null || true
kill "${HEALTH_PID}" 2>/dev/null || true
rm -f "${SONYA_READY_FILE}"
log "worker exited rc=${RC}"
exit "${RC}"
