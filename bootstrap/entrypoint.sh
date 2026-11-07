#!/usr/bin/env bash
# Base-image ENTRYPOINT: bootstrap (pinned SHA) -> drop GitHub token -> exec the release's run_worker.sh.
# Never prints env values; the token is never written to disk, argv or a log.
set -Eeuo pipefail
umask 077
log() { printf '[entrypoint] %s\n' "$*" >&2; }

export SONYA_STATE_DIR="${SONYA_STATE_DIR:-/run/sonya}"
mkdir -p "${SONYA_STATE_DIR}"
# Start-up timeline (names + epoch seconds only; see scripts/ops/timeline.py). Truncated on every container start (RESUME restarts it).
export SONYA_TIMELINE_FILE="${SONYA_TIMELINE_FILE:-${SONYA_STATE_DIR}/timeline.jsonl}"
: > "${SONYA_TIMELINE_FILE}"
mark() { python3 -c 'import json,sys,time;open(sys.argv[1],"a").write(json.dumps({"name":sys.argv[2],"t":round(time.time(),3),**dict(a.split("=",1) for a in sys.argv[3:])})+"\n")' "${SONYA_TIMELINE_FILE}" "$@" || true; }
CACHED=no; [[ -f "${SONYA_RELEASES_DIR:-/opt/sonya/releases}/${SONYA_RELEASE_SHA:-none}/.sonya-verified.json" ]] && CACHED=yes
mark entrypoint_start "release_cached=${CACHED}"
log "SONYA video worker base image (version=${SONYA_IMAGE_VERSION:-unknown}) release pin=${SONYA_RELEASE_SHA:0:12}"

if ! RELEASE_DIR="$(python3 "${SONYA_BOOTSTRAP_DIR:-/opt/sonya-bootstrap}/sonya_bootstrap.py")"; then
  log "bootstrap failed - worker NOT started"
  exit 2
fi
unset GITHUB_RELEASE_TOKEN GH_TOKEN GITHUB_TOKEN   # credentials must not outlive the download
mark bootstrap_done

export SONYA_RELEASE_DIR="${RELEASE_DIR}" PYTHONPATH="${RELEASE_DIR}"
exec "${RELEASE_DIR}/deploy/vast/run_worker.sh"
