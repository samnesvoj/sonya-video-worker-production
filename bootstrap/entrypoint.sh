#!/usr/bin/env bash
# Base-image ENTRYPOINT: bootstrap (pinned SHA) -> drop GitHub token -> exec the release's run_worker.sh.
# Never prints env values; the token is never written to disk, argv or a log.
set -Eeuo pipefail
umask 077
log() { printf '[entrypoint] %s\n' "$*" >&2; }

export SONYA_STATE_DIR="${SONYA_STATE_DIR:-/run/sonya}"
mkdir -p "${SONYA_STATE_DIR}"
log "SONYA video worker base image (version=${SONYA_IMAGE_VERSION:-unknown}) release pin=${SONYA_RELEASE_SHA:0:12}"

if ! RELEASE_DIR="$(python3 "${SONYA_BOOTSTRAP_DIR:-/opt/sonya-bootstrap}/sonya_bootstrap.py")"; then
  log "bootstrap failed - worker NOT started"
  exit 2
fi
unset GITHUB_RELEASE_TOKEN GH_TOKEN GITHUB_TOKEN   # credentials must not outlive the download

export SONYA_RELEASE_DIR="${RELEASE_DIR}" PYTHONPATH="${RELEASE_DIR}"
exec "${RELEASE_DIR}/deploy/vast/run_worker.sh"
