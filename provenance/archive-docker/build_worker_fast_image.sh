#!/usr/bin/env bash
# build_worker_fast_image.sh
# Build and (optionally) push the lightweight sonya-worker:fast image.
#
# Usage:
#   bash deploy/docker/build_worker_fast_image.sh
#
# Set GHCR_TOKEN env var to push to GHCR after build.

set -euo pipefail

IMAGE_BASE="ghcr.io/samnesvoj/sonya-worker"
TAG_FAST="fast"
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
TAG_SHA="fast-${SHA}"

echo ""
echo "==> Building ${IMAGE_BASE}:${TAG_FAST} (fast lightweight image) ..."
echo "    Dockerfile: deploy/docker/Dockerfile.worker.fast"
echo "    Context:    . (repo root)"
echo ""

docker build \
  -f deploy/docker/Dockerfile.worker.fast \
  --build-arg SONYA_GIT_SHA="$(git rev-parse HEAD)" \
  -t "${IMAGE_BASE}:${TAG_FAST}" \
  -t "${IMAGE_BASE}:${TAG_SHA}" \
  .

echo ""
echo "==> Image size (uncompressed):"
SIZE_BYTES=$(docker image inspect "${IMAGE_BASE}:${TAG_FAST}" --format='{{.Size}}')
SIZE_GB=$(echo "scale=2; ${SIZE_BYTES} / 1073741824" | bc)
echo "    ${SIZE_GB} GB  (${SIZE_BYTES} bytes)"

echo ""
echo "==> Built tags:"
echo "    ${IMAGE_BASE}:${TAG_FAST}"
echo "    ${IMAGE_BASE}:${TAG_SHA}"

if [[ -n "${GHCR_TOKEN:-}" ]]; then
  echo ""
  echo "==> Logging in to GHCR and pushing ..."
  GHCR_USER="${GHCR_USERNAME:-samnesvoj}"
  echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin
  docker push "${IMAGE_BASE}:${TAG_FAST}"
  docker push "${IMAGE_BASE}:${TAG_SHA}"
  echo "==> Pushed: ${IMAGE_BASE}:${TAG_FAST}"
else
  echo ""
  echo "GHCR_TOKEN not set — skipping push."
  echo "To push: export GHCR_TOKEN=<token> && bash deploy/docker/build_worker_fast_image.sh"
fi
