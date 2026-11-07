#!/usr/bin/env bash
# build_worker_image.sh — Build and push SONYA GPU worker Docker image
# Run from the repo root:
#   bash deploy/docker/build_worker_image.sh
#
# Requires: Docker, git

set -euo pipefail

IMAGE_BASE="ghcr.io/samnesvoj/sonya-worker"
TAG_LATEST="${IMAGE_BASE}:latest"
GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo "nogit")
TAG_SHA="${IMAGE_BASE}:${GIT_SHA}"

echo "=== SONYA Worker Docker Build ==="
echo "Image:   ${TAG_LATEST}"
echo "SHA tag: ${TAG_SHA}"
echo ""

# Build from repo root so COPY instructions resolve correctly
docker build \
    -f deploy/docker/Dockerfile.worker \
    --build-arg SONYA_GIT_SHA="$(git rev-parse HEAD)" \
    -t "${TAG_LATEST}" \
    -t "${TAG_SHA}" \
    .

echo ""
echo "Build successful."
echo ""

# Optional push to GHCR
read -r -p "Push to ghcr.io? (y/N) " PUSH
if [[ "${PUSH}" == "y" || "${PUSH}" == "Y" ]]; then
    # Login if credentials are set
    if [[ -n "${GHCR_TOKEN:-}" && -n "${GHCR_USERNAME:-}" ]]; then
        echo "Logging in to ghcr.io as ${GHCR_USERNAME}..."
        echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME}" --password-stdin
    fi
    docker push "${TAG_LATEST}"
    docker push "${TAG_SHA}"
    echo "Pushed: ${TAG_LATEST}"
    echo "Pushed: ${TAG_SHA}"
else
    echo "Skipped push. To push manually:"
    echo "  docker push ${TAG_LATEST}"
    echo "  docker push ${TAG_SHA}"
fi
