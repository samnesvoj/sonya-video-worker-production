# build_worker_image.ps1 — Build and push SONYA GPU worker Docker image
# Run from the repo root:
#   .\deploy\docker\build_worker_image.ps1
#
# Requires: Docker Desktop, git

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$IMAGE_BASE = "ghcr.io/samnesvoj/sonya-worker"
$TAG_LATEST = "${IMAGE_BASE}:latest"

# Git SHA tag (first 12 chars)
$GIT_SHA = (git rev-parse --short=12 HEAD 2>$null)
if (-not $GIT_SHA) {
    $GIT_SHA = "nogit"
}
$TAG_SHA = "${IMAGE_BASE}:${GIT_SHA}"

Write-Host "=== SONYA Worker Docker Build ==="
Write-Host "Image:  $TAG_LATEST"
Write-Host "SHA tag: $TAG_SHA"
Write-Host ""

# Build from repo root so COPY instructions work correctly
docker build `
    -f deploy/docker/Dockerfile.worker `
    --build-arg SONYA_GIT_SHA=$(git rev-parse HEAD) `
    -t $TAG_LATEST `
    -t $TAG_SHA `
    .

if ($LASTEXITCODE -ne 0) {
    Write-Error "docker build failed"
    exit 1
}

Write-Host ""
Write-Host "Build successful."
Write-Host ""

# Optional: push to GHCR
$PUSH = Read-Host "Push to ghcr.io? (y/N)"
if ($PUSH -eq "y" -or $PUSH -eq "Y") {
    # Login if GHCR_TOKEN is set in environment
    if ($env:GHCR_TOKEN -and $env:GHCR_USERNAME) {
        Write-Host "Logging in to ghcr.io as $env:GHCR_USERNAME..."
        $env:GHCR_TOKEN | docker login ghcr.io -u $env:GHCR_USERNAME --password-stdin
    }
    docker push $TAG_LATEST
    docker push $TAG_SHA
    Write-Host "Pushed: $TAG_LATEST"
    Write-Host "Pushed: $TAG_SHA"
} else {
    Write-Host "Skipped push. To push manually:"
    Write-Host "  docker push $TAG_LATEST"
    Write-Host "  docker push $TAG_SHA"
}
