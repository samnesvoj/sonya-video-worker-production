# build_worker_fast_image.ps1
# Build and (optionally) push the lightweight sonya-worker:fast image.
#
# Usage:
#   .\deploy\docker\build_worker_fast_image.ps1
#
# Set GHCR_TOKEN env var to push to GHCR after build.

$ErrorActionPreference = "Stop"

$IMAGE_BASE = "ghcr.io/samnesvoj/sonya-worker"
$TAG_FAST   = "fast"
$SHA        = (git rev-parse --short HEAD 2>$null) ?? "local"
$TAG_SHA    = "fast-$SHA"

Write-Host ""
Write-Host "==> Building ${IMAGE_BASE}:${TAG_FAST} (fast lightweight image) ..." -ForegroundColor Cyan
Write-Host "    Dockerfile: deploy/docker/Dockerfile.worker.fast"
Write-Host "    Context:    . (repo root)"
Write-Host ""

docker build `
  -f deploy/docker/Dockerfile.worker.fast `
  --build-arg SONYA_GIT_SHA=$(git rev-parse HEAD) `
  -t "${IMAGE_BASE}:${TAG_FAST}" `
  -t "${IMAGE_BASE}:${TAG_SHA}" `
  .

if ($LASTEXITCODE -ne 0) {
    Write-Error "docker build failed"
    exit 1
}

Write-Host ""
Write-Host "==> Image size (uncompressed):" -ForegroundColor Cyan
docker image inspect "${IMAGE_BASE}:${TAG_FAST}" --format='{{.Size}}' | ForEach-Object {
    $bytes = [int64]$_
    $gb = [math]::Round($bytes / 1GB, 2)
    Write-Host "    ${gb} GB  ($_  bytes)"
}

Write-Host ""
Write-Host "==> Built tags:" -ForegroundColor Green
Write-Host "    ${IMAGE_BASE}:${TAG_FAST}"
Write-Host "    ${IMAGE_BASE}:${TAG_SHA}"

if ($env:GHCR_TOKEN) {
    Write-Host ""
    Write-Host "==> Logging in to GHCR and pushing ..." -ForegroundColor Cyan
    $user = if ($env:GHCR_USERNAME) { $env:GHCR_USERNAME } else { "samnesvoj" }
    $env:GHCR_TOKEN | docker login ghcr.io -u $user --password-stdin
    docker push "${IMAGE_BASE}:${TAG_FAST}"
    docker push "${IMAGE_BASE}:${TAG_SHA}"
    Write-Host "==> Pushed: ${IMAGE_BASE}:${TAG_FAST}" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "GHCR_TOKEN not set — skipping push." -ForegroundColor Yellow
    Write-Host "To push:  `$env:GHCR_TOKEN = '<token>'; .\deploy\docker\build_worker_fast_image.ps1"
}
