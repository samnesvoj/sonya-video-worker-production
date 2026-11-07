# syntax=docker/dockerfile:1.7
# SONYA video worker - BASE image: environment only, NO SONYA application code.
# The release (scripts/, modes/) is fetched at start by bootstrap from the PRIVATE GitHub repo at a pinned SHA.
# Rebuild only when requirements/worker.lock.txt or system packages change.
#   docker buildx build --platform linux/amd64 -f deploy/docker/Dockerfile \
#     --build-arg BASE_LOCK_SHA256=$(shasum -a 256 requirements/worker.lock.txt | cut -d' ' -f1) \
#     --build-arg IMAGE_VERSION=<version> -t sonya-video-worker-base:<version> .
# No secrets at build time; the build context is limited by .dockerignore.
FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ffmpeg git ca-certificates curl libglib2.0-0 libgomp1 libgl1 tini \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin sonya \
 && mkdir -p /opt/sonya/releases /opt/sonya/models /work/tmp /run/sonya /opt/sonya-bootstrap \
 && chown -R sonya:sonya /opt/sonya /work /run/sonya

# Hash-locked deps. torch 2.5.1+cu124 wheels bundle the CUDA runtime; the driver comes from the host (NVIDIA runtime).
COPY requirements/worker.lock.txt /opt/sonya-bootstrap/worker.lock.txt
RUN pip install --require-hashes --no-deps -r /opt/sonya-bootstrap/worker.lock.txt

COPY bootstrap/sonya_bootstrap.py bootstrap/healthd.py bootstrap/entrypoint.sh /opt/sonya-bootstrap/
RUN chmod 0555 /opt/sonya-bootstrap/*.py /opt/sonya-bootstrap/entrypoint.sh && chmod 0444 /opt/sonya-bootstrap/worker.lock.txt

ARG BASE_LOCK_SHA256
ARG IMAGE_VERSION=dev
ENV SONYA_BASE_LOCK_SHA256=${BASE_LOCK_SHA256} SONYA_IMAGE_VERSION=${IMAGE_VERSION} \
    APP_ENV=production WORKER_BACKEND_MODE=api MODELS_LOCAL_DIR=/opt/sonya/models SONYA_STATE_DIR=/run/sonya

USER sonya
WORKDIR /opt/sonya
# Loopback only; no EXPOSE, no inbound ports.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD curl -fsS http://127.0.0.1:8099/healthz >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/sonya-bootstrap/entrypoint.sh"]
