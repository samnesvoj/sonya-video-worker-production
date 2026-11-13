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

# Hash-locked deps in THREE layers (big + stable first): Docker/Vast download layers in parallel and reuse an unchanged big layer from the
# host cache. requirements/split/*.lock.txt are generated from worker.lock.txt by tools/split_lock.py (CI checks they match).
# The +cu126 wheels live on the PyTorch index, not on PyPI: the lock pins them (with hashes) but cannot carry the index URL.
# Each RUN also removes build-time-only files (C/CUDA headers, static libs) IN THE SAME LAYER so the size really shrinks.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
# SPLIT_DIR / LOCK_FILE let CI build a CANDIDATE dependency set (e.g. a patched PyTorch) from the same Dockerfile; defaults = production.
ARG SPLIT_DIR=requirements/split
ARG LOCK_FILE=requirements/worker.lock.txt
ENV SP=/usr/local/lib/python3.11/site-packages
COPY ${SPLIT_DIR}/01-cuda-libs.lock.txt /tmp/req/01.txt
RUN pip install --require-hashes --no-deps --extra-index-url "${TORCH_INDEX_URL}" -r /tmp/req/01.txt \
 && { find "$SP/nvidia" -type d -name include -prune -exec rm -rf {} + || true; } \
 && { find "$SP/nvidia" -name '*.a' -delete || true; } && rm -rf /tmp/req/01.txt /root/.cache
COPY ${SPLIT_DIR}/02-torch.lock.txt /tmp/req/02.txt
# triton (0.67 GB unpacked) is only used by torch.compile / inductor, which SONYA never calls; the package files go,
# its dist-info stays (versions + `pip check` unchanged, torch sees "no triton" and never imports it).
RUN pip install --require-hashes --no-deps --extra-index-url "${TORCH_INDEX_URL}" -r /tmp/req/02.txt \
 && rm -rf "$SP/torch/include" "$SP/torch/share" "$SP/triton" /tmp/req/02.txt /root/.cache
COPY ${SPLIT_DIR}/03-rest.lock.txt /tmp/req/03.txt
# Slim (SPEED_REPORT.md, 2026-11-08) - same versions, same lock, only files no code path loads:
#  * opencv: the three 4.14.0.94 wheels share ONE cv2/ package; the active cv2.abi3.so belongs to opencv-contrib-python and
#    links only $ORIGIN/../opencv_contrib_python.libs -> the other two *.libs dirs (0.21 GB) are dead. Checked by hash below.
#  * mediapipe (0.13 GB): only the removed old virality path imported it (optional import everywhere).
#  * onnx/backend/test (25 MB): onnx's own test data.
COPY tools/slim_site_packages.py /tmp/slim.py
RUN pip install --require-hashes --no-deps -r /tmp/req/03.txt \
 && python3 /tmp/slim.py "$SP" && rm -rf /tmp/req/03.txt /tmp/slim.py /root/.cache
# yt-dlp (pinned, hash-checked) + deno (pinned, sha256-checked): URL jobs (source_fetch) download on the GPU server, because the Moscow VPS
# cannot reach youtube.com. Small layers at the end: the big dependency layers above stay cached.
ARG DENO_VERSION=2.9.8
ARG DENO_SHA256=e890975bc2549807bbcc0b584ffafda72cc7c93349dd99c80483dd784f6c4609
COPY requirements/ytdlp.lock.txt /tmp/req/ytdlp.txt
RUN pip install --require-hashes --no-deps -r /tmp/req/ytdlp.txt && rm -rf /tmp/req/ytdlp.txt /root/.cache \
 && curl -fsSL -o /tmp/deno.zip "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" \
 && echo "${DENO_SHA256}  /tmp/deno.zip" | sha256sum -c - \
 && python3 -c "import zipfile;zipfile.ZipFile('/tmp/deno.zip').extract('deno','/usr/local/bin')" \
 && chmod 0755 /usr/local/bin/deno && rm /tmp/deno.zip && deno --version && yt-dlp --version
# The full lock (hash-checked by bootstrap against the release) stays as a tiny read-only file.
COPY ${LOCK_FILE} /opt/sonya-bootstrap/worker.lock.txt

COPY bootstrap/sonya_bootstrap.py bootstrap/healthd.py bootstrap/entrypoint.sh /opt/sonya-bootstrap/
RUN chmod 0555 /opt/sonya-bootstrap/*.py /opt/sonya-bootstrap/entrypoint.sh && chmod 0444 /opt/sonya-bootstrap/worker.lock.txt

ARG BASE_LOCK_SHA256
ARG IMAGE_VERSION=dev
# "video" makes the NVIDIA runtime mount libnvidia-encode/-decode: without it NVENC fails and the clips render on the CPU
# (run 4, 2026-11-08: "render encoder: cpu @ 6M", 179 s vs 108 s on the Mac mini). The dispatcher also passes it per instance.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV SONYA_BASE_LOCK_SHA256=${BASE_LOCK_SHA256} SONYA_IMAGE_VERSION=${IMAGE_VERSION} \
    APP_ENV=production WORKER_BACKEND_MODE=api MODELS_LOCAL_DIR=/opt/sonya/models SONYA_STATE_DIR=/run/sonya

USER sonya
WORKDIR /opt/sonya
# Loopback only; no EXPOSE, no inbound ports.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD curl -fsS http://127.0.0.1:8099/healthz >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/sonya-bootstrap/entrypoint.sh"]
