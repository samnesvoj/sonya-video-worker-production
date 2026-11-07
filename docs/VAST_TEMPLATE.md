# Vast.ai Template (not created yet — no paid actions)

| Parameter | Value |
|---|---|
| Image | `ghcr.io/samnesvoj/sonya-video-worker-production-base@sha256:<digest from publish-base-image>` (digest, never `latest`); may be public: no SONYA code inside |
| Launch mode | Docker ENTRYPOINT (not SSH/Jupyter); on-start script empty |
| Disk | ≥ 80 GB |
| Ports | none (outbound only; health is 127.0.0.1) |
| GPU | 1×, VRAM ≥ 24 GB (3090/4090/A5000/L4/L40), CUDA ≥ 12.4, verified, reliability ≥ 0.98 |

Template env (not secret): `SONYA_RELEASE_SHA=<40-hex>` · `SONYA_RELEASE_REPO=samnesvoj/sonya-video-worker-production` · `WORKER_BACKEND_MODE=api` · `SONYA_MIN_VRAM_MB=20000` · `SONYA_REQUIRE_MODEL_LOCK=1` · `SONYA_REQUIRE_NVENC=1` (after GPU check) · `ENABLED_MODES` empty (= `virality,beep`).
Per-instance env added by the dispatcher (secret, never in the Template): `GITHUB_RELEASE_TOKEN` (minted per instance) · `WORKER_SECRET` · `BACKEND_API_URL` · `S3_*` · `MODELS_S3_BUCKET`.
Update = change `SONYA_RELEASE_SHA` (and the image digest if dependencies changed): only NEW instances pick it up.
