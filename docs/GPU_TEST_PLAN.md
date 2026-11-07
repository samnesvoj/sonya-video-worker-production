# Final GPU test plan (NOT started; needs your explicit OK)
Cost: read-only Vast search (2026-11-07; verified, reliability>0.98, CUDA≥12.4, disk≥80 GB; 64 offers) shows RTX 3090 24 GB from ≈ $0.17/h; 4090/A5000/L4 roughly $0.3–0.6/h. 1–1.5 h ≈ $0.3–1.0 plus traffic. Suggested cap $3, auto-destroy at 90 min.
Blockers: (1) GitHub App created + token minting wired into the dispatcher; (2) base image built in Actions and its digest; (3) `release/models.lock.json` filled; (4) staging/stub backend with its own WORKER_SECRET (not the real user queue); (5) Template created per VAST_TEMPLATE.md.
Steps: 1 instance → bootstrap OK, readiness OK (GPU/CUDA/torch.cuda/NVENC/models) → negatives (wrong SHA refused; Trailer job → `mode_disabled`; SIGTERM requeues) → one short real clip per enabled mode (virality, beep), compare to Mac mini results (ffprobe, contact sheet) → record VRAM/RAM/time/cost, NVENC vs CPU → log leak grep → destroy instance. Opt-in modes only afterwards, one by one.

## Additional checks for idle-stop and accounting (same paid session)
- `SONYA_FLEET_DRY_RUN=false`, one test job: confirm create → running → claim → completed → **stop at ≈60 s** → `exited`; then a second test job → **RESUME** time and whether the GPU is still available; then wait out retention → DESTROY and confirm the instance disappears from `GET /api/v1/instances/`.
- Capture the real shapes of `PUT /asks/{id}/`, `GET /api/v1/instances/`, `GET /api/v0/charges/` and compare with `dispatcher/vast_api.py` and `accounting/vast_billing.py` (they are written from the CLI source, not from live responses).
- Kill test: `vastai stop instance` while a job is `mode_running` → the job must return to `queued` with `WORKER_SHUTDOWN` and complete after RESUME/CREATE.
- Measure: real GPU-minutes per video, model size/download time, stopped-disk billing rate, SIGTERM grace.
