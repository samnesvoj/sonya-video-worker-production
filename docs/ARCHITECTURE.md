# Architecture

```
MacBook ─push→ GitHub PRIVATE repo ← CI: tests, gitleaks, pip-audit, docker build
SONYA VPS (trusted: App private key, payments, DB, long-lived secrets)
  dispatcher: mint token (read-only, 1 repo, ≤1h) + create Vast instance (Template + env)
Vast instance: base image (deps + bootstrap, NO SONYA code)
  entrypoint.sh → sonya_bootstrap.py → git fetch <SHA> → verify SHA/manifest/lock → unset token
  → run_worker.sh (from the verified release) → preflight → readiness → /run/sonya/ready
  → healthd 127.0.0.1:8099 → gpu_worker --poll → HTTPS API (Bearer) → S3
```
**Variant A vs B.** A (chosen): update = change `SONYA_RELEASE_SHA`; image rebuilt only on dependency change; no registry credentials needed because the image holds no code (can be public); GitHub token ≤1h read-only per instance. B (private image by digest): every code change needs a build and a multi-GB pull, and a registry pull credential must live in Vast. Neither hides code from the host owner.
**Shared vs per-mode.** `scripts/gpu_worker.py` is the shared shell (claim → input from S3 → models → runner → upload → status/heartbeat → complete/fail → cleanup). Modes keep their own runners; `scripts/shared/*` (ASR, crop, subtitles, beep, selection) are reused where the archive already shared them. New: `mode_policy`, `ops/readiness`, graceful SIGTERM, readiness gate.
**Release identity.** `git fetch --depth 1 origin <sha>` with `transfer.fsckObjects`, `FETCH_HEAD==SHA` and `HEAD==SHA`; optional tag must resolve to the SHA. `main` HEAD is never used. Running instances never update; releases on disk are immutable.
**Cannot be promised:** the host owner can read code, env, memory and videos. Mitigation: ≤1h read-only token, no payment/DB/long-lived GitHub keys on Vast, target state moves LLM keys off the GPU.
**Production compatibility** (`contracts/worker_api_bd8c968.json`, tests/test_api_contract.py): all 7 client endpoints exist on bd8c968; worker statuses ⊂ server `_VALID_STATUSES`; `claim` accepts `modes`. `gpu_worker.py` was byte-identical to bd8c968 before our changes. The live API was not called, the VPS not touched.
