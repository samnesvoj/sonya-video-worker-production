# Status (2026-11-07)
**Done, tested locally (CPU, Apple Silicon):** Trailer hard-disabled; explicit claim allow-list; SIGTERM requeue; pinned-SHA bootstrap (identity, manifest, lock, redaction, symlink checks); readiness gate; loopback health; hashed lock; API-contract tests vs bd8c968; archive's own 581 tests green. Total: see CI / `pytest -q`.
**Simulated only:** `tools/local_chain_smoke.py` (stub API, dummy model files, GPU check skipped).
**Needs real Vast GPU:** image build/install of the lock (torch/mediapipe/numpy 2.x), CUDA/NVENC, VRAM/time/cost, GPU quality of virality/beep vs Mac mini, every opt-in mode.
**Needs you / outside this repo:** GitHub App; first image publish; fill `models.lock.json`; dispatcher change on the VPS (token + Template); staging backend for the test; decision on opt-in modes.
