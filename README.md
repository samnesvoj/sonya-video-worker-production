# SONYA video worker — production candidate (Vast.ai)

PRIVATE repository. Built only from the Mac mini archive `sonya-vast-handoff-6992288` (branch `feat/trailer-cinematic`, commit `6992288`);
the older `samnesvoj/sonya-gpu-worker` repo and the production repo were NOT used as a base (production `bd8c968` was only read to check API contracts).
The Docker image holds no SONYA code: the release is fetched at start from this repo at a pinned commit SHA.

Docs: [STATUS](docs/STATUS.md) · [FLEET_IDLE_STOP](docs/FLEET_IDLE_STOP.md) · [COSTS](docs/COSTS.md) · [ACCOUNTING](docs/ACCOUNTING.md) · [SAMPLE_REPORTS](docs/SAMPLE_REPORTS.md) · [DEPLOY_AND_TESTING](docs/DEPLOY_AND_TESTING.md)
 · [STATUS](docs/STATUS.md) · [MODES](docs/MODES_REGISTRY.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [VAST_TEMPLATE](docs/VAST_TEMPLATE.md) ·
[GITHUB_APP_SETUP](docs/GITHUB_APP_SETUP.md) · [RUNBOOK](docs/RUNBOOK.md) · [SECURITY_AUDIT](docs/SECURITY_AUDIT.md) ·
[ARCHIVE_INVENTORY](docs/ARCHIVE_INVENTORY.md) · [GPU_TEST_PLAN](docs/GPU_TEST_PLAN.md)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install pytest pyyaml requests pydantic fastapi boto3 numpy pillow python-dotenv httpx scenedetect
pip install --no-deps opencv-python==4.14.0.94 opencv-contrib-python==4.14.0.94 opencv-python-headless==4.14.0.94
python -m pytest -q                 # CPU tests
python tools/local_chain_smoke.py   # entrypoint -> bootstrap -> readiness -> worker -> SIGTERM (stub API; needs a commit)
```
