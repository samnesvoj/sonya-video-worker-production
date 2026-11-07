## What changed (one mode / one problem)
Mode: `virality | beep | stories | educational | streamer | sonya_gen | infra`  ·  Branch: `mode/<mode>/<slug>`

## Problem seen after the Vast test
<!-- job id / symptom / control video name. Never paste user video, transcripts or tokens. -->

## Evidence (attach `report.md` / `report.json` from `tools/compare_runs.py`)
- [ ] `python tools/mode_dev.py test <mode>` is green
- [ ] control video compared with the previous version: verdict ___ , SSIM ___ , duration Δ ___
- [ ] I watched the output (not only the automatic checks)
- [ ] shared code touched? (`scripts/shared/*`) → tests of every mode that imports it were run: ___

## Safety
- [ ] Trailer untouched and still disabled; no change to `mode_policy.py` HARD_DISABLED / default modes
- [ ] no secrets, videos, transcripts, model files in the diff
- [ ] worker API contract unchanged (`tests/test_api_contract.py`) or change agreed with the owner

## Release
This PR does **not** release anything. After merge, only the owner cuts a release (`docs/dev/RELEASE_AND_ROLLBACK.md`).
