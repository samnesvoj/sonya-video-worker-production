# Old virality path - REMOVED (2026-11-08)

Owner, 2026-11-08: «ТАК СРОЧНО ПЕРЕНЕСТИ ВЕРСИЮ МАК МИНИ !!!! старое это не надо», and permission to
delete the old way once the Mac mini version passed the Mac mini check (it did: kishlak_15m 3/3 clips,
crop, 403/403 subtitle lines; BEEP 19/19 intervals; then the same on Vast, run 4).

Virality now runs only the **Mac mini recipe** - what the SONYA Desktop daemon
(`sonya-desktop/desktop/sonya_desktop/modes/worker.py`, `mode.run`) passes to the same runner:
`scripts/shared/cut_profile.py` (selection `story`, crop `tracked` per clip with YuNet, burned ASS
subtitles, mandatory ASR, Desktop encoder args at 6 Mb/s). Anything else is refused by
`modes/virality/runner.py`; no words → `ASR_FAILED`, no story → `NO_CLIPS` (Desktop fails the same way).

## Removed from virality

| Old step | Was in |
|---|---|
| `hook_mode_v1` + visual boosts, 0–30 s fallback without ASR | `modes/virality/runner.py::_legacy_segments` |
| Gemini / local `select_clips` selection (`llm=on`) | `modes/virality/runner.py::_select_and_compose` |
| `crop_hints` + `compose_vertical_clip` letterbox composer | `modes/virality/runner.py::_compose` |
| rollback switch `SONYA_CUT_PROFILE=legacy` | `scripts/shared/cut_profile.py::applies` |

Shared modules (`cut_selection.select_clips`, `smart_crop_adapter`, `gemini_composer`, `sonya_enhancer`,
`legacy_gpu/*`) stay: stories, streamer, educational and trailer use them (Trailer is not touched).

## Full copy of the old way

* git tag `old-worker-way` (= 02d4cfb, before the Mac mini port), pushed to origin;
* `/Users/agent/SONYA/archive/old-worker-way-2026-11-08.tar.gz` on the Mac mini.

Rollback = deploy the previous release SHA (CUTOVER.md), not an env switch.
