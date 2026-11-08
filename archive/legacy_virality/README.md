# Old virality path - switched OFF (2026-11-08), kept for rollback

Owner, 2026-11-08: «ТАК СРОЧНО ПЕРЕНЕСТИ ВЕРСИЮ МАК МИНИ !!!! старое это не надо».

Production virality jobs now run the **Mac mini recipe** - exactly what the SONYA Desktop daemon
(`sonya-desktop/desktop/sonya_desktop/modes/worker.py`, `mode.run`) passes to the same runner:
`scripts/shared/cut_profile.py` (selection `story` / `qa`, crop `tracked` per clip with YuNet,
burned ASS subtitles «highlight» + Dela Gothic One, mandatory ASR, Desktop encoder args at 6 Mb/s).

The Cut library itself (`cut_selection`, `boundaries`, `crop/*`, `subtitles`, `subtitle_styles`, fonts,
`llm_editor`, `beep*`, `profanity`) is byte-identical to `sonya-desktop` (checked 2026-11-08 with `diff -rq`).

## What is OFF for production virality jobs

| Old step | Where | Why it is off |
|---|---|---|
| whole-video YOLO / pose / webcam / Gemini layout / lip sync | `scripts/shared/enhancers/sonya_enhancer.py` (skipped when `crop.mode=tracked`) | Desktop analyses each clip only; run 2 spent ~6 min of YOLO on a 15 min video |
| `crop_hints` + `compose_vertical_clip` (gemini_composer, letterbox for wide shots) | `modes/virality/runner.py::_compose` legacy branch | black bars, no subtitles (run 2) |
| `hook_mode_v1` + visual boosts when there are no ASR words | `modes/virality/runner.py::_legacy_segments` | Desktop never cuts without ASR: the job fails with `ASR_FAILED` |
| Gemini selection (`llm=on`, `select_clips`) | `scripts/shared/cut_selection.select_clips` | the story selector + LLM editor replaced it on Desktop |

The code is NOT deleted: the modules are shared with other modes (stories, streamer, educational, trailer)
that this change does not touch. Deleting anything only after the owner agrees.

## Rollback (no new release)

Set `SONYA_CUT_PROFILE=legacy` in the worker env (Vast template / `run_worker.sh`) and restart the worker:
every virality job goes through the old path again. Unit tests of the old path run with this switch
(`tests/test_cut_determinism.py`, `tests/test_mode_fallback_warnings.py`, `tests/test_virality_boundaries.py`).
