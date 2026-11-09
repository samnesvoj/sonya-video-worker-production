# Educational v4 — lesson series with templates

Branch `feat/educational-v4` (from `origin/release/first-vast-test`, v2 overlays cherry-picked). Concept for the owner:
`/Users/agent/SONYA/worker-prep/edu-v4/EDU_CONCEPT.md`; test report: `.../edu-v4/REPORT.md`.

## Pipeline (modes/educational/runner.py)

1. Words: `enrich_video_for_mode` (GigaAM, or `params.word_timestamps_path`).
2. `educational_mode_v5.run_educational_mode_v5` — topics, windows, audio/semantic scores (audio via ffmpeg + numpy when
   librosa is absent — the worker image has none). Failure → warning `educational_v5_failed`, selection continues.
3. `selection.select_lesson_clips` (edu-select-v4): candidates = whole sentences from `scripts/shared/cut_selection`
   (the Cut rules: start = sentence start, end = statement end, never `?` / `…`), score = Cut story score + v5 semantic
   layer + v5 moment + «complete thought» heuristics; one clip per topic first → series in lecture order.
4. Optional Gemini editor (`editor.py`, analysis only): one request per video with ≤ 24 candidate texts; returns per id
   score / complete / self_contained / title / topic. Timestamps never come from the model. Budget forecast before the
   request (`overlays/budget.py`, 5 RUB per video, ledger per job in `<out>/edu_llm_ledger.json`).
5. `scripts/shared/boundaries.refine_segments` — exact render edges on ASR words (−150 ms pre-roll, as Cut).
6. Per clip: `stage.choose` → **stage** (whole slide/board on the template background, rounded corners) or **crop**
   (a big talking head, existing `compose_vertical_clip`). Stage render failure → crop + warning `edu_stage_failed`.
7. `overlays.apply_overlays(template=…)`: template subtitles (ASS, libass), the title (first 3.6 s), cards (v2 planner +
   evidence gate; on a stage cards live above the slide and never cover it).
8. No v4 segments at all → the old fallback (first minute / layout cuts) with warning `educational_engine_unavailable`.

## Job parameters

| param | meaning | default |
|---|---|---|
| `settings.edu_template` / `template` | 1 Белый лист, 2 Серебро, 3 Графит (dark), 4 Фиолет | 1 |
| `settings.subtitleStyle = "none"` / `subtitles=false` | no burned subtitles | on |
| `edu_title=false` | no title | on |
| `llm` = `on`/`off`, `edu.editor` | Gemini editor + card planner | on when a provider is configured (same rule as Cut: `SONYA_LLM_EDITOR`) |
| `edu.cards` / `edu_overlays` | cards | follows the LLM switch |
| `edu.frame` = `stage`/`crop` | force the frame layout | auto (YuNet faces) |
| `max_clips` | series length | 6 |

## Enable on Vast

The mode stays opt-in (`scripts/mode_policy.py`): add `educational` to `ENABLED_MODES` on the worker. The worker FFmpeg
must have libass (it already burns Cut subtitles). For the cheap model set `TIMEWEB_GEMINI_MODEL=gemini/gemini-3.1-flash-lite`.
Fonts ship in `scripts/shared/fonts` (OFL: Onest, Golos Text, Manrope, Unbounded — static subsets; Dela Gothic One).

## Site

`sonya-production` branch `feat/edu-templates-ui` (local, not pushed): a 4-picture picker under the mode cards, shown
only for Educational; `collectFormData()` adds `settings.edu_template`. Backend passes `params` through unchanged.
