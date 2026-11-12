# Mode registry — what the archive really proves

The archive contains code, specs and reports, **not** the Mac mini runtime (videos, models, benchmarks were excluded). So evidence below
is what the archive's own documents record (owner-reviewed real clips, measured timings); nothing was re-run on real video or on a GPU here.
Unit tests are not counted as proof of a mode.

| Mode | Evidence in archive | GPU proof | Worker default | Status |
|---|---|---|---|---|
| virality (Cut) | `docs/CUT_PLAYBOOK.md`: rules and measured failures from owner review of real clips; Mac mini M4 timings (2.5 h interview: ASR 154 s, 8 clips ≈70 s, 1 clip ≈12 s); INVENTORY: real clips 12–17 Oct | none (Mac path: GigaAM/mlx; worker uses faster-whisper CUDA) | **ENABLED** | working on Mac; GPU path unproven |
| beep | `docs/MODES_SPEC.md`: "Desktop: real run happened"; INVENTORY: fixture 5/5 words, 0 false | none | **ENABLED** (server does not dispatch beep yet; harmless) | confirmed on fixture |
| stories | INVENTORY: ran via fallback, no real-video proof | none | opt-in (`ENABLED_MODES`) | needs work |
| educational | same | none | opt-in | needs work; v2 overlay cards behind `params["edu_overlays"]` (off), see EDUCATIONAL_V2.md |
| streamer | 1-second fixture only; batch flow covered by unit tests | none | opt-in | beta |
| sonya_gen | 17 frames OK (bf16); 73 frames = noise on MPS; paused | none | opt-in | partial/paused |
| trailer_film_breaker, trailer_cinematic | real outputs recorded in TRAILER_* reports | none | **HARD-DISABLED in code** | OFF; code, tests, models kept |

Enabling an opt-in mode for production = real-video + GPU proof, then add it to `DEFAULT_ENABLED_MODES`, `release/manifest.json`, new SHA.
Server `ALLOWED_MODES` (bd8c968) still lists stories/educational/streamer/sonya_gen: until those are enabled here, jobs of those modes must keep going to the old worker pool (the claim filter ensures this worker never takes them).
