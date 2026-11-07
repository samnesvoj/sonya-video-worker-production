# educational

Layout-aware educational content extraction.

## Layout detection

| Layout type | Strategy |
|-------------|----------|
| screen_share (>50% segments) | preserve_screen — letterbox |
| single_speaker (>50% segments) | person_focus — person center crop |
| other | default — smart crop |

## Pipeline

1. Gemini layout analysis → detect dominant layout
2. `educational_mode_v5.py` (legacy_gpu) — segment extraction
3. Layout-aware crop hints → smart crop → 1080×1920
