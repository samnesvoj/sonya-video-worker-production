# stories

Extracts narrative story segments from long-form video.

## Pipeline

1. `story_mode_v1.py` (legacy_gpu) — story arc extraction
2. `llm_segment_analysis.py` (legacy_gpu) — LLM-based segment scoring
3. Enrichment: visual_events (YOLO), layout_segments (Gemini), gemini_moments
4. Smart vertical crop → 1080×1920
