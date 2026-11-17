"""
SONYA Trailer (production, 2026-11-17; owner: «улучши режим Трейлер во всех аспектах»).

A TikTok-style film recap from ONE storyline of a film / episode, told by a first-person narrator:
hook from the middle → backstory → rising stakes → twist → cliffhanger; ~60–100 s + 2–3 shorts of 30–45 s.

    script.py   GigaAM words → Gemini (Timeweb, text + frame collages only, ≤ 1.5 ₽): storyline → narration →
                fact check against the transcript; free deterministic checks (names, digits, pace, order)
    shots.py    the episode's shots (cuts, faces, motion, burned-in captions) + the frame collages the writer sees
    voice.py    local TTS in its own process (Qwen3-TTS, Apache-2.0) → per-phrase WAVs; GigaAM on the result for
                word times (subtitles) and the «no clipped words» check
    edit.py     phrase → shots (the writer's picks + the shots of the lines it retells), cut rhythm 1.8–2.8 s
    render.py   crop on the main subject (film picture ~3:4 big on 9:16 over its own blur), ducking of the film
                sound under the voice, subtitles in the site's style on the picture, hook title, cliffhanger,
                small SONYA end card; validator + contact sheet
    pipeline.py the whole run for the worker (modes/virality/runner.py dispatches clipType=trailer here)

Docs: docs/TRAILER_PROD.md.
"""
