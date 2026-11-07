# streamer (beta)

Streamer highlight extraction using webcam detection, lip sync, and yuvelirochka clipper stack.

## Pipeline

1. Webcam detector → person tracking boxes (required model: webcam_detector.pt)
2. Lip sync detection → active speaker segments
3. `legacy/analyzer.py` + `legacy/clipper.py` (from yuvelirochka)
4. Smart vertical crop → 1080×1920

## Status

Beta — not production. `webcam_detector.pt` is required (`optional: false`).
