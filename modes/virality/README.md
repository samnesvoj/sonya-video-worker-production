# virality

Selects the highest-hook moments from a video for maximum engagement.

## Pipeline

1. `hook_mode_v1.py` (legacy_gpu) — extracts candidate segments
2. `modes_scoring.py` (legacy_gpu) — scores segments
3. Visual boosts: Gemini moments, person in first 3s, layout, speaker/pose
4. Smart vertical crop → 1080×1920

## Models (optional)

| Key | S3 path |
|-----|---------|
| yolo11n | models/common/yolo11n.pt |
| yolo11n_pose | models/common/yolo11n-pose.pt |
| webcam_detector | models/common/webcam_detector.pt |
