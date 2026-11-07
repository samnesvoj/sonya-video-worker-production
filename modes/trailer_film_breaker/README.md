# trailer_film_breaker

Production trailer mode. Produces vertical 1080×1920 cinematic clips from long-form video.

## What it does

- Runs full enrichment pipeline via `sonya_enhancer`
- Gemini layout + composition analysis
- YOLO person/object detection
- Word-level subtitle alignment (faster-whisper)
- Smart vertical crop (1080×1920) with center-crop fallback

## Required model

| Key | S3 path | Required |
|-----|---------|----------|
| trailer_yolo_custom | models/trailer/best.pt | **yes** |
| yolo11n | models/common/yolo11n.pt | optional |
| yolo11n_pose | models/common/yolo11n-pose.pt | optional |
| webcam_detector | models/common/webcam_detector.pt | optional |

## Usage

```bash
python scripts/gpu_worker.py --once --job-id JOB_ID
```

## Notes

- Does NOT use `scripts/legacy_gpu/trailer_mode_v3.py`
- `downloader.py` is NOT connected — accepts local file only
