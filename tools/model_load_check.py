#!/usr/bin/env python3
"""
model_load_check.py - does THIS PyTorch load and run every model of the first GPU test? CPU only, offline, ~1 minute. Run it on the Mac mini
in two venvs (current torch and the candidate torch) BEFORE any paid GPU time: it catches the classic upgrade break - the stricter default of
torch.load (weights_only=True since PyTorch 2.6) refusing an old or custom checkpoint such as webcam_detector.pt.
  python tools/model_load_check.py --models-dir ~/SONYA-models
Prints torch/ultralytics versions, then per model: OK / FAIL (+ first 120 chars of the error). Exit 1 on any FAIL. Never uploads anything.
"""
import argparse, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--models-dir", required=True, type=pathlib.Path)
a = ap.parse_args()
man = json.loads((ROOT / "release" / "models.required.json").read_text())
import torch
print(f"torch {torch.__version__} | numpy {np.__version__}")
fails = 0
frame = (np.random.RandomState(0).rand(640, 640, 3) * 255).astype("uint8")
for m in man["models"]:
    p = a.models_dir / m["local_path"]
    if m["local_path"].endswith(".pt"):
        t0 = time.time()
        try:
            from ultralytics import YOLO
            r = YOLO(str(p)).predict(frame, device="cpu", verbose=False)
            print(f"OK    {m['key']:18s} {p.stat().st_size/1e6:7.1f} MB  predict ok ({time.time()-t0:.1f}s)")
        except FileNotFoundError:
            print(f"MISSING {m['key']} ({p})"); fails += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {m['key']:18s} {type(exc).__name__}: {str(exc)[:120]}"); fails += 1
wd = a.models_dir / "whisper-base"
try:
    from faster_whisper import WhisperModel
    segs, info = WhisperModel(str(wd), device="cpu", compute_type="int8").transcribe(np.zeros(16000 * 3, dtype="float32"))
    list(segs); print(f"OK    whisper-base       loaded + transcribed 3 s of silence")
except Exception as exc:  # noqa: BLE001
    print(f"FAIL  whisper-base       {type(exc).__name__}: {str(exc)[:120]}"); fails += 1
sys.exit(1 if fails else 0)
