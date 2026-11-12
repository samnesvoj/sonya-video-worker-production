#!/usr/bin/env python3
"""Runs INSIDE the image (CI, CPU): wall time of importing each heavy dependency in a fresh interpreter (min of 3) and of reading the
package tree. Page cache is warm after the first run, so this is a LOWER bound for a cold Vast host. No GPU involved."""
import subprocess, sys, time

MODS = ["numpy", "cv2", "torch", "torchvision", "torchaudio", "onnxruntime", "ultralytics", "mediapipe", "faster_whisper", "scenedetect", "boto3", "yaml", "fastapi", "openai", "google.genai"]
print("| module | import seconds (min of 3) |\n|---|---|")
tot = 0.0
for m in MODS:
    best = min((lambda t0: (subprocess.run([sys.executable, "-c", f"import {m}"], capture_output=True), time.time() - t0)[1])(time.time()) for _ in range(3))
    tot += best
    print(f"| {m} | {best:.2f} |")
t0 = time.time()
subprocess.run([sys.executable, "-c", "import numpy, cv2, torch, torchvision, ultralytics, mediapipe, faster_whisper, scenedetect, boto3, yaml"], capture_output=True)
print(f"\nAll heavy imports together in one process: {time.time()-t0:.2f} s (sum of separate: {tot:.2f} s)")
