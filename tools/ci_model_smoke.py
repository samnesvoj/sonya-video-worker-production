#!/usr/bin/env python3
"""
Runs INSIDE an image in CI (CPU only). Proves the dependency set actually RUNS the models SONYA uses - the part imports cannot show
(torch.load of a real checkpoint under the installed PyTorch, torchvision NMS, ctranslate2 loading a real Whisper model).
  1. downloads the PUBLIC models from release/models.required.json (pinned URL/revision) and verifies size + SHA-256
  2. YOLO11n and YOLO11n-pose: predict() on a synthetic frame
  3. faster-whisper base: load from a local dir (the way transcriber.py does) and transcribe 3 s of silence
The private webcam_detector.pt cannot be tested here. Not a GPU test.
"""
import hashlib, json, os, pathlib, sys, time, urllib.request

man = json.load(open(sys.argv[1]))
dest = pathlib.Path(os.environ.get("SMOKE_DIR", "/tmp/models"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")
rows, t_all = [], time.time()


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        rows.append((name, "ok", f"{time.time() - t0:.1f}s", str(out or "")[:90]))
    except Exception as exc:  # noqa: BLE001
        rows.append((name, "FAIL", f"{time.time() - t0:.1f}s", f"{type(exc).__name__}: {str(exc)[:150]}"))


def fetch(m):
    p = dest / m["local_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 7):          # GitHub release CDN returns transient 5xx; only the transport is retried, the hash check stays strict
        h, n = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(m["source"]["url"], timeout=120) as r, open(p, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk); h.update(chunk); n += len(chunk)
            break
        except OSError as exc:
            if attempt == 6:
                raise
            print(f"download retry {attempt}/6 for {m['key']}: {exc}", flush=True)
            time.sleep(min(10 * attempt, 40))
    assert n == m["public_size"], f"size {n} != {m['public_size']}"
    assert h.hexdigest() == m["public_sha256"], "sha256 mismatch"
    return f"{n/1e6:.1f} MB sha256 ok"


public = [m for m in man["models"] if m["source"]["type"] == "public" and m.get("public_sha256") and not m["key"].startswith("flux_")]  # flux_*: multi-GB, source url is a HF repo page (sha256 pinned in models.lock.json, verified at publish + on the worker)      # verifiable ones only (GigaAM: sha256 comes from the Mac mini)
for m in public:
    step("download+verify " + m["key"], lambda m=m: fetch(m))

import numpy as np
from ultralytics import YOLO

frame = (np.random.RandomState(0).rand(640, 640, 3) * 255).astype("uint8")
for key, fn in (("yolo11n", "common/yolo11n.pt"), ("yolo11n_pose", "common/yolo11n-pose.pt")):
    step("predict " + key, lambda fn=fn: f"{len(YOLO(str(dest / fn)).predict(frame, device='cpu', verbose=False)[0].boxes)} boxes on noise")


def whisper():
    from faster_whisper import WhisperModel
    m = WhisperModel(str(dest / "whisper-base"), device="cpu", compute_type="int8")
    segs, info = m.transcribe(np.zeros(16000 * 3, dtype="float32"), language="ru")
    return f"loaded; {len(list(segs))} segments on silence; lang={info.language}"


step("faster-whisper base load+transcribe", whisper)
import torch, torchvision
step("torchvision nms (compiled op)", lambda: torchvision.ops.nms(torch.tensor([[0., 0, 10, 10], [1, 1, 11, 11]]), torch.tensor([0.9, 0.8]), 0.5).tolist())
print("| step | result | time | detail |\n|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
print(f"\ntorch {torch.__version__} torchvision {torchvision.__version__} numpy {np.__version__}; total {time.time()-t_all:.0f}s (CPU, not GPU)")
sys.exit(1 if any(r[1] == "FAIL" for r in rows) else 0)
