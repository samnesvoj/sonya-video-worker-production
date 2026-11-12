#!/usr/bin/env python3
"""Remove files of the base image that no SONYA code path loads - versions and dist-info stay (lock, pip check, worker manifest unchanged).

  python3 tools/slim_site_packages.py /usr/local/lib/python3.11/site-packages

Fails the build (exit 1) when the assumption behind a removal does not hold, e.g. the active cv2 is not opencv-contrib-python.
"""
from __future__ import annotations

import base64
import hashlib
import shutil
import sys
from pathlib import Path

DEAD_DIRS = ("mediapipe", "onnx/backend/test")
CV2_OWNER = "opencv_contrib_python"
CV2_LIBS = ("opencv_python.libs", "opencv_python_headless.libs", "opencv_contrib_python.libs")


def record_hash(p: Path) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b"=").decode()


def cv2_owner(sp: Path) -> str:
    so = next((sp / "cv2").glob("cv2*.so"), None)
    if so is None:
        raise SystemExit("slim: cv2 binary not found")
    h = record_hash(so)
    owners = [rec.parent.name.split("-")[0] for rec in sp.glob("opencv_*-*.dist-info/RECORD") if h in rec.read_text()]
    if len(owners) != 1:
        raise SystemExit(f"slim: cannot tell which opencv wheel owns {so.name}: {owners}")
    return owners[0]


def slim(sp: Path) -> int:
    freed = 0

    def rm(p: Path) -> None:
        nonlocal freed
        if p.exists():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
            print(f"slim: removed {p.relative_to(sp)}")

    owner = cv2_owner(sp)
    if owner != CV2_OWNER:
        raise SystemExit(f"slim: active cv2 belongs to {owner}, expected {CV2_OWNER} (the Mac mini reference build) - not slimming")
    keep = f"{owner}.libs"
    if not (sp / keep).is_dir():
        raise SystemExit(f"slim: {keep} missing")
    for d in CV2_LIBS:
        if d != keep:
            rm(sp / d)
    for d in DEAD_DIRS:
        rm(sp / d)
    print(f"slim: freed {freed / 1e6:.0f} MB")
    return freed


if __name__ == "__main__":
    slim(Path(sys.argv[1]))
