"""
check_dependencies.py
=====================
Checks that all required Python packages are importable.
Run before starting worker: python scripts/check_dependencies.py
"""
from __future__ import annotations

import importlib
import sys

BACKEND_DEPS = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("psycopg2", "psycopg2"),
    ("boto3", "boto3"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
    ("yaml", "pyyaml"),
]

WORKER_DEPS = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("ultralytics", "ultralytics"),
    ("faster_whisper", "faster-whisper"),
    ("openai", "openai"),
    ("google.genai", "google-genai"),
    ("boto3", "boto3"),
]


def check_list(deps: list[tuple[str, str]], label: str) -> bool:
    ok = True
    print(f"\n=== {label} ===")
    for import_name, pkg_name in deps:
        try:
            importlib.import_module(import_name)
            print(f"  OK  {pkg_name}")
        except ImportError:
            print(f"  MISSING  {pkg_name}")
            ok = False
    return ok


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if target in ("backend", "all"):
        ok &= check_list(BACKEND_DEPS, "Backend dependencies")
    if target in ("worker", "all"):
        ok &= check_list(WORKER_DEPS, "Worker dependencies")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
