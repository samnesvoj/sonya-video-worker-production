"""
worker_manifest.py
==================
What exactly ran: one JSON that makes a worker run reproducible.

  commit       SONYA_GIT_SHA (baked into the image at build time), else `git rev-parse` locally
  image        SONYA_IMAGE (the runner passes the digest-pinned ref it started) + SONYA_IMAGE_DIGEST
  runtime      Python, platform, torch / CUDA / cuDNN / GPU, FFmpeg, OpenCV, numpy, key packages
  deps         sha256 of the sorted `name==version` list of every installed distribution
  modes        version constants of the pipeline stages (selection, boundaries, crop, subtitles, editor)
  models       model files under SONYA_MODELS_DIR / ./models: name, size, sha256 of the first 16 MB + size
  env          SONYA_* / WORKER_* / APP_* / MODE flags — values of anything secret-looking are never read

Never fails: a missing piece is reported as null with a reason.

CLI:  python -m scripts.shared.worker_manifest [--out FILE]
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_VERSION = "worker-manifest-v1"
ROOT = Path(__file__).resolve().parents[2]

_ENV_PREFIXES = ("SONYA_", "WORKER_", "APP_", "VAST_DEBUG")
_ENV_EXACT = {"MODE", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
_SECRET = re.compile(r"(?i)(SECRET|TOKEN|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTH|COOKIE|DSN|DATABASE_URL)")
_PACKAGES = ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless", "opencv-contrib-python",
             "ultralytics", "mediapipe", "faster-whisper", "ctranslate2", "onnxruntime", "scenedetect",
             "pillow", "openai", "google-genai")
_MODE_VERSIONS = {            # module: attribute names holding a stage version
    "scripts.shared.cut_selection": ("CANDIDATES_VERSION", "LOCAL_RANK_V3", "RANKING_VERSION", "PROMPT_VERSION"),
    "scripts.shared.boundaries": ("VERSION",),
    "scripts.shared.crop.smart_crop_adapter": ("CROP_VERSION", "CROP_VERSION_V2", "FIT_VERSION"),
    "scripts.shared.subtitles": ("SUBTITLES_VERSION",),
    "scripts.shared.llm_editor": ("PROMPT_VERSION",),
}


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _commit() -> Dict[str, Any]:
    sha = os.environ.get("SONYA_GIT_SHA")
    if sha and sha != "unknown":                          # Dockerfile default when no build arg was given
        return {"sha": sha, "source": "image"}
    sha = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    return {"sha": sha, "source": "git" if sha else None, "dirty": bool(dirty) if sha else None}


def _image() -> Dict[str, Any]:
    ref = os.environ.get("SONYA_IMAGE")
    digest = os.environ.get("SONYA_IMAGE_DIGEST")
    if not digest and ref and "@sha256:" in ref:
        digest = ref.split("@", 1)[1]
    return {"ref": ref, "digest": digest, "pinned": bool(digest)}


def _versions() -> Dict[str, Optional[str]]:
    try:
        from importlib import metadata
    except ImportError:                                   # pragma: no cover
        return {}
    out: Dict[str, Optional[str]] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _deps_hash() -> Dict[str, Any]:
    try:
        from importlib import metadata
        pins = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in metadata.distributions()
                       if d.metadata["Name"]})
    except Exception as e:                                 # broken dist metadata: report, don't fail
        return {"sha256": None, "count": None, "error": type(e).__name__}
    return {"sha256": hashlib.sha256("\n".join(pins).encode()).hexdigest(), "count": len(pins)}


def _torch() -> Dict[str, Any]:
    try:
        import torch
    except Exception as e:
        return {"available": False, "reason": type(e).__name__}
    info: Dict[str, Any] = {"available": True, "version": torch.__version__, "cuda": torch.version.cuda,
                            "cuda_available": bool(torch.cuda.is_available())}
    try:
        info["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    return info


def _ffmpeg() -> Dict[str, Any]:
    exe = os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    first = (_run([exe, "-hide_banner", "-version"]) or "").splitlines()[:1]
    return {"path": exe, "version": first[0] if first else None}


def _opencv() -> Dict[str, Any]:
    try:
        import cv2
    except Exception as e:
        return {"version": None, "reason": type(e).__name__}
    haar = Path(getattr(cv2, "data", None).haarcascades) if getattr(cv2, "data", None) else None
    return {"version": cv2.__version__,
            "haar_frontalface": bool(haar and (haar / "haarcascade_frontalface_default.xml").is_file())}


def _modes() -> Dict[str, Any]:
    import importlib
    out: Dict[str, Any] = {}
    for mod, attrs in _MODE_VERSIONS.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            out[mod.rsplit(".", 1)[1]] = {"error": type(e).__name__}
            continue
        vals = {a: getattr(m, a) for a in attrs if isinstance(getattr(m, a, None), str)}
        if vals:
            out[mod.rsplit(".", 1)[1]] = vals
    return out


def _file_id(p: Path) -> str:
    """sha256 of size + the first 16 MB: stable, cheap even for multi-GB weights."""
    h = hashlib.sha256(str(p.stat().st_size).encode())
    with p.open("rb") as f:
        h.update(f.read(16 * 1024 * 1024))
    return h.hexdigest()[:32]


def _models(limit: int = 200) -> Dict[str, Any]:
    roots = [Path(os.environ["SONYA_MODELS_DIR"])] if os.environ.get("SONYA_MODELS_DIR") else []
    roots += [ROOT / "models", Path("/opt/sonya/models")]
    seen, files = set(), []
    for root in roots:
        if not root.is_dir() or root.resolve() in seen:
            continue
        seen.add(root.resolve())
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".pt", ".pth", ".onnx", ".safetensors", ".bin", ".ckpt",
                                                    ".gguf", ".tflite", ".task", ".engine", ".model"):
                try:
                    files.append({"path": str(p.relative_to(root)), "root": str(root),
                                  "size": p.stat().st_size, "id": _file_id(p)})
                except OSError:
                    continue
                if len(files) >= limit:
                    return {"files": files, "truncated": True}
    return {"files": files, "truncated": False}


def _env() -> Dict[str, str]:
    out = {}
    for k, v in os.environ.items():
        if not (k in _ENV_EXACT or k.startswith(_ENV_PREFIXES)):
            continue
        out[k] = "***" if _SECRET.search(k) else v[:200]
    return dict(sorted(out.items()))


def build_manifest(include_models: bool = True) -> Dict[str, Any]:
    t0 = time.monotonic()
    m = {"manifest_version": MANIFEST_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "commit": _commit(), "image": _image(),
         "runtime": {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
                     "platform": platform.platform(), "machine": platform.machine(),
                     "torch": _torch(), "ffmpeg": _ffmpeg(), "opencv": _opencv(), "packages": _versions()},
         "deps": _deps_hash(), "modes": _modes(),
         "models": _models() if include_models else None, "env": _env()}
    m["build_ms"] = int((time.monotonic() - t0) * 1000)
    return m


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-models", action="store_true")
    a = ap.parse_args(argv)
    text = json.dumps(build_manifest(include_models=not a.no_models), ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
