"""
modes/sonya_gen/runner.py - SONYA Gen (images) on the GPU worker.

FLUX.1-schnell Q4_K_S (Apache-2.0) via a private headless ComfyUI + ComfyUI-GGUF (pinned, sha256-checked tarballs, installed once into a
venv that sees the worker's torch). Same workflow as the Desktop Gen (modes/sonya_gen/workflow.py). Output: gen_<n>.png + gen_meta.json.
No Gemini here (Gemini is analysis only). No input video: input_video_path is ignored.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import translate
from . import workflow as wf

logger = logging.getLogger(__name__)

COMFY_URL = "https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/v0.39.2.tar.gz"
COMFY_SHA256 = "1d655e22cf546700fc1d6d2536e27701695c5832a51f63bb16c41866ac09be0b"
COMFY_DIRNAME = "ComfyUI-0.39.2"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
GGUF_URL = f"https://github.com/city96/ComfyUI-GGUF/archive/{GGUF_COMMIT}.tar.gz"
GGUF_SHA256 = "688126d5c2b8c4ff061f56b23c3c6791ac01dc366e98d41ef1aaef374c36f283"

MAX_PROMPT = 1000
MAX_COUNT = 4
START_TIMEOUT = 300
JOB_TIMEOUT = 600


class GenError(RuntimeError):
    pass


def _root() -> Path:
    r = os.environ.get("SONYA_COMFY_DIR", "/work/comfy")
    p = Path(r)
    try:
        p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            raise OSError
    except OSError:
        p = Path("/tmp/sonya-comfy")
        p.mkdir(parents=True, exist_ok=True)
    return p


def _models_dir() -> Path:
    return Path(os.environ.get("MODELS_LOCAL_DIR", "/opt/sonya/models"))


def _fetch_tar(url: str, sha: str, dest: Path, top: str) -> Path:
    tgz = dest / (top + ".tgz")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as r, open(tgz, "wb") as f:
        for b in iter(lambda: r.read(1 << 20), b""):
            h.update(b)
            f.write(b)
    if h.hexdigest() != sha:
        tgz.unlink(missing_ok=True)
        raise GenError(f"sha256 mismatch for {url}")
    with tarfile.open(tgz) as t:
        t.extractall(dest, filter="data")
    tgz.unlink(missing_ok=True)
    return dest / top


def ensure_comfy() -> tuple[Path, str]:
    """Install ComfyUI + ComfyUI-GGUF once (venv with --system-site-packages: the worker's CUDA torch is reused). Returns (dir, python)."""
    root = _root()
    ok = root / ".installed-v1"
    cdir = root / COMFY_DIRNAME
    py = str(root / "venv" / "bin" / "python")
    if ok.exists() and cdir.is_dir() and os.path.exists(py):
        return cdir, py
    logger.info("[sonya_gen] installing ComfyUI into %s", root)
    shutil.rmtree(cdir, ignore_errors=True)
    shutil.rmtree(root / "venv", ignore_errors=True)
    _fetch_tar(COMFY_URL, COMFY_SHA256, root, COMFY_DIRNAME)
    g = _fetch_tar(GGUF_URL, GGUF_SHA256, root, f"ComfyUI-GGUF-{GGUF_COMMIT}")
    shutil.rmtree(cdir / "custom_nodes" / "ComfyUI-GGUF", ignore_errors=True)
    shutil.move(str(g), str(cdir / "custom_nodes" / "ComfyUI-GGUF"))
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(root / "venv")], check=True)
    for req in (cdir / "requirements.txt", cdir / "custom_nodes" / "ComfyUI-GGUF" / "requirements.txt"):
        r = subprocess.run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", str(req)],
                           capture_output=True, text=True)
        if r.returncode:
            raise GenError("ComfyUI deps install failed: " + (r.stderr or "")[-300:])
    ok.write_text("ok")
    return cdir, py


def _link_models(cdir: Path) -> None:
    """models are published as flux/<file>; ComfyUI wants them under unet/, text_encoders/, vae/ (symlinks, nothing copied)."""
    src = _models_dir() / "flux"
    for sub, name in (("unet", wf.MODEL["files"]["unet"][0]), ("text_encoders", wf.MODEL["files"]["t5"][0]),
                      ("text_encoders", wf.MODEL["files"]["clip_l"][0]), ("vae", wf.MODEL["files"]["vae"][0])):
        s = src / name
        if not s.is_file():
            raise GenError(f"model file missing: {name}")
        d = cdir / "models" / sub
        d.mkdir(parents=True, exist_ok=True)
        t = d / name
        if t.is_symlink() or t.exists():
            t.unlink()
        t.symlink_to(s)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(base: str, path: str, data: Optional[dict] = None, timeout: int = 30) -> bytes:
    req = urllib.request.Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _generate_one(base: str, prompt: str, seed: int, w: int, h: int, steps: int) -> bytes:
    graph = wf.build(prompt, seed, w, h, steps, "sonya_gen")
    pid = json.loads(_http(base, "/prompt", {"prompt": graph}))["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < JOB_TIMEOUT:
        hist = json.loads(_http(base, f"/history/{pid}"))
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                raise GenError("ComfyUI reported an error")
            for out in hist[pid].get("outputs", {}).values():
                for im in out.get("images", []):
                    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                    return _http(base, "/view?" + q, timeout=60)
            raise GenError("ComfyUI produced no image")
        time.sleep(0.5)
    raise GenError("generation timed out")


def run(input_video_path: str = "", output_dir: str = "", params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    params = params or {}
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise GenError("prompt is empty")
    if len(prompt) > MAX_PROMPT:
        raise GenError(f"prompt is longer than {MAX_PROMPT} characters")
    aspect = str(params.get("aspect") or "1:1")
    size = str(params.get("size") or "standard")
    try:
        w, h = wf.dims(aspect, size)
        count = max(1, min(MAX_COUNT, int(params.get("count") or 1)))
        steps = max(1, min(8, int(params.get("steps") or wf.MODEL["default_steps"])))
        seed0 = int(params["seed"]) if params.get("seed") is not None else random.SystemRandom().randrange(2**31)
    except (ValueError, TypeError) as e:
        raise GenError(f"bad parameters: {e}") from e
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_prompt, warn = translate.to_english(prompt)

    def prog(p: float, msg: str) -> None:
        if progress_callback:
            progress_callback(p, msg)

    prog(0.05, "comfy: install")
    # one generation per server at a time (two ComfyUI + FLUX would not fit on a 12-24 GB card)
    lock = open(_root() / "gen.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        return _run_locked(out, prompt, gen_prompt, warn, aspect, size, w, h, steps, count, seed0, prog)
    finally:
        lock.close()


def _run_locked(out: Path, prompt: str, gen_prompt: str, warn, aspect: str, size: str, w: int, h: int, steps: int, count: int, seed0: int, prog) -> Dict[str, Any]:
    cdir, py = ensure_comfy()
    _link_models(cdir)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(out.parent / "comfy.log", "ab")
    proc = subprocess.Popen([py, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch", "--dont-print-server"],
                            cwd=str(cdir), stdout=log, stderr=subprocess.STDOUT)
    files = []
    try:
        t0 = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise GenError("ComfyUI exited at start")
            try:
                _http(base, "/system_stats", timeout=3)
                break
            except Exception:
                if time.monotonic() - t0 > START_TIMEOUT:
                    raise GenError("ComfyUI did not start")
                time.sleep(1)
        prog(0.2, "comfy: ready")
        seeds = []
        for i in range(count):
            seed = (seed0 + i) % (2**31)
            png = _generate_one(base, gen_prompt, seed, w, h, steps)
            if len(png) < 1000 or png[:8] != b"\x89PNG\r\n\x1a\n":
                raise GenError("generated file is not a PNG")
            p = out / f"gen_{i + 1}.png"
            p.write_bytes(png)
            files.append(str(p))
            seeds.append(seed)
            prog(0.2 + 0.75 * (i + 1) / count, f"image {i + 1}/{count}")
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    meta = {"mode": "sonya_gen", "ai_generated": True, "model": wf.MODEL["id"], "license": wf.MODEL["license"],
            "workflow": wf.WORKFLOW_VERSION, "aspect": aspect, "size": size, "width": w, "height": h, "steps": steps,
            "seeds": seeds, "prompt": prompt, "negative_applied": False}
    if gen_prompt != prompt:
        meta["prompt_en"] = gen_prompt
    if warn:
        meta["warning"] = warn
    (out / "gen_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"clips": files, "mode": "sonya_gen", "status": "ok", "meta": meta}
