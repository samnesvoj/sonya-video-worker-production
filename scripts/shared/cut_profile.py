"""
scripts/shared/cut_profile.py
=============================
The Mac mini reference recipe (SONYA Desktop, docs CUT_PLAYBOOK) for production virality jobs.

The production site sends only what the user picked (``clipType``, ``settings.subtitleStyle``…), never the
Cut settings (``selection`` / ``crop``). This module turns every such job into the recipe the owner approved
on the Mac mini (the virality runner runs nothing else):

  * selection ``story`` (cut-local-rank-v3: true sentences, question → whole answer, promo never a clip),
    strategy ``qa`` (``hooks`` when the user picked Hooks); the LLM editor (Gemini via Timeweb / OpenRouter)
    re-ranks the pool when a provider key is configured, else the local order is used;
  * crop ``tracked`` / ``per_clip`` / ``smart``: per clip cuts + YuNet faces, full-screen 9:16, split for two;
  * subtitles burned (ASS), font Dela Gothic One for the Viral style, never a hook title on top.

  * ASR is mandatory (Desktop: «Cut без неё не запускается»): no words → the job fails with ASR_FAILED,
    never the old no-words path;
  * render: the Desktop encoder args (NVENC p6 / libx264 slow, 6 Mb/s cap) via SONYA_RENDER_ARGS.

The old virality path (whole-video YOLO + gemini_composer + hook_mode) was removed 2026-11-08 (owner:
«старое это не надо»); full copy: git tag old-worker-way + /Users/agent/SONYA/archive/old-worker-way-2026-11-08.tar.gz
(archive/legacy_virality/README.md). A job that already carries Desktop settings (selection=story +
crop.mode=tracked) is left as is.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROFILE_VERSION = "cut-profile-macmini-v1"

# site subtitle style (index.html: Viral «Яркие, анимированные» / Minimal «Элегантные» / Classic «Белые внизу»)
#   → (subtitle_styles preset, font id or None = the preset's own font)
SITE_STYLES = {
    "viral": ("highlight", "dela"),      # Desktop default: per-word highlight, owner's font pick
    "minimal": ("champagne", None),      # noble pearl / champagne, readable on any footage
    "basic": ("minimal", None),          # plain white, bottom
}
DEFAULT_STYLE = "viral"                  # job without settings (API / tests) → the reference look


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_has_ass(ffmpeg: Optional[str] = None) -> bool:
    """libass in this ffmpeg (needed to burn subtitles)."""
    try:
        out = subprocess.run([ffmpeg or os.environ.get("SONYA_FFMPEG") or "ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:            # noqa: BLE001 - no ffmpeg → the render fails later with a clear error
        return False
    return any(line.split()[1:2] == ["ass"] for line in out.splitlines() if len(line.split()) > 1)


def editor_available(env: Optional[Dict[str, str]] = None) -> bool:
    env = dict(os.environ) if env is None else env
    if str(env.get("SONYA_LLM_EDITOR", "auto")).lower() in ("0", "off", "false", "no"):
        return False
    from scripts.shared import llm_provider
    return bool(llm_provider.configured_providers(env))


def applies(mode: str, params: Dict[str, Any]) -> bool:
    """Every production virality job runs the Mac mini recipe (owner 2026-11-08: «старое это не надо»).
    Left untouched only: a job that already carries the Desktop settings itself (selection=story or
    crop.mode=tracked: tests / manual runs)."""
    if mode != "virality":
        return False
    crop = params.get("crop") if isinstance(params.get("crop"), dict) else {}
    if params.get("selection") == "story" and crop.get("mode") == "tracked" and params.get("llm") != "on":
        return False
    if "selection" in params or "crop" in params:
        logger.warning("[cut_profile] legacy Cut settings in the job (selection=%r crop=%r) are ignored: "
                       "the Mac mini recipe replaces the old path", params.get("selection"), crop.get("mode"))
    return True


# ── render encoder (Desktop: sonya_desktop/media/encoders.py, quality preset, 6 Mb/s) ──────────────────
# Same args as the Desktop probe picks: the 6 Mb/s cap keeps a 60 s clip ≈ 45 MB (Telegram's 50 MB limit,
# CUT_PLAYBOOK). NVENC on Vast; libx264 when there is no working NVENC (CPU box, broken driver).
RENDER_BITRATE = "6M"


def encoder_args(family: str, bitrate: Optional[str] = RENDER_BITRATE) -> list:
    """sonya_desktop.media.encoders.encoder_args(family, "h264", "quality", bitrate) for nvenc / cpu."""
    if family == "nvenc":
        args = ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "20"]
        args += ["-b:v", bitrate] if bitrate else ["-b:v", "0"]
    elif family == "cpu":
        args = ["-c:v", "libx264", "-preset", "slow", "-crf", "19"] + (["-b:v", bitrate] if bitrate else [])
    else:
        raise ValueError(f"unsupported encoder family {family}")
    return args + ["-pix_fmt", "yuv420p"]


def encode_timeout(duration_s: float) -> float:
    """Watchdog for ONE encode: a hung ffmpeg must not block a job forever (L40S 2026-11-09: one NVENC render waited
    11+ min at CPU 0). Generous: libx264 slow of a 60 s 1080x1920 clip on a busy CPU takes ~1-3 min."""
    return max(float(os.environ.get("SONYA_RENDER_TIMEOUT_MIN_S", "300")), 10.0 * max(0.0, duration_s))


def run_encode(make_cmd, args: list, *, timeout: Optional[float] = None, cwd: Optional[str] = None, what: str = "render"):
    """Run ``make_cmd(args)``. When the NVENC encode fails or hangs (GeForce: at most 8 NVENC sessions at once;
    L40S: cuCtxCreate CUDA_ERROR_OPERATING_SYSTEM with 6 jobs), the SAME command is re-run once with the libx264
    args instead of dropping the clip. -> (CompletedProcess, "nvenc" | "cpu" | "nvenc->cpu", seconds)."""
    t0 = time.monotonic()
    nvenc = "h264_nvenc" in args

    def once(a):
        try:
            return subprocess.run(make_cmd(a), capture_output=True, text=True, cwd=cwd, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(make_cmd(a), 124, "", f"timeout after {timeout:.0f} s")

    r = once(args)
    used = "nvenc" if nvenc else "cpu"
    if r.returncode != 0 and nvenc:
        logger.warning("[cut_profile] %s: NVENC failed (%s) - re-encoding on CPU (libx264)", what, (r.stderr or "")[-200:].strip())
        bitrate = args[args.index("-b:v") + 1] if "-b:v" in args and args[args.index("-b:v") + 1] != "0" else None
        r, used = once(encoder_args("cpu", bitrate)), "nvenc->cpu"
    return r, used, round(time.monotonic() - t0, 1)


def _encoder_works(ffmpeg: str, family: str) -> bool:
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi",
           "-i", "testsrc2=size=320x240:rate=30:duration=0.5", *encoder_args(family), "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except Exception:            # noqa: BLE001 - no ffmpeg / timeout → not usable
        return False


_FAMILY: Dict[str, str] = {}


def encoder_family(ffmpeg: Optional[str] = None) -> str:
    """"nvenc" when a 0.5 s NVENC smoke encode works (once per process), else "cpu"."""
    ff = ffmpeg or os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    if ff not in _FAMILY:
        _FAMILY[ff] = "nvenc" if _encoder_works(ff, "nvenc") else "cpu"
    return _FAMILY[ff]


def ensure_render_args(ffmpeg: Optional[str] = None) -> Optional[str]:
    """Set SONYA_RENDER_ARGS (read by smart_crop_adapter._env_encoder_args) once per process, like the
    Desktop daemon does before every Cut run. An explicit SONYA_RENDER_ARGS wins. -> chosen family."""
    import json
    if os.environ.get("SONYA_RENDER_ARGS"):
        return None
    family = encoder_family(ffmpeg)
    os.environ["SONYA_RENDER_ARGS"] = json.dumps(encoder_args(family))
    logger.info("[cut_profile] render encoder: %s @ %s", family, RENDER_BITRATE)
    return family


def apply_profile(mode: str, params: Dict[str, Any], input_video_path: str, work_dir: str,
                  *, ass_supported: Optional[bool] = None, editor: Optional[bool] = None) -> Dict[str, Any]:
    """New params dict with the Mac mini recipe (or ``params`` unchanged when it does not apply)."""
    if not applies(mode, params):
        return params
    out = dict(params)
    clip_type = str(params.get("clipType") or "").lower()
    settings = params.get("settings") if isinstance(params.get("settings"), dict) else {}
    story = {"strategy": "hooks" if clip_type == "hooks" else "qa"}
    if editor if editor is not None else editor_available():
        story["editor"] = True
    out["selection"] = "story"
    out["story"] = dict(params.get("story") or {}, **story)
    crop: Dict[str, Any] = {"mode": "tracked", "analysis": "per_clip", "layout": "smart",
                            "source_sha256": _sha256(input_video_path),
                            "clip_cache_dir": str(Path(work_dir) / "vision" / "clips")}
    style = str(settings.get("subtitleStyle") or DEFAULT_STYLE).lower()
    if style not in ("none", "off"):
        preset, font = SITE_STYLES.get(style, SITE_STYLES[DEFAULT_STYLE])
        ok = ffmpeg_has_ass() if ass_supported is None else ass_supported
        crop["subtitles"] = {"mode": "burn", "burn_supported": ok, "style": preset, "hook": False,
                             **({"font": font} if font else {})}
        if not ok:
            logger.warning("[cut_profile] ffmpeg has no libass - subtitles cannot be burned")
    out["crop"] = crop
    out["cut_profile"] = PROFILE_VERSION
    out.pop("llm", None)             # legacy Gemini selection switch: the story selector has its own editor
    try:
        ensure_render_args()
    except Exception as exc:         # noqa: BLE001 - the renderer's libx264 default still works
        logger.warning("[cut_profile] encoder probe failed: %s", exc)
    logger.info("[cut_profile] %s: strategy=%s editor=%s subtitles=%s", PROFILE_VERSION, story["strategy"],
                bool(story.get("editor")), (crop.get("subtitles") or {}).get("style", "off"))
    return out
