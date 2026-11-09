"""
nvenc_guard.py - run one ffmpeg ENCODE so that the NVENC session cap of GeForce drivers never fails a job.

GeForce drivers allow only a few simultaneous NVENC sessions per system (scripts/ops/gpu_plan.nvenc_session_limit: 3 / 5 / 8
by driver; data-center cards: no cap). A server runs N jobs x SONYA_CLIP_WORKERS clip renders at once, so on a 2x RTX 3090 box
8 jobs would ask for 16 sessions. run():

  * command without h264_nvenc / hevc_nvenc            -> subprocess.run unchanged
  * no cap (SONYA_NVENC_MAX_SESSIONS unset or 0)       -> NVENC as before
  * cap set: takes one of the cap's slots (flock on $SONYA_NVENC_SLOT_DIR/slot-<k>.lock, shared by every worker process of
    the server, released by the kernel even if the process dies); no slot free -> THIS encode runs with libx264 instead
  * NVENC refused anyway (session cap of another container on the same host, driver error) -> the same encode again on the CPU

The CPU args are exactly the Desktop "quality" CPU args (scripts/shared/cut_profile.encoder_args("cpu", same bitrate)), i.e. the
fallback every server without NVENC already uses; moments, crop and subtitles do not depend on the encoder.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

NVENC_CODECS = ("h264_nvenc", "hevc_nvenc")
# options only h264_nvenc understands (value follows); dropped when the encode moves to libx264
NVENC_ONLY = {"-rc", "-cq", "-tune", "-multipass", "-rc-lookahead", "-spatial-aq", "-spatial_aq", "-temporal-aq",
              "-temporal_aq", "-aq-strength", "-gpu", "-b_ref_mode", "-zerolatency", "-surfaces", "-delay"}
# stderr of an NVENC encode that did not run for an NVENC reason (cap, driver, no libnvidia-encode, an ffmpeg built without
# NVENC: "Unknown encoder" / its options unknown) -> the same encode on the CPU. Anything else (bad input) is a real error.
REFUSED = re.compile(r"nvenc|NV_ENC|OpenEncodeSession|incompatible client key|No capable devices|CUDA|cuda|"
                     r"out of memory|Cannot load libnvidia-encode|Unknown encoder|Unrecognized option", re.I)


def max_sessions() -> int:
    try:
        return max(0, int(os.environ.get("SONYA_NVENC_MAX_SESSIONS") or 0))
    except ValueError:
        return 0


def uses_nvenc(cmd: Sequence[str]) -> bool:
    return any(str(a) in NVENC_CODECS for a in cmd)


def to_cpu(cmd: Sequence[str]) -> List[str]:
    """The same ffmpeg command with the NVENC encoder replaced by the Desktop CPU encoder (libx264 slow crf 19, same
    bitrate cap; "-b:v 0" = NVENC's uncapped quality mode -> no cap)."""
    from scripts.shared.cut_profile import encoder_args
    src = [str(a) for a in cmd]
    out: List[str] = []
    at: Optional[int] = None
    bitrate: Optional[str] = None
    i = 0
    while i < len(src):
        a = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else None
        if a in ("-c:v", "-vcodec", "-codec:v") and nxt in NVENC_CODECS:
            at = len(out) if at is None else at
            i += 2
            continue
        if a in NVENC_ONLY and nxt is not None:
            i += 2
            continue
        if a == "-preset" and nxt is not None and re.fullmatch(r"p[1-7]|slow|medium|fast|hq|hp|ll|llhq|llhp|default", nxt):
            i += 2                                   # the libx264 preset is set below
            continue
        if a == "-b:v" and nxt is not None:
            bitrate = None if nxt == "0" else nxt
            i += 2
            continue
        out.append(a)
        i += 1
    if at is None:
        return src
    cpu = [x for x in encoder_args("cpu", bitrate) if x not in ("-pix_fmt", "yuv420p")]
    if "-pix_fmt" not in out:
        cpu += ["-pix_fmt", "yuv420p"]
    out[at:at] = cpu
    if "-y" not in out and "-n" not in out:
        out.insert(1, "-y")                          # a refused NVENC run may have left a partial output file
    return out


@contextlib.contextmanager
def session(limit: Optional[int] = None) -> Iterator[bool]:
    """True = an NVENC slot is ours for the duration (or there is no cap), False = all slots busy."""
    n = max_sessions() if limit is None else limit
    if n <= 0:
        yield True
        return
    import fcntl
    d = Path(os.environ.get("SONYA_NVENC_SLOT_DIR") or Path(os.environ.get("SONYA_STATE_DIR", "/run/sonya")) / "nvenc")
    fd = None
    try:
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            f = os.open(str(d / f"slot-{k}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fd = f
                break
            except OSError:
                os.close(f)
    except OSError as exc:            # slot dir not writable: behave as before (NVENC; a refusal still falls back below)
        logger.warning("[nvenc] slot dir unusable (%s): no session accounting", type(exc).__name__)
        yield True
        return
    try:
        yield fd is not None
    finally:
        if fd is not None:
            os.close(fd)              # closing releases the flock


def run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    """subprocess.run(cmd, **kw) for an ffmpeg encode, NVENC-session safe (see module doc)."""
    if not uses_nvenc(cmd):
        return subprocess.run(list(cmd), **kw)
    with session() as got:
        if got:
            r = subprocess.run(list(cmd), **kw)
            err = r.stderr if isinstance(r.stderr, str) else (r.stderr or b"").decode("utf-8", "replace")
            if r.returncode == 0 or not REFUSED.search(err or ""):
                return r
            logger.warning("[nvenc] encode refused on the GPU (%s) - this clip is encoded on the CPU",
                           (err or "").strip().splitlines()[-1:] or "no stderr")
        else:
            logger.info("[nvenc] all %d NVENC sessions busy - this clip is encoded on the CPU", max_sessions())
    return subprocess.run(to_cpu(cmd), **kw)
