"""
mode_policy.py - which modes this GPU worker may run.

Three tiers:
  HARD_DISABLED  never runnable (Trailer is OFF in SONYA production). No env/Template/job can enable it;
                 turning Trailer on is a code change + new release SHA.
  DEFAULT        enabled out of the box: modes with real-video evidence from the Mac mini work.
  OPT_IN         registered and tested, but NOT enabled until GPU/real-video proof exists;
                 enable per Template with ENABLED_MODES=... (explicit decision).
ENABLED_MODES (comma list) selects from DEFAULT+OPT_IN; it can never add a hard-disabled mode.
"""
from __future__ import annotations

import os
from typing import List, Optional

HARD_DISABLED_MODES = frozenset({"trailer_film_breaker", "trailer_cinematic"})
HARD_DISABLED_PREFIXES = ("trailer",)
DEFAULT_ENABLED_MODES = ("virality", "beep")
OPT_IN_MODES = ("stories", "educational", "streamer", "sonya_gen")
DISABLED_CODE = "mode_disabled"


class ModeDisabledError(RuntimeError):
    code = DISABLED_CODE


def is_hard_disabled(mode: Optional[str]) -> bool:
    m = (mode or "").strip().lower()
    return m in HARD_DISABLED_MODES or m.startswith(HARD_DISABLED_PREFIXES)


def enabled_modes() -> List[str]:
    raw = os.environ.get("ENABLED_MODES", "").strip()
    wanted = [m.strip().lower() for m in raw.split(",") if m.strip()] if raw else list(DEFAULT_ENABLED_MODES)
    known = set(DEFAULT_ENABLED_MODES) | set(OPT_IN_MODES)
    return [m for m in dict.fromkeys(wanted) if m in known and not is_hard_disabled(m)]


def claimable_modes() -> List[str]:
    """Allow-list for the backend claim call (never empty: an empty filter would mean 'any mode')."""
    modes = enabled_modes()
    if not modes:
        raise ModeDisabledError("ENABLED_MODES leaves no runnable mode")
    return modes


def assert_mode_allowed(mode: Optional[str]) -> str:
    m = (mode or "").strip().lower()
    if is_hard_disabled(m):
        raise ModeDisabledError(f"mode {m!r} is disabled in this release")
    if m not in enabled_modes():
        raise ModeDisabledError(f"mode {m!r} is not enabled on this worker")
    return m
