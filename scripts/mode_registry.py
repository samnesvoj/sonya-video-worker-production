"""
mode_registry.py
================
Discovers and returns runner callables for each production mode.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional

from scripts.mode_policy import assert_mode_allowed

_MODES_DIR = Path(__file__).parent.parent / "modes"

_REGISTRY: Dict[str, str] = {
    "trailer_film_breaker": "modes.trailer_film_breaker.runner",
    "virality": "modes.virality.runner",
    "stories": "modes.stories.runner",
    "educational": "modes.educational.runner",
    "streamer": "modes.streamer.runner",
    "sonya_gen": "modes.sonya_gen.runner",
    "beep": "modes.beep.runner",
}


def get_runner(mode: str) -> Callable:
    """Return the run() callable for the given mode name."""
    mode = assert_mode_allowed(mode)  # Trailer and non-enabled modes stop here
    module_path = _REGISTRY.get(mode)
    if not module_path:
        raise ValueError(f"Unknown mode: {mode!r}. Available: {list(_REGISTRY)}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import runner for mode {mode!r}: {e}") from e
    if not hasattr(module, "run"):
        raise AttributeError(f"Runner module {module_path} has no run() function")
    return module.run


def list_modes() -> list[str]:
    return list(_REGISTRY.keys())
