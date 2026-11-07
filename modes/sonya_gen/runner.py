"""
modes/sonya_gen/runner.py
==========================
Placeholder — not production.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def run(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    logger.warning("[sonya_gen] This mode is a placeholder and not production-ready.")
    return {"clips": [], "mode": "sonya_gen", "status": "placeholder"}
