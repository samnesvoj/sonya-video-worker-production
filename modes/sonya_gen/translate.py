"""FLUX.1-schnell understands English only: a non-Latin prompt is translated to an English image prompt (text-only LLM call,
cents). Any failure -> the original prompt is used and a warning is returned (never an error)."""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from scripts.shared import llm_provider as lp

logger = logging.getLogger(__name__)
SUFFIX = ", no text, no logos"
TIMEOUT = 10.0
_SYSTEM = ("You translate image-generation prompts into English for the FLUX text-to-image model. Keep EVERY detail "
           "(subjects, colours, setting, style, light, camera). Do not add logos, text, brands or new objects. "
           'Reply with JSON only: {"prompt": "<english prompt>"}.')


def needs_translation(prompt: str) -> bool:
    return any(c.isalpha() and not ("a" <= c.lower() <= "z") for c in prompt)


def to_english(prompt: str, *, providers=None, transport=None) -> Tuple[str, Optional[str]]:
    """-> (prompt_for_model, warning|None)."""
    if not needs_translation(prompt):
        return prompt, None
    try:
        provs = providers
        if provs is None:
            p = lp.timeweb_provider()
            provs = [p] if p else lp.configured_providers()
        if not provs:
            raise lp.LLMDisabled("no provider")
        res = lp.chat_json([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                           prompt_version="gen-translate-v1", opt_in=True, providers=provs,
                           transport=transport or lp.certifi_transport(), timeout=TIMEOUT,
                           extra={"max_tokens": 400}, max_request_bytes=20_000)
        en = str(json.loads(res.text).get("prompt") or "").strip().rstrip(".")
        if not en or needs_translation(en):
            raise ValueError("empty or non-English answer")
        en += SUFFIX
        logger.info("[sonya_gen] prompt translated: %r -> %r", prompt[:300], en[:300])
        return en, None
    except Exception as e:  # noqa: BLE001 — translation must never fail the job
        logger.warning("[sonya_gen] prompt translation failed (%s); using the original", type(e).__name__)
        return prompt, "prompt_translation_failed"
