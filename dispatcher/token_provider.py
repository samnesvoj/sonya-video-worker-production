"""Per-instance GitHub release token (central side). Uses the GitHub App private key, which exists ONLY on this host.
A fresh read-only installation token (contents:read, ONE repository, <=1 h) is minted for every create call; nothing is cached,
stored or logged."""
from __future__ import annotations

import os
from typing import Callable, Mapping


def github_app_token_provider(repo_name: str, environ: Mapping[str, str] = os.environ, **kw) -> Callable[[], str]:
    def _get() -> str:
        from tools.mint_release_token import mint
        return mint(repo_name, environ=environ, **kw)
    return _get
