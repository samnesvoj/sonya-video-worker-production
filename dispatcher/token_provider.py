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


def static_token_provider(token: str) -> Callable[[], str]:
    """First Vast test only: a pre-issued read-only token (GITHUB_RELEASE_TOKEN in the operator's shell) instead of the GitHub App.
    Bootstrap uses it exactly like an installation token (git basic auth, unset after the download)."""
    if not token:
        raise ValueError("empty GITHUB_RELEASE_TOKEN")
    return lambda: token
