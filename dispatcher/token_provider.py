"""Per-instance GitHub release token (central side). Uses the GitHub App private key, which exists ONLY on this host.
A fresh read-only installation token (contents:read, ONE repository, <=1 h) is minted for every create call; nothing is cached,
stored or logged."""
from __future__ import annotations

import os
from pathlib import Path
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


def file_token_provider(path: str) -> Callable[[], str]:
    """A read-only token kept in a 0600 file on the VPS (e.g. a fine-grained PAT with 'Public repositories (read-only)' and
    no permissions: the worker repo is public). Re-read on every create, so the file can be rotated without a restart."""
    def _get() -> str:
        p = Path(path)
        if p.stat().st_mode & 0o077:
            raise RuntimeError("GITHUB_RELEASE_TOKEN_FILE must not be group/world accessible (chmod 600)")
        token = p.read_text().strip()
        if not token:
            raise ValueError("empty GITHUB_RELEASE_TOKEN_FILE")
        return token
    return _get


def release_token_provider(repo_name: str, environ: Mapping[str, str] = os.environ) -> Callable[[], str]:
    """GitHub App when GH_APP_ID is set (preferred: a token per instance); otherwise GITHUB_RELEASE_TOKEN_FILE."""
    if environ.get("GH_APP_ID"):
        return github_app_token_provider(repo_name, environ)
    if environ.get("GITHUB_RELEASE_TOKEN_FILE"):
        return file_token_provider(environ["GITHUB_RELEASE_TOKEN_FILE"])
    return github_app_token_provider(repo_name, environ)
