"""Per-instance GitHub release token (central side). Uses the GitHub App private key, which exists ONLY on this host.
A fresh read-only installation token (contents:read, ONE repository, <=1 h) is minted for every create call; nothing is cached,
stored or logged."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
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


def repo_is_public(repo: str, *, opener=urllib.request.urlopen, timeout: float = 10.0) -> bool:
    """Unauthenticated GitHub API check. A private repo answers 404 to anonymous callers -> False."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "sonya-fleet"})
    try:
        with opener(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return data.get("private") is False and data.get("visibility", "public") == "public"


def anonymous_public_provider(repo: str, *, check=repo_is_public, ttl: float = 3600.0,
                              clock=time.monotonic) -> Callable[[], str]:
    """Public worker repo: NO token goes to Vast instances (bootstrap fetches the pinned SHA anonymously).
    Returns "" while the repo is public (checked without auth, cached for `ttl`); refuses to create once it is not."""
    state = {"at": None, "public": False}

    def _get() -> str:
        now = clock()
        if state["at"] is None or now - state["at"] > ttl or not state["public"]:
            state["public"], state["at"] = bool(check(repo)), now
        if not state["public"]:
            raise RuntimeError(f"{repo} is not public: configure GITHUB_RELEASE_TOKEN_FILE (fine-grained, read-only "
                               "Contents of this one repo) or a GitHub App")
        return ""
    return _get


def release_token_provider(repo: str, environ: Mapping[str, str] = os.environ) -> Callable[[], str]:
    """`repo` = owner/name. GitHub App when GH_APP_ID is set; else GITHUB_RELEASE_TOKEN_FILE; else anonymous (public repo only).
    Never put a token with write rights here: it travels to third-party Vast hosts."""
    if environ.get("GH_APP_ID"):
        return github_app_token_provider(repo.split("/")[-1], environ)
    if environ.get("GITHUB_RELEASE_TOKEN_FILE"):
        return file_token_provider(environ["GITHUB_RELEASE_TOKEN_FILE"])
    return anonymous_public_provider(repo)
