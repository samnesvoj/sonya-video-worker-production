#!/usr/bin/env python3
"""
sonya_bootstrap.py - fetch and verify ONE pinned release of the SONYA video worker.

Runs inside the code-free base image BEFORE any SONYA application code exists on the
instance. Standard library + git only. This file is baked into the image by CI and is
never loaded from the release it verifies.

Env (validated, values never printed):
  SONYA_RELEASE_SHA      required, full 40-hex commit SHA. Branches/HEAD/short SHAs are rejected.
  SONYA_RELEASE_REPO     owner/name of the repo (default below)
  GITHUB_RELEASE_TOKEN   optional. Public repo: absent -> anonymous fetch (no credential on the instance).
                         Private repo: short-lived (<=1h) read-only token for that one repo.
                         Reaches git only through the child's environment (not argv, URL, file, git config).
                         Optional when the release is already verified on disk.
  SONYA_RELEASE_TAG      optional; the tag must resolve to the SHA
  SONYA_RELEASES_DIR     default /opt/sonya/releases
  SONYA_CURRENT_LINK     default /opt/sonya/current
  SONYA_BASE_LOCK_SHA256 sha256 of the dependency lock baked into this image; release must match
Stdout: verified release dir (last line). Logs go to stderr.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_REPO = "samnesvoj/sonya-video-worker-production"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MANIFEST_REL = "release/manifest.json"
VERIFIED_MARK = ".sonya-verified.json"
REQUIRED_DISABLED_MODES = {"trailer_film_breaker", "trailer_cinematic"}


class BootstrapError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", file=sys.stderr, flush=True)


class Redactor:
    """Strips the token (raw and base64 forms) from any text we emit."""

    def __init__(self, secret: Optional[str]) -> None:
        self._needles: List[str] = []
        if secret:
            self._needles = [
                secret,
                base64.b64encode(f"x-access-token:{secret}".encode()).decode(),
                base64.b64encode(secret.encode()).decode(),
            ]

    def __call__(self, text: str) -> str:
        for n in self._needles:
            text = text.replace(n, "***")
        return text


def validate_inputs(env: Dict[str, str]) -> Dict[str, str]:
    sha = (env.get("SONYA_RELEASE_SHA") or "").strip().lower()
    if not SHA_RE.match(sha):
        raise BootstrapError("SONYA_RELEASE_SHA must be a full 40-hex commit SHA (branches/HEAD are not accepted)")
    repo = (env.get("SONYA_RELEASE_REPO") or DEFAULT_REPO).strip()
    if not REPO_RE.match(repo) or ".." in repo:
        raise BootstrapError("SONYA_RELEASE_REPO must look like owner/name")
    tag = (env.get("SONYA_RELEASE_TAG") or "").strip()
    if tag and not TAG_RE.match(tag):
        raise BootstrapError("SONYA_RELEASE_TAG has unsafe characters")
    return {"sha": sha, "repo": repo, "tag": tag}


def _git_env(token: Optional[str], home: Path) -> Dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LC_ALL": "C",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "transfer.fsckObjects",
        "GIT_CONFIG_VALUE_0": "true",
    }
    if token:  # header is scoped to https://github.com/ only
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            GIT_CONFIG_COUNT="2",
            GIT_CONFIG_KEY_1="http.https://github.com/.extraheader",
            GIT_CONFIG_VALUE_1=f"AUTHORIZATION: basic {basic}",
        )
    return env


def _run_git(args: List[str], cwd: Path, env: Dict[str, str], redact: Redactor, timeout: int = 300) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(f"git {args[0]} timed out") from exc
    except FileNotFoundError as exc:
        raise BootstrapError("git is not installed in the image") from exc
    if p.returncode != 0:
        detail = redact((p.stderr or p.stdout or "").strip())[-400:]
        raise BootstrapError(f"git {args[0]} failed (exit {p.returncode}): {detail}")
    return p.stdout.strip()


def fetch_release(sha: str, repo: str, tag: str, token: Optional[str], dest: Path) -> None:
    """Fetch exactly `sha` (depth 1) and place a .git-free checkout at `dest`."""
    redact = Redactor(token)
    work = Path(tempfile.mkdtemp(prefix="fetch-", dir=dest.parent))
    try:
        env = _git_env(token, work)
        rd = work / "repo"
        _run_git(["init", "-q", str(rd)], work, env, redact)
        remote = f"https://github.com/{repo}.git"
        if os.environ.get("SONYA_ENV_TEST") == "1" and os.environ.get("SONYA_TEST_REMOTE_URL"):
            remote = os.environ["SONYA_TEST_REMOTE_URL"]  # unit tests only
        _run_git(["remote", "add", "origin", remote], rd, env, redact)
        if tag:
            out = _run_git(["ls-remote", "origin", f"refs/tags/{tag}^{{}}", f"refs/tags/{tag}"], rd, env, redact)
            if sha not in {ln.split()[0] for ln in out.splitlines() if ln.strip()}:
                raise BootstrapError("SONYA_RELEASE_TAG does not resolve to SONYA_RELEASE_SHA")
        _run_git(["fetch", "-q", "--depth", "1", "--no-tags", "origin", sha], rd, env, redact)
        if _run_git(["rev-parse", "--verify", "FETCH_HEAD^{commit}"], rd, env, redact).lower() != sha:
            raise BootstrapError("fetched commit does not match SONYA_RELEASE_SHA")
        _run_git(["-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha], rd, env, redact)
        if _run_git(["rev-parse", "HEAD"], rd, env, redact).lower() != sha:
            raise BootstrapError("checked-out HEAD does not match SONYA_RELEASE_SHA")
        shutil.rmtree(rd / ".git")  # no git metadata, no credentials on the instance
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(rd, dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def verify_release(release_dir: Path, sha: str, env: Dict[str, str]) -> dict:
    mp = release_dir / MANIFEST_REL
    if not mp.is_file():
        raise BootstrapError(f"{MANIFEST_REL} missing in release")
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BootstrapError(f"{MANIFEST_REL} is not valid JSON") from exc
    if manifest.get("schema_version") != 1:
        raise BootstrapError("unsupported manifest schema_version")
    if not REQUIRED_DISABLED_MODES <= set(manifest.get("disabled_modes") or []):
        raise BootstrapError("release does not declare Trailer modes as disabled - refusing to start")
    enabled = set(manifest.get("enabled_modes") or [])
    if not enabled or any(m.startswith("trailer") for m in enabled):
        raise BootstrapError("release enables a Trailer mode or no mode at all - refusing to start")
    expected = (env.get("SONYA_BASE_LOCK_SHA256") or "").strip().lower()
    if expected:
        lock = release_dir / "requirements" / "worker.lock.txt"
        if not lock.is_file():
            raise BootstrapError("requirements/worker.lock.txt missing in release")
        actual = hashlib.sha256(lock.read_bytes()).hexdigest()
        if actual != str(manifest.get("requirements_lock_sha256") or "").lower() or actual != expected:
            raise BootstrapError("release dependency lock differs from the lock baked into this image; use a matching base image")
    for rel in ("scripts/gpu_worker.py", "scripts/mode_policy.py", "deploy/vast/run_worker.sh"):
        if not (release_dir / rel).is_file():
            raise BootstrapError(f"required file missing in release: {rel}")
    root = release_dir.resolve()
    for p in release_dir.rglob("*"):
        if p.is_symlink():
            t = Path(os.path.realpath(p))
            if root not in t.parents and t != root:
                raise BootstrapError(f"symlink escapes release dir: {p.relative_to(release_dir)}")
    return manifest


def _mark(release_dir: Path, sha: str, manifest: dict) -> None:
    (release_dir / VERIFIED_MARK).write_text(
        json.dumps({"sha": sha, "version": manifest.get("version"), "verified_at": int(time.time())}), encoding="utf-8")


def already_verified(release_dir: Path, sha: str) -> bool:
    m = release_dir / VERIFIED_MARK
    try:
        return m.is_file() and json.loads(m.read_text()).get("sha") == sha
    except ValueError:
        return False


def point_current(link: Path, target: Path) -> None:
    tmp = link.with_name(link.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link)


def bootstrap(env: Optional[Dict[str, str]] = None) -> Path:
    env = dict(os.environ if env is None else env)
    cfg = validate_inputs(env)
    sha = cfg["sha"]
    token = (env.get("GITHUB_RELEASE_TOKEN") or "").strip() or None
    releases = Path(env.get("SONYA_RELEASES_DIR", "/opt/sonya/releases"))
    link = Path(env.get("SONYA_CURRENT_LINK", "/opt/sonya/current"))
    releases.mkdir(parents=True, exist_ok=True)
    dest = releases / sha
    if dest.is_dir() and already_verified(dest, sha):
        log(f"release {sha[:12]} already on disk and verified - no network needed")
        verify_release(dest, sha, env)  # manifest/lock re-checked on every start
    else:
        if token:
            log(f"fetching release {sha[:12]} from {cfg['repo']} (read-only, pinned SHA)")
            fetch_release(sha, cfg["repo"], cfg["tag"], token, dest)
        else:
            # Public repo: no credential at all on the instance. Integrity does not depend on auth: the exact commit SHA is
            # fetched and re-checked, then the manifest and dependency lock are verified below.
            log(f"fetching release {sha[:12]} from {cfg['repo']} anonymously (public repo, pinned SHA, no token)")
            try:
                fetch_release(sha, cfg["repo"], cfg["tag"], None, dest)
            except BootstrapError as exc:
                raise BootstrapError(f"anonymous download failed ({exc}); if the repo is private, GITHUB_RELEASE_TOKEN "
                                     "(read-only, Contents of this one repo) is required") from exc
        manifest = verify_release(dest, sha, env)
        _mark(dest, sha, manifest)
        log(f"release {sha[:12]} identity and manifest verified (version={manifest.get('version')})")
    point_current(link, dest)
    return dest


def main() -> int:
    try:
        path = bootstrap()
    except BootstrapError as exc:
        log(f"FAILED: {exc}")
        return 2
    except Exception as exc:  # never dump env or tracebacks that may carry secrets
        log(f"FAILED: unexpected {type(exc).__name__}")
        return 3
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
