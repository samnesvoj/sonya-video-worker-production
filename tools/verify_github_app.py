#!/usr/bin/env python3
"""
verify_github_app.py - prove the GitHub App is as narrow as intended. Run on the machine that holds the App key (VPS / MacBook).
It mints ONE installation token and checks, WITHOUT ever printing the token and WITHOUT any write attempt:
  1. the token's permissions are exactly contents:read (+ metadata:read, which GitHub always adds)
  2. the token is limited to exactly ONE repository, the target one
  3. it expires in <= 61 minutes
  4. the target repository is readable (HTTP 200)           [uses the token for one GET]
  5. another repository of yours is NOT visible (HTTP 404)  [--other-repo, optional but recommended]
Env: GH_APP_ID, GH_APP_INSTALLATION_ID, GH_APP_PRIVATE_KEY_FILE.     Exit 0 only if everything holds.
  python tools/verify_github_app.py --repo samnesvoj/sonya-video-worker-production --other-repo samnesvoj/sonya-production
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mint_release_token import mint_full  # noqa: E402

ALLOWED_PERMS = {"contents": "read", "metadata": "read"}


def http_status(url: str, token: str, opener=urllib.request.urlopen) -> int:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "sonya-verify-app"})
    try:
        with opener(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def verify(repo: str, other_repo: str | None, opener=urllib.request.urlopen, environ=os.environ, now=None) -> list[tuple[str, bool, str]]:
    name = repo.split("/")[-1]
    resp = mint_full(name, opener, environ)
    token = resp["token"]
    perms = resp.get("permissions", {})
    repos = [r.get("full_name") for r in resp.get("repositories", [])]
    exp = dt.datetime.fromisoformat(resp["expires_at"].replace("Z", "+00:00"))
    ttl = (exp - (now or dt.datetime.now(dt.timezone.utc))).total_seconds() / 60
    out = [
        ("permissions are only contents:read (+metadata:read)", perms == ALLOWED_PERMS or perms == {"contents": "read"}, json.dumps(perms)),
        ("limited to exactly one repository (the target)", resp.get("repository_selection") == "selected" and repos == [repo], f"selection={resp.get('repository_selection')} repos={repos}"),
        ("token lifetime <= 61 min", 0 < ttl <= 61, f"{ttl:.0f} min"),
        ("target repository readable", http_status(f"https://api.github.com/repos/{repo}", token, opener) == 200, ""),
    ]
    if other_repo:
        out.append(("another repository is NOT visible (404)", http_status(f"https://api.github.com/repos/{other_repo}", token, opener) == 404, other_repo))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="samnesvoj/sonya-video-worker-production"); ap.add_argument("--other-repo")
    a = ap.parse_args()
    try:
        res = verify(a.repo, a.other_repo)
    except Exception as exc:  # never dump env or key material
        print(f"FAILED to mint/verify: {type(exc).__name__}")
        sys.exit(2)
    for label, ok, detail in res:
        print(("PASS " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    sys.exit(0 if all(ok for _, ok, _ in res) else 1)
