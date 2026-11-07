#!/usr/bin/env python3
"""
mint_release_token.py - short-lived, read-only GitHub App installation token for ONE repo.
RUNS ONLY ON TRUSTED SONYA INFRASTRUCTURE (VPS dispatcher / your Mac), NEVER on a Vast instance:
it needs the GitHub App private key. GitHub expires the token after 1 hour; permissions: contents:read.
Env: GH_APP_ID, GH_APP_INSTALLATION_ID, GH_APP_PRIVATE_KEY_FILE (chmod 600).  Needs: pip install cryptography
Prints the token to stdout only (put it into the create payload env of that single instance).
"""
from __future__ import annotations

import argparse, base64, json, os, sys, time, urllib.request


def _b64(b: bytes) -> bytes:
    return base64.urlsafe_b64encode(b).rstrip(b"=")


def app_jwt(app_id: str, key_pem: bytes) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    now = int(time.time())
    head = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({"iat": now - 30, "exp": now + 9 * 60, "iss": app_id}).encode())
    sig = serialization.load_pem_private_key(key_pem, password=None).sign(head + b"." + body, padding.PKCS1v15(), hashes.SHA256())
    return (head + b"." + body + b"." + _b64(sig)).decode()


def mint(repo: str, opener=urllib.request.urlopen, environ=os.environ) -> str:
    with open(environ["GH_APP_PRIVATE_KEY_FILE"], "rb") as f:
        jwt = app_jwt(environ["GH_APP_ID"], f.read())
    req = urllib.request.Request(
        f"https://api.github.com/app/installations/{environ['GH_APP_INSTALLATION_ID']}/access_tokens",
        data=json.dumps({"repositories": [repo], "permissions": {"contents": "read"}}).encode(),
        headers={"Authorization": f"Bearer {jwt}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "sonya-release-token"}, method="POST")
    with opener(req, timeout=20) as r:
        return json.load(r)["token"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="sonya-video-worker-production")
    sys.stdout.write(mint(ap.parse_args().repo))
