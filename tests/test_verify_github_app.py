import datetime as dt
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from tools import verify_github_app as v

REPO = "samnesvoj/sonya-video-worker-production"
NOW = dt.datetime(2026, 11, 8, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def env(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    f = tmp_path / "k.pem"
    f.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return {"GH_APP_ID": "1", "GH_APP_INSTALLATION_ID": "2", "GH_APP_PRIVATE_KEY_FILE": str(f)}


def opener_for(resp, repo_status=200, other_status=404):
    class R:
        def __init__(s, body, status=200): s.body, s.status = body, status
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def read(s): return json.dumps(s.body).encode()
    def op(req, timeout=0):
        if req.full_url.endswith("/access_tokens"):
            return R(resp)
        if REPO in req.full_url:
            if repo_status != 200:
                import urllib.error; raise urllib.error.HTTPError(req.full_url, repo_status, "x", {}, None)
            return R({}, 200)
        import urllib.error; raise urllib.error.HTTPError(req.full_url, other_status, "x", {}, None)
    return op


def good(**over):
    r = {"token": "ghs_SECRET_TOKEN_VALUE", "expires_at": "2026-11-08T12:59:00Z", "permissions": {"contents": "read", "metadata": "read"},
         "repository_selection": "selected", "repositories": [{"full_name": REPO}]}
    r.update(over)
    return r


def run(env, resp, **kw):
    return v.verify(REPO, "samnesvoj/sonya-production", opener_for(resp, **kw), env, now=NOW)


def test_a_correctly_narrow_app_passes_every_check(env):
    assert all(ok for _, ok, _ in run(env, good()))


@pytest.mark.parametrize("over,label", [
    ({"permissions": {"contents": "write", "metadata": "read"}}, "permissions"),
    ({"permissions": {"contents": "read", "pull_requests": "write"}}, "permissions"),
    ({"repository_selection": "all"}, "one repository"),
    ({"repositories": [{"full_name": REPO}, {"full_name": "samnesvoj/other"}]}, "one repository"),
    ({"expires_at": "2026-11-08T20:00:00Z"}, "lifetime"),
])
def test_too_broad_app_fails(env, over, label):
    bad = [l for l, ok, _ in run(env, good(**over)) if not ok]
    assert any(label in l for l in bad)


def test_visible_other_repo_or_unreadable_target_fails(env):
    assert not all(ok for _, ok, _ in run(env, good(), other_status=200))
    assert not all(ok for _, ok, _ in run(env, good(), repo_status=404))


def test_token_is_never_in_the_output(env, capsys):
    for label, ok, detail in run(env, good()):
        print(label, detail)
    assert "ghs_SECRET_TOKEN_VALUE" not in capsys.readouterr().out
