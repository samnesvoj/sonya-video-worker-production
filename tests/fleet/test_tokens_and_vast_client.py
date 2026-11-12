import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from dispatcher.vast_api import VastClient, VastError, VastInstance
from tools import mint_release_token as mrt


def test_github_app_token_request_is_minimal_and_key_never_leaves(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    kf = tmp_path / "app.pem"
    kf.write_bytes(pem)
    seen = {}

    class R:
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def __getattr__(s, n): raise AttributeError
        def read(s): return json.dumps({"token": "ghs_MINTED"}).encode()

    def opener(req, timeout=0):
        seen["url"], seen["body"], seen["auth"] = req.full_url, json.loads(req.data), req.headers["Authorization"]
        return R()
    env = {"GH_APP_ID": "123", "GH_APP_INSTALLATION_ID": "456", "GH_APP_PRIVATE_KEY_FILE": str(kf)}
    assert mrt.mint("sonya-video-worker-production", opener=opener, environ=env) == "ghs_MINTED"
    assert seen["url"].endswith("/app/installations/456/access_tokens")
    assert seen["body"] == {"repositories": ["sonya-video-worker-production"], "permissions": {"contents": "read"}}
    assert seen["auth"].startswith("Bearer ") and "PRIVATE KEY" not in json.dumps(seen)
    jwt = seen["auth"].split()[1]
    assert jwt.count(".") == 2


class T:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, body))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_client_retries_5xx_and_never_echoes_key_or_body():
    t = T((503, {}), (200, {"instances": [{"id": 1, "label": "sonya-pool-a", "actual_status": "running", "intended_status": "running", "dph_total": "0.2"}]}))
    c = VastClient("VASTKEY-SECRET", dry_run=False, transport=t, sleep=lambda s: None)
    (i,) = c.list_instances()
    assert isinstance(i, VastInstance) and i.id == 1 and len(t.calls) == 2
    assert "VASTKEY-SECRET" not in repr(c)
    t2 = T((401, {"detail": "bad key VASTKEY-SECRET"}))
    with pytest.raises(VastError) as ei:
        VastClient("VASTKEY-SECRET", dry_run=False, transport=t2, sleep=lambda s: None).stop(5)
    assert "VASTKEY-SECRET" not in str(ei.value)


def test_dry_run_blocks_every_write_but_allows_reads():
    t = T((200, {"instances": []}))
    c = VastClient("k", dry_run=True, transport=t)
    c.stop(1); c.resume(1); c.destroy(1); assert c.create(1, {}) is None
    assert t.calls == []
    c.list_instances()
    assert len(t.calls) == 1


def test_stop_resume_destroy_use_official_endpoints():
    t = T((200, {}), (200, {}), (200, {}))
    c = VastClient("k", dry_run=False, transport=t)
    c.stop(7); c.resume(7); c.destroy(7)
    assert [(m, u.split("vast.ai")[1], b) for m, u, h, b in t.calls] == [
        ("PUT", "/api/v0/instances/7/", {"state": "stopped"}), ("PUT", "/api/v0/instances/7/", {"state": "running"}), ("DELETE", "/api/v0/instances/7/", {})]


def test_create_requires_new_contract():
    t = T((200, {"success": True, "new_contract": 99}), (200, {"success": True}))
    c = VastClient("k", dry_run=False, transport=t)
    assert c.create(1, {"image": "x"}) == 99
    with pytest.raises(VastError):
        c.create(1, {})


def test_release_token_provider_file_and_app(tmp_path, monkeypatch):
    from dispatcher import token_provider as tp
    f = tmp_path / "tok"
    f.write_text("github_pat_x\n")
    f.chmod(0o600)
    assert tp.release_token_provider("r", {"GITHUB_RELEASE_TOKEN_FILE": str(f)})() == "github_pat_x"
    f.chmod(0o644)
    with pytest.raises(RuntimeError):
        tp.release_token_provider("r", {"GITHUB_RELEASE_TOKEN_FILE": str(f)})()
    seen = []
    monkeypatch.setattr(tp, "github_app_token_provider", lambda repo, env: seen.append(repo) or (lambda: "app"))
    assert tp.release_token_provider("r", {"GH_APP_ID": "1", "GITHUB_RELEASE_TOKEN_FILE": str(f)})() == "app"
    assert seen == ["r"]
