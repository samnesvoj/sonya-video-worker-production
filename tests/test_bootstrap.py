"""Bootstrap: pinned-SHA fetch from a local stand-in remote + release verification."""
import base64, hashlib, json, os, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bootstrap"))
import sonya_bootstrap as sb  # noqa: E402

TOKEN = "ghs_FAKETESTTOKEN0123456789abcdefghij"
LOCK = "lock-v1\n"
LOCK_SHA = hashlib.sha256(LOCK.encode()).hexdigest()


def git(*a, cwd):
    e = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=e).stdout.strip()


def make_remote(tmp_path, over=None):
    src = tmp_path / "src"
    for d in ("release", "requirements", "scripts", "deploy/vast"):
        (src / d).mkdir(parents=True)
    (src / "requirements/worker.lock.txt").write_text(LOCK)
    m = {"schema_version": 1, "version": "t1", "enabled_modes": ["virality"],
         "disabled_modes": ["trailer_film_breaker", "trailer_cinematic"], "requirements_lock_sha256": LOCK_SHA}
    m.update(over or {})
    (src / "release/manifest.json").write_text(json.dumps(m))
    for f in ("scripts/gpu_worker.py", "scripts/mode_policy.py", "deploy/vast/run_worker.sh"):
        (src / f).write_text("#\n")
    git("init", "-q", "-b", "main", cwd=src)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=src)
    git("add", "-A", cwd=src)
    git("commit", "-qm", "r", cwd=src)
    return src, git("rev-parse", "HEAD", cwd=src)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SONYA_ENV_TEST", "1")
    return {"SONYA_ENV_TEST": "1", "SONYA_RELEASES_DIR": str(tmp_path / "rel"), "SONYA_CURRENT_LINK": str(tmp_path / "cur"),
            "GITHUB_RELEASE_TOKEN": TOKEN, "SONYA_RELEASE_REPO": "o/r", "SONYA_BASE_LOCK_SHA256": LOCK_SHA}


def run(env, src, sha, **extra):
    e = {**env, "SONYA_RELEASE_SHA": sha, **extra}
    os.environ["SONYA_TEST_REMOTE_URL"] = f"file://{src}"
    try:
        return sb.bootstrap(e)
    finally:
        os.environ.pop("SONYA_TEST_REMOTE_URL", None)


def test_fetches_exact_sha_no_git_no_token(tmp_path, env):
    src, sha = make_remote(tmp_path)
    out = run(env, src, sha)
    assert out.name == sha and not (out / ".git").exists()
    assert os.path.realpath(env["SONYA_CURRENT_LINK"]) == str(out.resolve())
    assert TOKEN not in "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())


@pytest.mark.parametrize("bad", ["main", "HEAD", "abc1234", "", "../../etc", "Z" * 40])
def test_rejects_non_sha_pins(tmp_path, env, bad):
    src, _ = make_remote(tmp_path)
    with pytest.raises(sb.BootstrapError):
        run(env, src, bad)


def test_unknown_sha_fails(tmp_path, env):
    src, _ = make_remote(tmp_path)
    with pytest.raises(sb.BootstrapError):
        run(env, src, "0" * 40)


def test_refuses_release_not_disabling_trailer(tmp_path, env):
    src, sha = make_remote(tmp_path, {"disabled_modes": []})
    with pytest.raises(sb.BootstrapError, match="Trailer"):
        run(env, src, sha)
    assert not (Path(env["SONYA_RELEASES_DIR"]) / sha / sb.VERIFIED_MARK).exists()


def test_refuses_release_enabling_trailer(tmp_path, env):
    src, sha = make_remote(tmp_path, {"enabled_modes": ["virality", "trailer_cinematic"]})
    with pytest.raises(sb.BootstrapError):
        run(env, src, sha)


def test_refuses_lock_mismatch_with_image(tmp_path, env):
    src, sha = make_remote(tmp_path)
    with pytest.raises(sb.BootstrapError, match="lock"):
        run(env, src, sha, SONYA_BASE_LOCK_SHA256="f" * 64)


def test_cached_release_needs_no_token(tmp_path, env):
    src, sha = make_remote(tmp_path)
    run(env, src, sha)
    e2 = {k: v for k, v in env.items() if k != "GITHUB_RELEASE_TOKEN"}
    assert sb.bootstrap({**e2, "SONYA_RELEASE_SHA": sha}).name == sha


def test_public_repo_downloads_anonymously_without_token(tmp_path, env, capsys):
    src, sha = make_remote(tmp_path)
    env.pop("GITHUB_RELEASE_TOKEN")
    out = run(env, src, sha)
    assert out.name == sha and not (out / ".git").exists()
    assert "anonymously" in capsys.readouterr().err


def test_anonymous_failure_explains_token_needed(tmp_path, env):
    src, sha = make_remote(tmp_path)
    env.pop("GITHUB_RELEASE_TOKEN")
    with pytest.raises(sb.BootstrapError, match="GITHUB_RELEASE_TOKEN"):
        run(env, tmp_path / "no-such-remote", sha)


def test_anonymous_still_refuses_wrong_sha(tmp_path, env):
    src, _ = make_remote(tmp_path)
    env.pop("GITHUB_RELEASE_TOKEN")
    with pytest.raises(sb.BootstrapError):
        run(env, src, "0" * 40)


def test_git_env_without_token_has_no_auth_header(tmp_path):
    e = sb._git_env(None, tmp_path)
    assert not any("extraheader" in str(v) for v in e.values()) and "AUTHORIZATION" not in json.dumps(e)


def test_tag_must_resolve_to_sha(tmp_path, env):
    src, sha = make_remote(tmp_path)
    git("tag", "v1", cwd=src)
    run(env, src, sha, SONYA_RELEASE_TAG="v1")
    (src / "x").write_text("2")
    git("add", "-A", cwd=src)
    git("commit", "-qm", "later", cwd=src)
    git("tag", "v2", cwd=src)
    with pytest.raises(sb.BootstrapError, match="TAG"):
        run({**env, "SONYA_RELEASES_DIR": str(tmp_path / "rel2")}, src, sha, SONYA_RELEASE_TAG="v2")


def test_symlink_escape_refused(tmp_path, env):
    src, _ = make_remote(tmp_path)
    os.symlink("/etc/passwd", src / "scripts/evil")
    git("add", "-A", cwd=src)
    git("commit", "-qm", "evil", cwd=src)
    with pytest.raises(sb.BootstrapError, match="symlink"):
        run(env, src, git("rev-parse", "HEAD", cwd=src))


def test_redactor_and_git_env_hide_token():
    b64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    out = sb.Redactor(TOKEN)(f"fatal {TOKEN} AUTHORIZATION: basic {b64}")
    assert TOKEN not in out and b64 not in out
    e = sb._git_env(TOKEN, Path("/tmp"))
    assert TOKEN not in json.dumps(e) and "x-access-token" not in e["GIT_CONFIG_VALUE_1"]


def test_main_never_prints_token(tmp_path, env, capsys, monkeypatch):
    src, sha = make_remote(tmp_path)
    for k, v in {**env, "SONYA_RELEASE_SHA": sha, "SONYA_TEST_REMOTE_URL": f"file://{src}", "SONYA_BASE_LOCK_SHA256": "bad"}.items():
        monkeypatch.setenv(k, v)
    assert sb.main() != 0
    cap = capsys.readouterr()
    assert TOKEN not in cap.out + cap.err
