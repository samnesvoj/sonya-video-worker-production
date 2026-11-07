import json
import pathlib
import shutil
import subprocess
import sys

import pytest

from tools import compare_runs, mode_dev, render_vast_template

ROOT = pathlib.Path(__file__).resolve().parents[1]
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe")


@pytest.fixture(scope="session")
def CLIP(tmp_path_factory):
    """3 s synthetic 320x240 clip with audio, generated on the fly (no binary fixtures in Git)."""
    p = tmp_path_factory.mktemp("clip") / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True) if shutil.which("ffmpeg") else None
    return p


def test_mode_map_covers_every_registered_non_trailer_mode_and_tests_exist():
    from scripts import mode_policy
    assert set(mode_dev.MAP["modes"]) == set(mode_policy.DEFAULT_ENABLED_MODES) | set(mode_policy.OPT_IN_MODES)
    for name, m in mode_dev.MAP["modes"].items():
        assert m["default_enabled"] == (name in mode_policy.DEFAULT_ENABLED_MODES)
        for t in m["tests"] + mode_dev.MAP["gate"]:
            assert (ROOT / "tests" / f"{t}.py").exists(), f"{name}: missing test {t}"
        for path in m["own"]:
            assert (ROOT / path).exists()
    assert set(mode_dev.MAP["disabled_forever_here"]) == set(mode_policy.HARD_DISABLED_MODES)


def test_trailer_is_not_workable_and_branch_names_are_validated():
    for m in ("trailer_film_breaker", "trailer_cinematic"):
        with pytest.raises(SystemExit):
            mode_dev.mode_or_die(m)
    assert mode_dev.branch_name("virality", "fix-crop-jitter") == "mode/virality/fix-crop-jitter"
    for bad in ("Fix Crop", "../x", "a", "x" * 60):
        with pytest.raises(SystemExit):
            mode_dev.branch_name("virality", bad)
    with pytest.raises(SystemExit):
        mode_dev.branch_name("nope", "ok-slug")


def test_mode_test_selection_always_includes_gate_tests():
    files = mode_dev.test_files("beep")
    assert "tests/test_mode_policy.py" in files and "tests/test_beep.py" in files and "tests/test_streamer_runner_split.py" not in files


def test_json_diff():
    d = compare_runs.json_diff({"a": 1, "b": [1, 2], "c": {"x": 1}}, {"a": 2, "b": [1], "d": 0, "c": {"x": 1}})
    assert any("/a" in x for x in d) and any("length 2 -> 1" in x for x in d) and any("/d: added" in x for x in d) and any("/c" in x for x in d) is False


@needs_ffmpeg
def test_identical_outputs_show_no_regression_and_degraded_output_is_flagged(tmp_path, CLIP):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    shutil.copy(CLIP, a / "clip.mp4"); shutil.copy(CLIP, b / "clip.mp4")
    rep = compare_runs.compare(a, b)
    assert rep["verdict"].startswith("NO_VISIBLE") and rep["videos"][0]["ssim"] > 0.99
    small = tmp_path / "c"; small.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLIP), "-vf", "scale=iw/2:ih/2", "-an", str(small / "clip.mp4")], check=True)
    rep2 = compare_runs.compare(a, small)
    assert rep2["verdict"] == "REVIEW" and any("resolution" in f for f in rep2["flags"])
    assert any("audio track lost" in f for f in rep2["flags"])


@needs_ffmpeg
def test_compare_cli_runs_baseline_in_a_temporary_worktree_and_leaves_tree_clean(tmp_path, CLIP):
    out = tmp_path / "res"
    before = subprocess.run(["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True).stdout
    rc = compare_runs.main(["--mode", "virality", "--input", str(CLIP), "--baseline", "HEAD", "--out", str(out),
                            "--runner-cmd", "cp {input} {out}/clip.mp4"])
    assert rc == 0 and (out / "report.json").exists() and (out / "report.md").read_text().startswith("# Comparison")
    assert subprocess.run(["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True).stdout == before


def test_compare_refuses_trailer():
    with pytest.raises(SystemExit):
        compare_runs.main(["--mode", "trailer_film_breaker", "--input", "x.mp4"])


def test_vast_template_render_pins_everything_and_carries_no_secret():
    img = "ghcr.io/samnesvoj/sonya-video-worker-production-base@sha256:" + "a" * 64
    t = render_vast_template.render(img, "b" * 40)
    blob = json.dumps(t)
    assert t["ports"] == [] and t["env"]["ENABLED_MODES"] == "virality,beep" and "trailer" not in blob.lower()
    for forbidden in ("TOKEN", "WORKER_SECRET", "S3_", "API_KEY", "PASSWORD"):
        assert forbidden not in blob
    with pytest.raises(SystemExit):
        render_vast_template.render("ghcr.io/x/y:latest", "b" * 40)
    with pytest.raises(SystemExit):
        render_vast_template.render(img, "main")


def _hook(lines, env=None):
    import os
    return subprocess.run(["bash", str(ROOT / "tools/git-hooks/pre-push")], input=lines, text=True, capture_output=True, env={**os.environ, **(env or {})}).returncode


def test_pre_push_hook_blocks_main_and_release_tags_but_not_mode_branches():
    z = "0" * 40
    assert _hook(f"refs/heads/x {z} refs/heads/main {z}\n") == 1
    assert _hook(f"refs/heads/x {z} refs/tags/worker-v1.2.3 {z}\n") == 1
    assert _hook(f"refs/heads/x {z} refs/heads/mode/virality/fix-crop {z}\n") == 0
    assert _hook(f"refs/heads/x {z} refs/heads/main {z}\n", {"SONYA_ALLOW_PROTECTED_PUSH": "1"}) == 0


def test_link_models_symlinks_existing_files_without_copy_or_overwrite(tmp_path):
    from tools import link_models
    src = tmp_path / "old" / "deep"; src.mkdir(parents=True)
    (src / "yolo11n.pt").write_bytes(b"x" * 10)
    mdir = tmp_path / "models"
    rc = link_models.main(["--from", str(tmp_path / "old"), "--models-dir", str(mdir)])
    assert rc == 1                                                         # a REQUIRED model (streamer face_landmarker) is still missing -> reported
    link = mdir / "common" / "yolo11n.pt"
    assert link.is_symlink() and link.read_bytes() == b"x" * 10
    assert not (mdir / "trailer").exists()                                   # Trailer models are never linked
    before = link.resolve()
    link_models.main(["--from", str(tmp_path / "old"), "--models-dir", str(mdir)])   # second run: no change
    assert link.resolve() == before
