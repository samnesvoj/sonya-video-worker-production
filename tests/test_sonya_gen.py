import pytest
from modes.sonya_gen import runner, workflow


def test_dims_multiples_of_16():
    for a in workflow.ASPECTS:
        for s in ("draft", "standard"):
            w, h = workflow.dims(a, s)
            assert w % 16 == 0 and h % 16 == 0


@pytest.mark.parametrize("params", [{}, {"prompt": "  "}, {"prompt": "x" * 1001}, {"prompt": "a", "aspect": "7:3"}, {"prompt": "a", "count": "x"}])
def test_bad_params_fail_before_any_install(params, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ensure_comfy", lambda: pytest.fail("must not install"))
    with pytest.raises((runner.GenError, ValueError)):
        runner.run("", str(tmp_path), params)


def test_pins_filled():
    assert len(runner.COMFY_SHA256) == 64 and len(runner.GGUF_SHA256) == 64


def test_policy_and_registry():
    from scripts import mode_policy
    assert "sonya_gen" in mode_policy.OPT_IN_MODES
    assert not mode_policy.is_hard_disabled("sonya_gen")


def test_workflow_files_match_manifest():
    import json, pathlib
    lock = json.loads((pathlib.Path(__file__).parents[1] / "release/models.lock.json").read_text())["models"]
    shas = {v["sha256"] for v in lock.values()}
    for _, sha in workflow.MODEL["files"].values():
        assert sha in shas
