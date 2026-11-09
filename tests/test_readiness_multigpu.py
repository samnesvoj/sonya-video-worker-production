"""readiness.check_gpu on fake 1 / 2 / 4 GPU servers: every card probed (CUDA + real NVENC) in its own pinned process,
a broken card is left out and reported, the plan file tells run_worker.sh which worker runs on which card."""
import json
import subprocess

import pytest

from scripts.ops import gpu_plan as gp
from scripts.ops import readiness as rd


@pytest.fixture
def server(monkeypatch, tmp_path):
    """-> configure(spec, cuda_bad=(), nvenc_bad=()) ; returns (errors, plan or None, timeline marks, log lines)."""
    monkeypatch.setenv("SONYA_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SONYA_TIMELINE_FILE", str(tmp_path / "timeline.jsonl"))
    monkeypatch.setenv("SONYA_GPU_PLAN_FILE", str(tmp_path / "gpu_plan.json"))
    monkeypatch.setenv("SONYA_MIN_VRAM_MB", "8000")
    monkeypatch.setenv("SONYA_CPU_THREADS", "64")
    monkeypatch.setenv("SONYA_RAM_MB", str(256 * 1024))
    for k in ("SONYA_SKIP_GPU_CHECK", "SONYA_REQUIRE_NVENC", "SONYA_WORKER_CONCURRENCY_MAX", "SONYA_NVENC_MAX_SESSIONS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(rd.shutil, "which", lambda name: "/usr/bin/" + name)
    lines = []
    monkeypatch.setattr(rd, "_p", lines.append)

    def configure(spec, conc="4", cuda_bad=(), nvenc_bad=(), nvenc_compiled=True):
        monkeypatch.setenv("SONYA_WORKER_CONCURRENCY", conc)
        fake = gp._fake_gpus(spec)
        for g in fake:
            g.cuda_ok = g.nvenc_ok = None
        monkeypatch.setattr(rd.gpu_plan, "detect", lambda: [gp.Gpu(**{**g.__dict__}) for g in fake])
        pinned = []

        def cuda(g, pin):
            pinned.append((g.index, pin))
            return {"ok": False, "error": "CUDA smoke test failed (RuntimeError)"} if g.index in cuda_bad else {"ok": True, "ms": 900 + g.index}
        monkeypatch.setattr(rd, "probe_cuda", cuda)
        monkeypatch.setattr(rd, "probe_nvenc", lambda g, pin: (False, "OpenEncodeSessionEx failed") if g.index in nvenc_bad else (True, ""))
        rd._STATE["nvenc_compiled"] = nvenc_compiled
        errs = rd.check_gpu()
        pf = tmp_path / "gpu_plan.json"
        marks = [json.loads(x) for x in (tmp_path / "timeline.jsonl").read_text().splitlines()] if (tmp_path / "timeline.jsonl").exists() else []
        return errs, (json.loads(pf.read_text()) if pf.exists() else None), marks, lines, pinned
    return configure


def test_one_gpu_like_before_not_pinned(server):
    errs, plan, marks, lines, pinned = server("1xNVIDIA GeForce RTX 3060:12288@570", conc="2")
    assert errs == [] and plan["concurrency"] == 2 and pinned == [(0, False)]
    assert any("nvenc: OK" in x for x in lines) and any(m["name"] == "cuda_init_done" for m in marks)
    assert not any(m["name"] == "gpu_degraded" for m in marks)


@pytest.mark.parametrize("n", [2, 4])
def test_every_card_probed_pinned_and_used(server, n):
    errs, plan, marks, lines, pinned = server(f"{n}xA40:46068", conc=str(2 * n))
    assert errs == [] and sorted(pinned) == [(i, True) for i in range(n)]
    assert plan["usable"] == list(range(n)) and set(plan["per_gpu_workers"].values()) == {2}
    assert sum(1 for x in lines if "cuda OK" in x and "nvenc: OK" in x) == n
    assert next(m for m in marks if m["name"] == "gpu_plan")["gpus"] == n


def test_broken_card_reported_and_server_still_ready(server):
    errs, plan, marks, lines, _ = server("2xA40:46068", conc="4", cuda_bad=(1,))
    assert errs == []
    assert plan["usable"] == [0] and plan["per_gpu_workers"] == {"0": 4}
    deg = next(m for m in marks if m["name"] == "gpu_degraded")
    assert deg["usable"] == 1 and deg["total"] == 2 and deg["failed"][0]["index"] == 1
    assert any("NOT USED" in x for x in lines) and any("working on 1 of 2 GPUs" in x for x in lines)


def test_all_cards_broken_fails_readiness(server):
    errs, plan, marks, *_ = server("2xA40:46068", cuda_bad=(0, 1))
    assert errs and "no usable GPU" in errs[0] and plan is None


def test_nvenc_broken_on_one_card_card_still_used_clips_on_cpu(server):
    errs, plan, _, lines, _ = server("2xA40:46068", nvenc_bad=(0,))
    assert errs == [] and plan["usable"] == [0, 1]
    assert [g["nvenc_ok"] for g in plan["gpus"]] == [False, True]
    assert any("render on the CPU" in x for x in lines)


def test_require_nvenc_drops_the_card_without_it(server, monkeypatch):
    monkeypatch.setenv("SONYA_REQUIRE_NVENC", "1")
    errs, plan, *_ = server("2xA40:46068", nvenc_bad=(1,))
    assert errs == [] and plan["usable"] == [0]
    errs, plan, *_ = server("2xA40:46068", nvenc_bad=(0, 1))
    assert errs == ["h264_nvenc does not encode"]


def test_card_below_min_vram_is_not_used(server, monkeypatch):
    monkeypatch.setenv("SONYA_MIN_VRAM_MB", "20000")
    errs, plan, *_ = server("1xA40:46068,1xNVIDIA GeForce RTX 3060:12288@570")
    assert errs == [] and plan["usable"] == [0]


def test_skip_gpu_check_writes_no_plan(server, monkeypatch, tmp_path):
    (tmp_path / "gpu_plan.json").write_text("{}")              # stale plan of a previous boot
    monkeypatch.setenv("SONYA_SKIP_GPU_CHECK", "1")
    assert rd.check_gpu() == [] and not (tmp_path / "gpu_plan.json").exists()


# ── the real probe helpers (subprocess mocked) ────────────────────────────────────────────────────────────────────
def _gpu():
    return gp.Gpu(index=1, uuid="GPU-bbbb2222-0000", name="NVIDIA A40", vram_total_mb=46068)


def test_probe_cuda_pins_the_child_and_checks_it_landed_on_that_card(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"ok": True, "ms": 700, "count": 1, "uuid": "bbbb2222-0000"}), "")
    monkeypatch.setattr(rd.subprocess, "run", run)
    assert rd.probe_cuda(_gpu(), True)["ok"] is True
    assert seen["CUDA_VISIBLE_DEVICES"] == "GPU-bbbb2222-0000" and seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.setattr(rd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, json.dumps({"ok": True, "ms": 1, "count": 1, "uuid": "aaaa1111-0000"}), ""))
    assert rd.probe_cuda(_gpu(), True) == {"ok": False, "error": "pinned process landed on another card"}
    monkeypatch.setattr(rd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, json.dumps({"ok": True, "count": 2}), ""))
    assert "did not isolate" in rd.probe_cuda(_gpu(), True)["error"]


def test_probe_cuda_crash_and_timeout(monkeypatch):
    monkeypatch.setattr(rd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, -11, "", "Segmentation fault"))
    assert rd.probe_cuda(_gpu(), True) == {"ok": False, "error": "CUDA probe crashed (rc -11)"}

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 180)
    monkeypatch.setattr(rd.subprocess, "run", hang)
    assert rd.probe_cuda(_gpu(), True) == {"ok": False, "error": "CUDA probe timed out"}


def test_probe_nvenc_runs_a_real_encode_on_the_pinned_card(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return subprocess.CompletedProcess(cmd, 1, "", "line\n[h264_nvenc] OpenEncodeSessionEx failed: incompatible client key (21)")
    monkeypatch.setattr(rd.subprocess, "run", run)
    ok, why = rd.probe_nvenc(_gpu(), True)
    assert not ok and "OpenEncodeSessionEx" in why and "h264_nvenc" in seen["cmd"] and seen["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-bbbb2222-0000"
