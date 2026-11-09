"""scripts/ops/gpu_plan.py: card detection, per-video resource budget, balanced card assignment, NVENC session cap, the plan."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ops import gpu_plan as gp

ROOT = Path(__file__).resolve().parents[1]
SMI_2xA40 = ("0, GPU-aaaa1111-0000-0000-0000-000000000000, NVIDIA A40, 46068, 3, 570.86.15\n"
             "1, GPU-bbbb2222-0000-0000-0000-000000000000, NVIDIA A40, 46068, 3, 570.86.15\n")


def gpus(spec, broken=()):
    g = gp._fake_gpus(spec)
    for i in broken:
        g[i].cuda_ok, g[i].error = False, "CUDA probe failed (test)"
    return g


# ── detection ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_parse_nvidia_smi_two_cards():
    g = gp.parse_smi(SMI_2xA40 + "garbage line\n")
    assert [(x.index, x.name, x.vram_total_mb, x.vram_free_mb, x.driver) for x in g] == [
        (0, "NVIDIA A40", 46068, 46065, "570.86.15"), (1, "NVIDIA A40", 46068, 46065, "570.86.15")]
    assert g[1].uuid.startswith("GPU-bbbb") and not g[0].consumer


def test_detect_uses_csv_query_and_survives_errors():
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, SMI_2xA40, "")
    assert len(gp.detect(run=run)) == 2 and "--format=csv,noheader,nounits" in seen[0]
    assert gp.detect(run=lambda *a, **k: subprocess.CompletedProcess(a, 9, "", "NVML: Driver/library version mismatch")) == []

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    assert gp.detect(run=boom) == []


def test_consumer_cards():
    assert gp.is_consumer("NVIDIA GeForce RTX 3060") and gp.is_consumer("NVIDIA TITAN RTX")
    assert not any(gp.is_consumer(n) for n in ("NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA RTX A4000", "Tesla T4"))


def test_cgroup_cpu_and_ram(tmp_path):
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    (tmp_path / "memory.max").write_text(str(2 * 2**30))  # below any runner RAM
    assert gp.cpu_threads(str(tmp_path)) <= 8.0
    assert gp.ram_mb(str(tmp_path)) == 2 * 1024
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.max").write_text("max\n")
    assert gp.cpu_threads(str(tmp_path)) >= 1 and gp.ram_mb(str(tmp_path)) > 0


# ── capacity (the measured budget) ────────────────────────────────────────────────────────────────────────────────
def test_capacity_rtx3060_box_matches_the_load_test():
    # RTX 3060 12 GB + 12 threads + 32 GB: CPU-limited at 3 (load test: 4 at once fit VRAM but each video +60 %)
    c = gp.capacity([12288 - 300], 12, 32 * 1024)
    assert c["per_gpu"] == [5] and c["by_cpu"] == 3 and c["by_ram"] == 7 and c["total"] == 3 and c["limit"] == "cpu"


def test_capacity_2xA40_and_4xL40S():
    c = gp.capacity([46065, 46065], 32, 128 * 1024)
    assert c["per_gpu"] == [8, 8] and c["total"] == 8 and c["limit"] == "cpu"
    c = gp.capacity([46065] * 4, 128, 512 * 1024)
    assert c["by_vram"] == 32 and c["total"] == 16 and c["limit"] == "max"          # auto plans at most 16 processes
    c = gp.capacity([46065] * 2, 64, 24 * 1024)
    assert c["total"] == 5 and c["limit"] == "ram"                                  # (24 - 4) GB / 4 GB


def test_capacity_never_below_one_and_no_gpu_means_vram_does_not_limit():
    assert gp.capacity([3000], 2, 4096)["total"] == 1
    assert gp.capacity([], 16, 64 * 1024)["by_vram"] == gp.AUTO_MAX


def test_budget_env_overrides_and_validation():
    b = gp.Budget.from_env({"SONYA_VRAM_PER_JOB_MB": "4000", "SONYA_CPU_THREADS_PER_JOB": "2.5", "SONYA_JOBS_PER_GPU_MAX": "3"})
    assert (b.vram_per_job_mb, b.threads_per_job, b.jobs_per_gpu_max, b.ram_per_job_mb) == (4000, 2.5, 3, 4096)
    assert gp.capacity([46065, 46065], 64, 256 * 1024, b)["per_gpu"] == [3, 3]
    with pytest.raises(ValueError):
        gp.Budget.from_env({"SONYA_RAM_PER_JOB_MB": "lots"})
    with pytest.raises(ValueError):
        gp.Budget.from_env({"SONYA_VRAM_PER_JOB_MB": "0"})


# ── assignment ────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n_gpus", [1, 2, 4])
@pytest.mark.parametrize("n_workers", [1, 2, 3, 4, 5, 8, 16])
def test_equal_cards_round_robin_no_card_idle_none_overloaded(n_gpus, n_workers):
    a = gp.assign(n_workers, [8] * n_gpus)
    counts = [a.count(g) for g in range(n_gpus)]
    assert max(counts) - min(counts) <= 1                                           # nobody carries 2 more than another
    assert min(counts) >= 1 or n_workers < n_gpus                                    # no idle card when there are enough workers
    assert a[:n_gpus] == list(range(min(n_gpus, n_workers)))                         # first round: one per card


def test_unequal_cards_get_workers_by_capacity():
    a = gp.assign(6, [8, 4])                     # a 48 GB and a 24 GB card
    assert (a.count(0), a.count(1)) == (4, 2)
    assert gp.assign(3, []) == [] and gp.assign(2, [0, 0]) == [0, 1]


# ── NVENC session cap ─────────────────────────────────────────────────────────────────────────────────────────────
def test_nvenc_cap_geforce_by_driver_datacenter_none_env_wins():
    assert gp.nvenc_session_limit(gpus("2xA40:46068")) == 0
    assert gp.nvenc_session_limit(gpus("1xNVIDIA GeForce RTX 3060:12288@570"), {}) == 8
    assert gp.nvenc_session_limit(gpus("1xNVIDIA GeForce RTX 3060:12288@535"), {}) == 5
    assert gp.nvenc_session_limit(gpus("2xNVIDIA GeForce RTX 3090:24576@525"), {}) == 3      # per SYSTEM, not per card
    assert gp.nvenc_session_limit(gpus("1xNVIDIA GeForce RTX 3060:12288@570"), {"SONYA_NVENC_MAX_SESSIONS": "0"}) == 0
    assert gp.nvenc_session_limit(gpus("2xA40:46068"), {"SONYA_NVENC_MAX_SESSIONS": "12"}) == 12


# ── the plan with fake 1 / 2 / 4 GPUs ─────────────────────────────────────────────────────────────────────────────
def plan(spec, conc, broken=(), threads=64, ram_gb=256, **env):
    return gp.build_plan(gpus(spec, broken), {"SONYA_WORKER_CONCURRENCY": str(conc), **env}, threads=threads, ram=ram_gb * 1024)


def test_one_gpu_nothing_pinned_like_before():
    p = plan("1xNVIDIA GeForce RTX 3060:12288@570", 2, threads=12, ram_gb=32)
    assert p["concurrency"] == 2 and p["source"] == "env" and p["warnings"] == []
    assert all("CUDA_VISIBLE_DEVICES" not in w["env"] for w in p["workers"])
    assert [w["env"]["SONYA_NVENC_MAX_SESSIONS"] for w in p["workers"]] == ["8", "8"]
    assert p["per_gpu_workers"] == {0: 2}


@pytest.mark.parametrize("n_gpus,conc", [(2, 4), (2, 5), (4, 8), (4, 4), (4, 16)])
def test_multi_gpu_pins_every_worker_to_one_card_evenly(n_gpus, conc):
    p = plan(f"{n_gpus}xA40:46068", conc)
    devs = [w["env"]["CUDA_VISIBLE_DEVICES"] for w in p["workers"]]
    assert len(devs) == conc and set(devs) == {g["uuid"] for g in p["gpus"]}
    counts = list(p["per_gpu_workers"].values())
    assert max(counts) - min(counts) <= 1 and sum(counts) == conc
    assert all(w["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for w in p["workers"])
    assert all("SONYA_NVENC_MAX_SESSIONS" not in w["env"] for w in p["workers"])     # A40: no cap


def test_auto_sizes_from_hardware_and_respects_the_dispatcher_cap():
    p = plan("2xA40:46068", "auto", threads=32, ram_gb=128)
    assert p["concurrency"] == 8 and p["source"] == "auto" and p["per_gpu_workers"] == {0: 4, 1: 4}
    p = plan("2xA40:46068", "auto", threads=32, ram_gb=128, SONYA_WORKER_CONCURRENCY_MAX="6")
    assert p["concurrency"] == 6 and p["per_gpu_workers"] == {0: 3, 1: 3}
    with pytest.raises(ValueError):
        plan("2xA40:46068", "auto", SONYA_WORKER_CONCURRENCY_MAX="99")


def test_broken_card_left_out_others_work_and_it_is_reported():
    p = plan("2xA40:46068", 4, broken=(1,))
    assert p["usable"] == [0] and p["per_gpu_workers"] == {0: 4}
    assert {w["env"]["CUDA_VISIBLE_DEVICES"] for w in p["workers"]} == {p["gpus"][0]["uuid"]}
    assert any("NOT used" in w for w in p["warnings"]) and any("1 of 2 GPUs" in w for w in p["warnings"])
    p = plan("4xL40S:46068", "auto", broken=(0, 2), threads=64, ram_gb=256)
    assert p["usable"] == [1, 3] and p["concurrency"] == 16 and p["per_gpu_workers"] == {1: 8, 3: 8}


def test_warnings_for_overcommit_and_idle_cards():
    p = plan("1xNVIDIA GeForce RTX 3060:12288@570", 6, threads=12, ram_gb=32)
    assert p["concurrency"] == 6 and any("above what the hardware carries" in w for w in p["warnings"])
    p = plan("4xA40:46068", 2)
    assert any("2 GPU(s) will idle" in w for w in p["warnings"])


def test_bad_concurrency_refused():
    for bad in ("0", "33", "two", "-1"):
        with pytest.raises(ValueError):
            plan("2xA40:46068", bad)


def test_no_gpu_check_unpinned_workers():
    p = gp.build_plan([], {"SONYA_WORKER_CONCURRENCY": "3"}, threads=8, ram=8192, gpu_check=False)
    assert [w["env"] for w in p["workers"]] == [{}, {}, {}]


def test_worker_lines_and_unsafe_values(tmp_path):
    p = plan("1xNVIDIA GeForce RTX 3060:12288@570", 1, threads=12, ram_gb=32)
    assert gp.worker_lines(p, str(tmp_path)) == [
        f"1 SONYA_GPU_INDEX=0 SONYA_NVENC_MAX_SESSIONS=8 SONYA_NVENC_PROBED=ok SONYA_NVENC_SLOT_DIR={tmp_path}/nvenc"]
    p["workers"][0]["env"]["X"] = "a b; rm -rf /"
    with pytest.raises(ValueError):
        gp.worker_lines(p, str(tmp_path))


def test_plan_is_json_serialisable_and_describe():
    p = plan("2xA40:46068", "auto", threads=32, ram_gb=128)
    back = json.loads(json.dumps(p))
    assert back["concurrency"] == 8
    assert "8 worker(s) (auto) on 2 GPU(s): GPU0 x4, GPU1 x4" in gp.describe(p)


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cli_workers_reads_the_plan_file(tmp_path):
    p = plan("2xA40:46068", 2)
    (tmp_path / "gpu_plan.json").write_text(json.dumps(p))
    env = {"SONYA_STATE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-m", "scripts.ops.gpu_plan", "workers"], capture_output=True, text=True, cwd=ROOT, env=env)
    assert out.returncode == 0
    lines = out.stdout.split("\n")[:2]
    assert lines[0].startswith("1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-fake0000") and "GPU-fake0001" in lines[1]


def test_cli_workers_without_plan_file_and_bad_value(tmp_path):
    env = {"SONYA_STATE_DIR": str(tmp_path), "SONYA_WORKER_CONCURRENCY": "2", "PATH": "/usr/bin:/bin",
           "SONYA_CPU_THREADS": "8", "SONYA_RAM_MB": "16000"}
    out = subprocess.run([sys.executable, "-m", "scripts.ops.gpu_plan", "workers"], capture_output=True, text=True, cwd=ROOT, env=env)
    assert out.returncode == 0 and out.stdout.split() == ["1", "2"]
    out = subprocess.run([sys.executable, "-m", "scripts.ops.gpu_plan", "workers"], capture_output=True, text=True, cwd=ROOT,
                         env={**env, "SONYA_WORKER_CONCURRENCY": "auto", "SONYA_WORKER_CONCURRENCY_MAX": "x"})
    assert out.returncode == 3 and "SONYA_WORKER_CONCURRENCY_MAX" in out.stderr


def test_cli_simulate(capsys):
    assert gp.main(["simulate", "--gpus", "2xA40:46068", "--threads", "32", "--ram-gb", "128", "--broken", "1"]) == 0
    out = capsys.readouterr().out
    assert "8 worker(s) (auto) on 1 GPU(s): GPU0 x8" in out and "WARNING working on 1 of 2 GPUs" in out


def test_readiness_nvenc_result_is_handed_to_each_worker():
    g = gpus("2xA40:46068")
    g[1].nvenc_ok = False
    p = gp.build_plan(g, {"SONYA_WORKER_CONCURRENCY": "4"}, threads=64, ram=256 * 1024)
    assert [w["env"]["SONYA_NVENC_PROBED"] for w in p["workers"]] == ["ok", "fail", "ok", "fail"]


def test_explicit_32_on_two_cards_is_16_per_card():
    p = plan("2xA40:46068", 32, threads=128, ram_gb=1024)
    assert p["concurrency"] == 32 and p["per_gpu_workers"] == {0: 16, 1: 16}


# ── CPU window per worker (2x A40 2026-11-09: 16+ workers on 128 threads ran the container out of threads) ──────────
def test_cpu_windows_off_for_small_servers_one_worker_and_env_zero():
    assert gp.cpu_windows(range(16), 2) == [None, None]                    # RTX 3060 box: unchanged
    assert gp.cpu_windows(range(128), 1) == [None]
    assert gp.cpu_windows(range(128), 2) == [None, None]                   # window would be the whole server
    assert gp.cpu_windows(range(128), 24, {"SONYA_CPU_WINDOW": "0"}) == [None] * 24


def test_cpu_windows_big_server_overlap_and_cover_every_cpu():
    w = gp.cpu_windows(range(128), 24, {})
    assert all(len(x) == 16 for x in w)                                     # max(16, 2*128/24 = 11)
    assert set().union(*map(set, w)) == set(range(128))
    w8 = gp.cpu_windows(range(128), 8, {})
    assert [len(x) for x in w8] == [32] * 8 and w8[0] == list(range(32)) and w8[1] == list(range(16, 48))
    w32 = gp.cpu_windows(range(128), 32, {"SONYA_CPU_WINDOW": "20"})
    assert all(len(x) == 20 for x in w32) and w32[31] == sorted([(124 + i) % 128 for i in range(20)])   # wraps around


def test_cpu_windows_use_the_allowed_cpu_ids_not_0_to_n():
    cpus = list(range(64, 128)) + list(range(192, 256))                     # a cpuset of 128 non-contiguous ids
    w = gp.cpu_windows(cpus, 16, {})
    assert all(set(x) <= set(cpus) for x in w) and set().union(*map(set, w)) == set(cpus)


def test_cpu_list_round_trip():
    assert gp.cpu_list([0, 1, 2, 3, 8, 9, 11]) == "0-3,8-9,11"
    assert gp.parse_cpu_list("0-3,8-9,11") == [0, 1, 2, 3, 8, 9, 11]
    assert gp.cpu_list(gp.cpu_windows(range(128), 32, {})[31]) == "0-11,124-127"


def test_plan_puts_the_window_into_the_worker_env_and_worker_lines_accept_it(tmp_path):
    gpus = [gp.Gpu(i, f"GPU-{i}", "A40", 46068, cuda_ok=True, nvenc_ok=True) for i in range(2)]
    plan = gp.build_plan(gpus, {"SONYA_WORKER_CONCURRENCY": "24"}, threads=128, ram=1_000_000, cpus=range(128))
    assert plan["cpu_window"] == 16
    assert plan["workers"][0]["env"]["SONYA_CPU_AFFINITY"] == "0-15"
    lines = gp.worker_lines(plan, str(tmp_path))
    assert "SONYA_CPU_AFFINITY=0-15" in lines[0] and len(lines) == 24
    small = gp.build_plan(gpus[:1], {"SONYA_WORKER_CONCURRENCY": "2"}, threads=16, ram=32768, cpus=range(16))
    assert all("SONYA_CPU_AFFINITY" not in w["env"] for w in small["workers"]) and small["cpu_window"] == 0
