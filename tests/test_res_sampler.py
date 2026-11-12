"""scripts/ops/res_sampler.py: never raises, summary = per-metric max."""
import pytest

from scripts.ops import res_sampler as r


def test_sample_without_gpu_and_summary():
    pytest.importorskip("psutil")                # in the image; not in the light CI test env
    s = r.sample(None, None)
    assert {"t", "cpu_pct", "ram_used_mb", "load1"} <= set(s) and "vram_used_mb" not in s
    out = r.summary([{"vram_used_mb": 3000, "cg_mem_mb": 5}, {"vram_used_mb": 7000, "cg_mem_mb": 9}, {"cpu_pct": 50}])
    assert out == {"n": 3, "vram_used_mb_max": 7000, "cg_mem_mb_max": 9, "cpu_pct_max": 50}


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SONYA_RES_SAMPLER", raising=False)
    assert r.run() is None                     # returns at once, uploads nothing


class FakeNvml:
    """Two cards: card 0 busy (2 workers), card 1 idle."""
    from types import SimpleNamespace as NS
    USED = {0: 5000 * 2**20, 1: 700 * 2**20}

    def nvmlDeviceGetMemoryInfo(self, h):
        return self.NS(used=self.USED[h], total=46068 * 2**20)

    def nvmlDeviceGetUtilizationRates(self, h):
        return self.NS(gpu=80 if h == 0 else 0)

    def nvmlDeviceGetEncoderUtilization(self, h):
        return (35 if h == 0 else 0, 167000)

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [1, 2] if h == 0 else []


def test_per_gpu_samples_and_summary():
    gs = r.gpu_samples(FakeNvml(), [0, 1])
    assert gs == [{"i": 0, "vram_used_mb": 5000, "vram_total_mb": 46068, "gpu_util": 80, "enc_util": 35, "procs": 2},
                  {"i": 1, "vram_used_mb": 700, "vram_total_mb": 46068, "gpu_util": 0, "enc_util": 0, "procs": 0}]
    rows = [{"vram_used_mb": 5700, "gpus": gs}, {"vram_used_mb": 900, "gpus": [dict(gs[0], gpu_util=40, vram_used_mb=200), gs[1]]}]
    s = r.summary(rows)
    assert s["vram_used_mb_max"] == 5700
    assert s["per_gpu"]["0"] == {"vram_used_mb_max": 5000, "gpu_util_max": 80, "enc_util_max": 35, "procs_max": 2,
                                 "gpu_util_avg": 60.0, "vram_total_mb": 46068}
    assert s["per_gpu"]["1"]["gpu_util_max"] == 0                                # the idle card is visible


def test_one_card_keeps_the_old_fields_only():
    pytest.importorskip("psutil")
    s = r.sample(FakeNvml(), [0])
    assert s["vram_used_mb"] == 5000 and s["gpu_util"] == 80 and "gpus" not in s
    s = r.sample(FakeNvml(), [0, 1])
    assert s["vram_used_mb"] == 5700 and s["vram_total_mb"] == 2 * 46068 and len(s["gpus"]) == 2
