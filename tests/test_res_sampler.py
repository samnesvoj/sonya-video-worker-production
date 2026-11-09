"""scripts/ops/res_sampler.py: never raises, summary = per-metric max."""
from scripts.ops import res_sampler as r


def test_sample_without_gpu_and_summary():
    s = r.sample(None, None)
    assert {"t", "cpu_pct", "ram_used_mb", "load1"} <= set(s) and "vram_used_mb" not in s
    out = r.summary([{"vram_used_mb": 3000, "cg_mem_mb": 5}, {"vram_used_mb": 7000, "cg_mem_mb": 9}, {"cpu_pct": 50}])
    assert out == {"n": 3, "vram_used_mb_max": 7000, "cg_mem_mb_max": 9, "cpu_pct_max": 50}


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SONYA_RES_SAMPLER", raising=False)
    assert r.run() is None                     # returns at once, uploads nothing
