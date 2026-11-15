"""Load test 2026-11-16: clips of one job in parallel by the CPU of this server and the jobs on it (was always 2)."""
import os

from scripts.shared import clip_parallel as cp


def test_auto_scales_with_free_cpu(monkeypatch):
    monkeypatch.delenv("SONYA_CLIP_WORKERS", raising=False)
    assert cp.clip_workers(10, threads=8, jobs=2) == 2          # RTX 3060 server, 2 jobs: the old 2
    assert cp.clip_workers(10, threads=16, jobs=1) == 4         # one job on 16 threads
    assert cp.clip_workers(10, threads=32, jobs=1) == 6         # capped
    assert cp.clip_workers(3, threads=32, jobs=1) == 3          # never more than the clips
    assert cp.clip_workers(10, threads=30.7, jobs=7) == 2       # busy power server: the floor
    monkeypatch.setenv("SONYA_CLIP_WORKERS", "3")
    assert cp.clip_workers(10, threads=64, jobs=1) == 3         # fixed by ops


def test_active_jobs_counts_live_worker_folders(tmp_path, monkeypatch):
    me = tmp_path / f"sonya_job1_p{os.getpid()}_ab"
    (me / "t").mkdir(parents=True)
    (tmp_path / "sonya_job2_p999999_cd").mkdir()                 # dead pid: not counted
    (tmp_path / "other").mkdir()
    monkeypatch.setenv("TMPDIR", str(me / "t"))
    assert cp.jobs_root() == tmp_path and cp.active_jobs() == 1
