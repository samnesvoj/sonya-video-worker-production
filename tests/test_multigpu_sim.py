"""tools/multigpu_sim.py: dry-run of a rush on the real FleetManager + the real worker plan (virtual time, no network)."""
from tools.multigpu_sim import main, simulate


def test_default_scheme_unchanged_three_rtx3060_class_servers():
    r = simulate(12, power=False)
    assert all(s["class"] == "standard" and s["slots_counted"] == 2 and s["workers"] == 2 for s in r["servers"])
    assert [s["role"] for s in r["servers"]].count("warm") == 1 and len(r["servers"]) == 3
    assert {s["state"] for s in r["servers"] if s["role"] == "burst"} == {"destroyed"}


def test_power_rush_uses_both_cards_evenly_and_is_faster():
    base, pw = simulate(12, power=False), simulate(12, power=True)
    (big,) = [s for s in pw["servers"] if s["class"] == "power"]
    assert big["gpu"] == "2x A40" and big["slots_counted"] == big["workers"] == 8
    assert big["workers_per_gpu"] == {0: 4, 1: 4} and set(big["videos_per_gpu"]) == {"0", "1"}
    assert big["state"] == "destroyed" and big["role"] == "burst"
    assert pw["all_done_s"] < base["all_done_s"]
    assert [s["role"] for s in pw["servers"]].count("warm") == 1                     # the warm is still the RTX 3060
    assert next(s for s in pw["servers"] if s["role"] == "warm")["gpu"] == "1x RTX 3060"


def test_broken_card_power_server_works_on_the_other_one():
    r = simulate(12, power=True, broken_card=True)
    (big,) = [s for s in r["servers"] if s["class"] == "power"]
    assert big["workers_per_gpu"] == {0: 8} and any("1 of 2 GPUs" in w for w in big["warnings"])
    assert r["all_done_s"] > 0


def test_cli(capsys):
    assert main(["--jobs", "6"]) == 0
    out = capsys.readouterr().out
    assert "default scheme" in out and "POWER class ON" in out
