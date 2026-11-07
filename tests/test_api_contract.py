"""Worker <-> production API contract (snapshot of bd8c968, inspected read-only; the live API is NOT called)."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "contracts/worker_api_bd8c968.json").read_text())
SRC = (ROOT / "scripts/gpu_worker.py").read_text()


def _norm(p):
    return re.sub(r"\{[^}]+\}", "{}", p)


def test_every_client_endpoint_exists_on_production():
    used = {_norm(m) for m in re.findall(r'["\'](/api/worker/[^"\']*)["\']', SRC)}
    assert used and used <= {_norm(p) for p in C["endpoints"]}


def test_worker_statuses_are_accepted_by_production():
    sent = set(re.findall(r'JOB_STATUS_\w+\s*=\s*"([a-z_]+)"', SRC)) | {"analyzing", "yolo", "scripting", "tts", "subtitles", "assembling"}
    assert sent <= set(C["valid_statuses"])


def test_claim_sends_modes_field_supported_by_server():
    assert 'body["modes"] = modes' in SRC and "modes?" in " ".join(C["claim"]["body"])


def test_worker_modes_are_subset_of_server_modes_plus_beep_and_trailer_blocked():
    from scripts import mode_policy
    server = set(C["server_allowed_modes"]) | {"beep"}
    assert set(mode_policy.DEFAULT_ENABLED_MODES) | set(mode_policy.OPT_IN_MODES) <= server
    assert not (set(mode_policy.DEFAULT_ENABLED_MODES) | set(mode_policy.OPT_IN_MODES)) & mode_policy.HARD_DISABLED_MODES
