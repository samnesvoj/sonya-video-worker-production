import json

from scripts.shared import llm_usage


def test_record_and_flush_one_job(tmp_path, monkeypatch):
    llm_usage.begin("job1", "virality", tmp_path)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "data:"}}]}]
    assert llm_usage.count_images(msgs) == 1
    llm_usage.record(provider="timeweb", model="g", prompt_version="v", input_tokens=10, output_tokens=2, images=1)
    got = {}
    key = llm_usage.flush(upload=lambda k, b: got.update({k: b}))
    assert key and key.startswith("worker-v2/llm-usage/") and "/job1-" in key
    doc = json.loads(got[key])
    assert doc["job_id"] == "job1" and doc["mode"] == "virality" and doc["calls"][0]["input_tokens"] == 10
    assert "text" not in json.dumps(doc["calls"])                         # never prompts or answers
    assert llm_usage.flush(upload=lambda k, b: None) is None              # job forgotten


def test_no_job_no_file_nothing(monkeypatch):
    monkeypatch.delenv(llm_usage.ENV_FILE, raising=False)
    llm_usage.record(provider="p", model="m", prompt_version="v", input_tokens=1, output_tokens=1)
    assert llm_usage.flush(upload=lambda k, b: 1 / 0) is None


def test_upload_failure_never_raises(tmp_path):
    llm_usage.begin("j", "m", tmp_path)
    llm_usage.record(provider="p", model="m", prompt_version="v", input_tokens=1, output_tokens=1)
    assert llm_usage.flush(upload=lambda k, b: 1 / 0) is None
