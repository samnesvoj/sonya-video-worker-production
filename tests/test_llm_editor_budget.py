"""Owner limit 2026-11-12: the cutting editor costs <= 1.5 RUB per video (one call, candidates only)."""
from scripts.shared import llm_editor as le


def _cands(n, chars=2000):
    return [{"id": f"c{i}", "start": i * 60.0, "end": i * 60.0 + 50, "text": "слово " * (chars // 6)} for i in range(n)]


def test_pool_and_text_are_capped_within_budget(monkeypatch):
    for k in ("SONYA_LLM_EDITOR_BUDGET_RUB", "SONYA_LLM_RUB_PER_M_IN", "SONYA_LLM_RUB_PER_M_OUT"):
        monkeypatch.delenv(k, raising=False)
    pool, msgs, est = le.fit_budget(_cands(40), 8)
    assert est <= 1.5
    assert all(len(json_item) <= le.TEXT_CHARS for json_item in (c["text"][:le.TEXT_CHARS] for c in pool))


def test_tight_budget_shrinks_pool_but_not_below_want(monkeypatch):
    monkeypatch.setenv("SONYA_LLM_EDITOR_BUDGET_RUB", "0.0001")
    pool, _, _ = le.fit_budget(_cands(30), 8)
    assert len(pool) == 8


def test_rerank_logs_cost_from_real_tokens(monkeypatch):
    class R:
        text = '{"clips": [{"id": "c0", "score": 80, "self_contained": true, "qa": true, "hook": "x", "keywords": [], "reason": "r"}]}'
        input_tokens, output_tokens, model = 8000, 1500, "m"

        def telemetry(self):
            return {"input_tokens": 8000, "output_tokens": 1500}

    seen = {}

    def chat(msgs, **kw):
        seen["n"] = len(msgs)
        return R()

    verdicts, meta = le.rerank(_cands(40, 300), 8, chat)
    assert meta["pool"] <= le.DEFAULT_POOL and verdicts
    assert meta["cost_rub_est"] == round(8000 * 40 / 1e6 + 1500 * 240 / 1e6, 3)
