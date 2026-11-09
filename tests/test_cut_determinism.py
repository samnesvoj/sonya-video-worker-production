"""Cut selection is a pure function of its input (scripts/shared/cut_selection.py)."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.shared import cut_selection as cs

REPO = Path(__file__).resolve().parents[1]




def interview(seed=0, n=40):
    rng = random.Random(seed)
    out, t = [], 0.4
    for s in range(n):
        k = rng.randint(4, 14)
        for i in range(k):
            d = rng.uniform(0.15, 0.6)
            out.append({"word": f" слово{s}_{i}" + ("." if i == k - 1 else ("," if rng.random() < 0.1 else "")),
                        "start": round(t, 3), "end": round(t + d, 3)})
            t += d + rng.uniform(0.02, 0.15)
        t += rng.uniform(0.3, 1.5)
    return out


CFG = cs.CandidateConfig(min_duration=10, max_duration=40, target_duration=20)


def stable(meta):
    return {k: meta[k] for k in ("selection_method", "selection_status", "candidates", "ranking_version",
                                 "tie_break", "selected")}


def test_shuffled_input_many_times_same_everything():
    words = interview(1)
    ref_segs, ref_meta = cs.select_clips(words, 4, cfg=CFG)
    for seed in range(15):
        w = [dict(x) for x in words]
        random.Random(seed).shuffle(w)
        segs, meta = cs.select_clips(w, 4, cfg=CFG)
        assert segs == ref_segs and stable(meta) == stable(ref_meta)


def test_equivalent_timestamp_formats_give_same_ids():
    words = interview(2)
    alt = [dict(w, start=str(w["start"]), end=w["end"] + 1e-9, word="  " + w["word"].strip() + " ") for w in words]
    a = [c.id for c in cs.build_candidates(cs.normalize_words(words)[0], CFG)]
    b = [c.id for c in cs.build_candidates(cs.normalize_words(alt)[0], CFG)]
    assert a == b


SNIPPET = """
import json, sys
sys.path.insert(0, {repo!r})
from scripts.shared import cut_selection as cs
from tests.test_cut_determinism import interview, CFG, stable
segs, meta = cs.select_clips(interview(3), 4, cfg=CFG)
print(json.dumps([segs, stable(meta)], sort_keys=True))
"""


def test_independent_of_pythonhashseed():
    outs = set()
    for seed in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", SNIPPET.format(repo=str(REPO))], env=env,
                           capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1


def test_fully_equal_features_break_ties_by_start_end_id():
    # identical sentences repeated: same text length, pace, completeness -> same local score
    words, t = [], 0.0
    for s in range(6):
        for i in range(10):
            words.append({"word": f" w{i}" + ("." if i == 9 else ""), "start": round(t, 3), "end": round(t + 0.4, 3)})
            t += 0.5
        t += 1.0
    cfg = cs.CandidateConfig(min_duration=4, max_duration=6, target_duration=5, max_per_start=1)
    ranked = cs.rank_local(cs.build_candidates(cs.normalize_words(words)[0], cfg), cfg)
    scores = {s for _, s, _ in ranked}
    assert len(scores) == 1
    assert [c.start for c, _, _ in ranked] == sorted(c.start for c, _, _ in ranked)


def chat_returning(build):
    def chat(messages, *, prompt_version, opt_in):
        ids = [c["id"] for c in json.loads(messages[1]["content"].split("\n", 1)[1])]
        return SimpleNamespace(text=json.dumps({"selected": build(ids)}),
                               telemetry=lambda: {"provider": "fake", "model": "m", "prompt_version": prompt_version})
    return chat


def test_permuted_llm_response_same_ranking():
    def picks(ids):
        return [{"id": ids[1], "score": 7, "reason": "a"}, {"id": ids[5], "score": 9, "reason": "b"},
                {"id": ids[9], "score": 7, "reason": "c"}, {"id": ids[14], "score": 3, "reason": "d"}]
    words = interview(4)
    ref = None
    for seed in range(8):
        def build(ids, seed=seed):
            p = picks(ids)
            random.Random(seed).shuffle(p)
            return p
        segs, meta = cs.select_clips(words, 3, llm_opt_in=True, cfg=CFG, chat=chat_returning(build))
        cur = (segs, [(x["id"], x["rank"], x["score_source"]) for x in meta["selected"]])
        ref = ref or cur
        assert cur == ref
    assert ref[1][0][2] == "llm" and [r for _, r, _ in ref[1]] == sorted(r for _, r, _ in ref[1])


def test_duplicate_and_hallucinated_ids():
    def build(ids):
        return [{"id": ids[2], "score": 4, "reason": "low"}, {"id": "c_fake", "score": 10},
                {"id": ids[2], "score": 8, "reason": "high"}]
    segs, meta = cs.select_clips(interview(5), 2, llm_opt_in=True, cfg=CFG, chat=chat_returning(build))
    assert [s["score"] for s in segs] == [8.0] and segs[0]["selection_reason"] == "high"
    assert any(p.startswith("duplicate_id:") for p in meta["llm"]["problems"])
    assert "unknown_id:c_fake" in meta["llm"]["problems"]


def test_metadata_has_rank_source_and_tie_break():
    segs, meta = cs.select_clips(interview(6), 3, cfg=CFG)
    assert meta["ranking_version"] == cs.RANKING_VERSION and meta["tie_break"] == cs.TIE_BREAK
    ranks = [x["rank"] for x in meta["selected"]]
    assert ranks[0] == 1 and ranks == sorted(ranks)
    assert all(x["score_source"] == "local" and "local_score" in x for x in meta["selected"])
