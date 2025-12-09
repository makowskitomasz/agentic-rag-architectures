from __future__ import annotations

import os
import sys
from typing import List

sys.path.append(os.path.join(os.getcwd(), ".."))

import numpy as np

from src.agents.active_retrieval import QueryRewrite, active_retrieval


def _fake_retrieval_factory(responses: List[List[dict]]):
    calls = {"index": 0, "queries": []}

    def _retrieval(query: str) -> List[dict]:
        calls["queries"].append(query)
        idx = min(calls["index"], len(responses) - 1)
        calls["index"] += 1
        return responses[idx]

    return _retrieval, calls


def test_active_retrieval_rewrites_when_insufficient() -> None:
    retrieval_fn, calls = _fake_retrieval_factory(
        [
            [{"chunk_id": "q1", "text": "context 1", "score": 1.0, "metadata": {}}],
            [{"chunk_id": "q2", "text": "context 2", "score": 1.0, "metadata": {}}],
        ]
    )
    eval_outcomes = iter([(False, "need more detail"), (True, "sufficient")])

    def evaluation_fn(query: str, answer: str) -> tuple[bool, str]:
        return next(eval_outcomes)

    def rewrite_fn(query: str, context: str) -> QueryRewrite:
        return QueryRewrite(revised_query=f"{query}-refined", reason="focus", is_final=False)

    result = active_retrieval(
        query="Original",
        chunks=[],
        embeddings=np.empty((0, 0)),
        index_map={},
        provider="openai",
        llm_model=None,
        use_hybrid=False,
        use_reranker=False,
        retrieval_fn=retrieval_fn,
        evaluation_fn=evaluation_fn,
        rewrite_fn=rewrite_fn,
        answer_fn=lambda q, ctx: f"answer:{q}",
        max_iterations=3,
    )

    assert len(result["logs"]) == 2
    assert calls["queries"] == ["Original", "Original-refined"]
    assert result["logs"][0]["new_chunks"] == 1
    assert result["logs"][1]["total_unique_chunks"] == 2


def test_active_retrieval_stops_when_sufficient_initially() -> None:
    retrieval_fn, calls = _fake_retrieval_factory(
        [[{"chunk_id": "c1", "text": "base", "score": 1.0, "metadata": {}}]]
    )

    def evaluation_fn(query: str, answer: str) -> tuple[bool, str]:
        return True, "complete"

    result = active_retrieval(
        query="Immediate",
        chunks=[],
        embeddings=np.empty((0, 0)),
        index_map={},
        provider="openai",
        llm_model=None,
        use_hybrid=False,
        use_reranker=False,
        retrieval_fn=retrieval_fn,
        evaluation_fn=evaluation_fn,
        answer_fn=lambda q, ctx: "answer",
        max_iterations=5,
    )

    assert len(result["logs"]) == 1
    assert calls["queries"] == ["Immediate"]
    assert result["final_query"] == "Immediate"
