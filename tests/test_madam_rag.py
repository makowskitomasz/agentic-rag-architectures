from __future__ import annotations

import os
import sys
from typing import List

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np

from src.agents.madam_rag import DebaterRecord, DebaterResponse, ModeratorDecision, ModeratorFollowUp, run_madam_rag


def test_madam_rag_without_followups():
    chunks = [{"chunk_id": "c1", "text": "alpha"}, {"chunk_id": "c2", "text": "beta"}]
    retrieval_calls: List[str] = []

    def retrieval_fn(question: str) -> List[dict]:
        retrieval_calls.append(question)
        return chunks

    def answer_fn(question: str, context: List[dict], name: str, round_id: int) -> str:
        return f"{name}:{question} cites {[c['chunk_id'] for c in context]}"

    def followup_fn(query: str, records: List[DebaterRecord], round_id: int) -> ModeratorFollowUp:
        return ModeratorFollowUp(ask_followup=False, question="")

    def decision_fn(query: str, records: List[DebaterRecord]) -> ModeratorDecision:
        assert all(len(record.responses) == 1 for record in records)
        return ModeratorDecision(winner="debater_a", reasoning="stub", final_answer="final")

    result = run_madam_rag(
        query="Test question",
        chunks=chunks,
        embeddings=np.empty((0, 0)),
        index_map={},
        provider="openai",
        llm_model=None,
        embedding_provider="openai",
        embedding_model="text-embedding",
        use_reranker=False,
        use_hybrid=False,
        followup_rounds=2,
        retrieval_fn=retrieval_fn,
        answer_fn=answer_fn,
        followup_fn=followup_fn,
        decision_fn=decision_fn,
    )

    assert result["answer"] == "final"
    assert retrieval_calls == ["Test question", "Test question"]
    assert len(result["debaters"]) == 2
