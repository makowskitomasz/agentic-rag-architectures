from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np

from src.agents.marag import AnalystBullet, AnalystSummary, run_marag


def _stub_researcher(**_: Dict) -> Dict[str, List[Dict[str, str]]]:
    return {"chunks": [{"chunk_id": "c1", "text": "alpha"}, {"chunk_id": "c2", "text": "beta"}], "logs": [{"msg": "stub"}]}


def _stub_analyst(**_: Dict) -> Tuple[List[Dict[str, str]], AnalystSummary]:
    summary = AnalystSummary(
        bullets=[
            AnalystBullet(bullet_id=1, chunk_id="c1", summary="alpha point"),
            AnalystBullet(bullet_id=2, chunk_id="c2", summary="beta point"),
        ]
    )
    return [{"chunk_id": "c1", "text": "alpha"}, {"chunk_id": "c2", "text": "beta"}], summary


def _stub_synth(query: str, summary: AnalystSummary) -> str:
    return f"{query} -> {len(summary.bullets)} bullets"


def test_marag_allows_dependency_injection() -> None:
    result = run_marag(
        query="Test query",
        chunks=[{"chunk_id": "c0", "text": "seed"}],
        embeddings=np.empty((0, 0)),
        index_map={},
        provider="openai",
        llm_model=None,
        embedding_provider="openai",
        embedding_model="text-embedding",
        researcher_handler=_stub_researcher,
        analyst_handler=_stub_analyst,
        synthesizer_handler=_stub_synth,
        role_sequence=["Researcher", "Analyst", "Synthesizer"],
    )

    assert result["answer"] == "Test query -> 2 bullets"
    assert len(result["bullets"]) == 2
    assert any(log["agent"] == "Planner" for log in result["logs"])
