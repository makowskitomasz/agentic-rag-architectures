from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np

from src.agents.routing_rag import RoutingDecision, routing_rag
from src.config import RoutingProfile


def test_routing_agent_calls_selected_executor(monkeypatch):
    profile = RoutingProfile(name="test", embedding_provider="openai", embedding_model="model")
    monkeypatch.setattr("src.agents.routing_rag.get_routing_profiles", lambda: {"test": profile})

    called = {}

    def executor(selected_profile, decision):
        called["profile"] = selected_profile.name
        called["pipeline"] = decision.pipeline
        return {"answer": "ok"}

    decision = RoutingDecision(embedding_profile="test", pipeline="vanilla", iterations=1, followup_rounds=0)

    result = routing_rag(
        query="What is routing?",
        chunks=[{"chunk_id": "c1", "text": "info"}],
        embeddings=np.empty((0, 0)),
        index_map={},
        llm_provider="openai",
        llm_model="gpt",
        executors={"vanilla": executor},
        decision_fn=lambda _: decision,
    )

    assert called["profile"] == "test"
    assert result["answer"] == "ok"
    assert result["plan"]["pipeline"] == "vanilla"
