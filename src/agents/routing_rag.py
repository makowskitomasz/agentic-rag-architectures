from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.llm_orchestrator import generate_structured_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator module is required for routing agent") from exc

try:
    from src.config import RoutingProfile, get_routing_profiles
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Routing profiles are required for routing agent") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("ROUTING_RAG")


class RoutingDecision(BaseModel):
    embedding_profile: str = Field(..., description="Key identifying routing profile")
    pipeline: str = Field(
        ...,
        pattern="^(vanilla|self_reflective|query_decomposition|chain_verification|active_retrieval|marag|madam_rag)$",
    )
    iterations: int = Field(1, ge=1, le=5)
    followup_rounds: int = Field(0, ge=0, le=2)
    reasoning: str = Field(default="", description="Short explanation of the decision")


ROUTING_PROMPT = (
    "You are a routing planner for a Retrieval-Augmented Generation system. "
    "Choose the best embedding profile and pipeline agent to answer the question. "
    "Pipelines: vanilla (standard RAG), self_reflective, query_decomposition, chain_verification, "
    "active_retrieval, marag, madam_rag. "
    "Return JSON with keys: embedding_profile (string), pipeline (string), iterations (1-5), "
    "followup_rounds (0-2, for madam_rag), reasoning (string). "
    "Selection guidance: "
    "Use vanilla for simple, direct, single-fact questions that likely have a clear answer in one chunk. "
    "Use self_reflective when the question is straightforward but benefits from critique/refinement "
    "(ambiguity, need to polish, risk of minor errors). "
    "Use query_decomposition for multi-hop or multi-part questions that require combining evidence "
    "from multiple sources or comparing items. "
    "Use chain_verification when correctness and factual validation are critical (high-stakes, "
    "claims to verify, contradictory evidence). "
    "Use active_retrieval when the topic is sparse/long-tail, the initial retrieval may be insufficient, "
    "or query rewriting could surface better context. "
    "Use marag for complex, research-style synthesis that benefits from a multi-role pipeline "
    "(collect evidence, analyze, then synthesize). "
    "Use madam_rag for debates, conflicting sources, or when you expect two competing interpretations "
    "and want a moderated resolution. "
    "If the question is simple, prefer vanilla; if multi-hop, prefer query_decomposition; "
    "if validation is essential, prefer chain_verification."
)


Executor = Callable[[RoutingProfile, RoutingDecision], Dict[str, object]]


def _plan_route(
    query: str,
    provider: str,
    model: str,
    temperature: float,
) -> RoutingDecision:
    prompt = f"{ROUTING_PROMPT}\n\nQuestion:\n{query}"
    return generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=model,
        temperature=temperature,
        response_model=RoutingDecision,
    )


def routing_rag(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    llm_provider: str,
    llm_model: str,
    executors: Dict[str, Executor],
    temperature: float = 1.0,
    decision_fn: Optional[Callable[[str], RoutingDecision]] = None,
) -> Dict[str, Any]:
    if not executors:
        raise ValueError("Executors mapping cannot be empty")

    logger.info("ROUTING_RAG | start | query_len=%d options=%d", len(query), len(executors))

    profiles = get_routing_profiles()
    planner = decision_fn or (lambda q: _plan_route(q, llm_provider, llm_model, temperature))
    decision = planner(query)
    profile = profiles.get(decision.embedding_profile) or next(iter(profiles.values()))
    executor = executors.get(decision.pipeline)
    if executor is None:
        raise ValueError(f"No executor configured for pipeline '{decision.pipeline}'")

    logger.info(
        "ROUTING_RAG | decision | pipeline=%s profile=%s iterations=%d",
        decision.pipeline,
        profile.name,
        decision.iterations,
    )
    result = executor(profile, decision)
    logger.success("ROUTING_RAG | completed | pipeline=%s", decision.pipeline)
    return {
        "query": query,
        "answer": result.get("answer", ""),
        "plan": decision.model_dump(),
        "profile": {
            "name": profile.name,
            "embedding_provider": profile.embedding_provider,
            "embedding_model": profile.embedding_model,
            "description": profile.description,
        },
        "pipeline": decision.pipeline,
        "result": result,
    }
