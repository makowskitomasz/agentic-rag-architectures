from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.config import get_embedding_config
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedding configuration is required for query decomposition RAG") from exc

try:
    from src.embedder import generate_query_embedding
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedder module is required for query decomposition RAG") from exc

try:
    from src.retriever import retrieve
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Retriever module is required for query decomposition RAG") from exc

try:
    from src.reranker import rerank_chunks
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Reranker module is required for query decomposition RAG") from exc

try:
    from src.llm_orchestrator import generate_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator module is required for query decomposition RAG") from exc

try:
    from src.agents.query_decomposition.planner import QueryPlanner
    from src.agents.query_decomposition.aggregator import AnswerAggregator
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Query decomposition agents are required") from exc

try:
    from src.models.query_decomposition_models import (
        QueryDecompositionOutput,
        QueryDecompositionPlan,
        SubQueryAnswer,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Query decomposition models are required") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("QUERY_DECOMP")


def _execute_subquery_rag(
    sub_query: str,
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    index_map: Dict[str, int],
    embedding_provider: str | None,
    embedding_model: str | None,
    provider: str,
    llm_model: str | None,
    temperature: float,
    k: int,
    threshold: float,
    use_hybrid: bool,
    lexical_weight: float,
    use_reranker: bool,
    reranker_model: str | None,
    rerank_top_k: Optional[int],
) -> SubQueryAnswer:
    embedding_config = get_embedding_config(embedding_provider)
    sub_query_embedding = generate_query_embedding(
        sub_query,
        provider=embedding_config.provider,
        model=embedding_model or embedding_config.model,
    )

    retrieved_chunks = retrieve(
        query_embedding=sub_query_embedding,
        embeddings=embeddings,
        index_map=index_map,
        chunks=chunks,
        k=k,
        threshold=threshold,
        query_text=sub_query,
        use_hybrid=use_hybrid,
        lexical_weight=lexical_weight,
    )
    reranked_chunks = retrieved_chunks
    if use_reranker and retrieved_chunks:
        target_top_k = rerank_top_k if rerank_top_k is not None else k
        reranked_chunks = rerank_chunks(
            query=sub_query,
            chunks=retrieved_chunks,
            model_name=reranker_model,
            top_k=target_top_k,
        )

    answer_text = generate_answer(
        query=sub_query,
        context_chunks=reranked_chunks,
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )

    return SubQueryAnswer(
        step_id=0,
        sub_query=sub_query,
        answer=answer_text,
        retrieved_chunks=reranked_chunks,
        metadata={"chunks_selected": len(reranked_chunks)},
    )


def query_decomposition_rag(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    provider: str = "openai",
    llm_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    k: int = 5,
    threshold: float = 0.5,
    use_hybrid: bool = True,
    lexical_weight: float = 0.5,
    use_reranker: bool = True,
    reranker_model: str | None = None,
    rerank_top_k: Optional[int] = None,
    temperature: float = 1.0,
    max_subqueries: int = 3,
    use_llm_planner: bool = False,
    planner_heuristics: Optional[List] = None,
) -> Dict[str, Any]:
    if not query.strip():
        raise ValueError("Query must be a non-empty string")
    pipeline_start = time.perf_counter()
    logger.info("QUERY_DECOMP | start | query_len=%d chunks=%d", len(query), len(chunks))

    planner = QueryPlanner(provider=provider, llm_model=llm_model, heuristics=planner_heuristics)
    plan: QueryDecompositionPlan = planner.plan(
        query=query,
        max_subqueries=max_subqueries,
        use_llm=use_llm_planner,
        temperature=temperature,
    )

    aggregator = AnswerAggregator(provider=provider, llm_model=llm_model, temperature=temperature)

    sub_answers: List[SubQueryAnswer] = []
    timings: Dict[str, float] = {"planning_ms": (time.perf_counter() - pipeline_start) * 1000}
    for step in plan.steps:
        step_start = time.perf_counter()
        answer = _execute_subquery_rag(
            sub_query=step.question,
            embeddings=embeddings,
            chunks=chunks,
            index_map=index_map,
            embedding_provider=embedding_provider or provider,
            embedding_model=embedding_model,
            provider=provider,
            llm_model=llm_model,
            temperature=temperature,
            k=k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
        )
        answer.step_id = step.step_id
        sub_answers.append(answer)
        timings[f"subquery_{step.step_id}_ms"] = (time.perf_counter() - step_start) * 1000

    aggregation_start = time.perf_counter()
    final_answer, notes = aggregator.aggregate(query, plan, sub_answers)
    timings["aggregation_ms"] = (time.perf_counter() - aggregation_start) * 1000
    timings["total_ms"] = (time.perf_counter() - pipeline_start) * 1000

    result = QueryDecompositionOutput(
        original_query=query,
        plan=plan,
        sub_answers=sub_answers,
        final_answer=final_answer,
        aggregator_notes=notes,
        timings=timings,
    )
    logger.success("QUERY_DECOMP | completed | total=%.2f ms", timings["total_ms"])
    return result.model_dump()
