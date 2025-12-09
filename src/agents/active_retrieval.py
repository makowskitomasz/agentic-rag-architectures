from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel, Field

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.config import get_embedding_config
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedding configuration is required for Active Retrieval") from exc

try:
    from src.embedder import generate_query_embedding
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedder module is required for Active Retrieval") from exc

try:
    from src.retriever import retrieve
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Retriever module is required for Active Retrieval") from exc

try:
    from src.reranker import rerank_chunks
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Reranker module is required for Active Retrieval") from exc

try:
    from src.llm_orchestrator import generate_answer, generate_structured_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator module is required for Active Retrieval") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("ACTIVE_RETRIEVAL")


class QueryRewrite(BaseModel):
    revised_query: str = Field(..., min_length=3)
    reason: str = ""
    is_final: bool = False


class ActiveRetrievalLog(BaseModel):
    step: int
    query: str
    chunks_retrieved: int
    new_chunks: int
    total_unique_chunks: int
    reason: str = ""


def _evaluate_answer_default(
    query: str,
    answer: str,
    provider: str,
    llm_model: str | None,
    temperature: float,
    sufficiency_threshold: float,
) -> Tuple[bool, str]:
    prompt = (
        "Evaluate whether the answer sufficiently addresses the question.\n"
        "Return JSON with keys 'is_sufficient' (true/false) and 'reason'.\n"
        "Sufficiency threshold: {threshold:.2f}.\n\n"
        "Question:\n{query}\n\n"
        "Answer:\n{answer}\n"
    ).format(query=query, answer=answer, threshold=sufficiency_threshold)

    class SufficiencyModel(BaseModel):
        is_sufficient: bool
        reason: str = ""

    result = generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=SufficiencyModel,
    )
    return result.is_sufficient, result.reason


def _rewrite_query_default(
    query: str,
    context: str,
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> QueryRewrite:
    prompt = (
        "Rewrite the question to improve retrieval. "
        "Return JSON with 'revised_query', optional 'reason', and boolean 'is_final' (true if no better rewriting is possible).\n\n"
        "Original question:\n{query}\n\n"
        "Context summary:\n{context}\n"
    ).format(query=query, context=context)

    return generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=QueryRewrite,
    )


def _default_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> str:
    return generate_answer(
        query=query,
        context_chunks=context_chunks,
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )


def active_retrieval(
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
    max_iterations: int = 3,
    sufficiency_threshold: float = 0.8,
    query_embedding: Optional[np.ndarray] = None,
    evaluation_fn: Optional[Callable[[str, str], Tuple[bool, str]]] = None,
    rewrite_fn: Optional[Callable[[str, str], QueryRewrite]] = None,
    answer_fn: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    retrieval_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    start = time.perf_counter()
    logger.info("ACTIVE_RETRIEVAL | start | query_len=%d", len(query))
    original_query = query.strip()
    retrieval_query = original_query

    embedding_provider = embedding_provider or provider
    embedding_config = None if retrieval_fn else get_embedding_config(embedding_provider)
    effective_embedding_model = embedding_model or (embedding_config.model if embedding_config else None)
    current_embedding = query_embedding

    collected_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    seen_ids: set[str] = set()
    logs: List[ActiveRetrievalLog] = []
    best_answer = ""

    eval_fn = evaluation_fn or (
        lambda q, ans: _evaluate_answer_default(q, ans, provider, llm_model, temperature, sufficiency_threshold)
    )
    rewrite_callable = rewrite_fn or (lambda q, ctx: _rewrite_query_default(q, ctx, provider, llm_model, temperature))
    answer_callable = answer_fn or (lambda q, ctx: _default_answer(q, ctx, provider, llm_model, temperature))

    def default_retrieve(query_text: str, embedding_hint: Optional[np.ndarray]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        emb = embedding_hint
        if emb is None:
            if embedding_config is None:
                raise RuntimeError("Embedding configuration unavailable for active retrieval.")
            emb = generate_query_embedding(
                query_text,
                provider=embedding_config.provider,
                model=effective_embedding_model,
            )
        retrieved = retrieve(
            query_embedding=emb,
            embeddings=embeddings,
            index_map=index_map,
            chunks=chunks,
            k=k,
            threshold=threshold,
            query_text=query_text,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
        )
        if use_reranker and retrieved:
            target_top_k = rerank_top_k if rerank_top_k is not None else k
            retrieved = rerank_chunks(query=query_text, chunks=retrieved, model_name=reranker_model, top_k=target_top_k)
        return retrieved, emb

    for iteration in range(1, max_iterations + 1):
        iteration_start = time.perf_counter()
        if retrieval_fn:
            retrieved = retrieval_fn(retrieval_query) or []
            current_embedding = None
        else:
            retrieved, current_embedding = default_retrieve(retrieval_query, current_embedding)

        new_chunks = 0
        for chunk in retrieved:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                new_chunks += 1
            collected_map[chunk_id] = chunk
        max_chunks = max(k * max_iterations, k)
        while len(collected_map) > max_chunks:
            collected_map.popitem(last=False)
        collected_chunks = list(collected_map.values())

        answer = answer_callable(original_query, collected_chunks)
        best_answer = answer

        sufficiency, reason = eval_fn(original_query, answer)
        logs.append(
            ActiveRetrievalLog(
                step=iteration,
                query=retrieval_query,
                chunks_retrieved=len(retrieved),
                new_chunks=new_chunks,
                total_unique_chunks=len(seen_ids),
                reason=reason,
            )
        )
        if sufficiency:
            break

        rewrite = rewrite_callable(original_query, " ".join(chunk.get("text", "") for chunk in collected_chunks))
        retrieval_query = rewrite.revised_query.strip()
        current_embedding = None
        if not retrieval_query:
            logger.warning("ACTIVE_RETRIEVAL | rewrite produced empty query; stopping.")
            break
        if rewrite.is_final:
            logger.info("ACTIVE_RETRIEVAL | LLM indicated final iteration.")
            break
        logger.info(
            "ACTIVE_RETRIEVAL | iteration %d -> new query: %s",
            iteration,
            retrieval_query,
        )
        logger.info(
            "ACTIVE_RETRIEVAL | iteration %d completed | chunks=%d time=%.2f ms",
            iteration,
            len(retrieved),
            (time.perf_counter() - iteration_start) * 1000,
        )

    total_time = (time.perf_counter() - start) * 1000
    logger.success("ACTIVE_RETRIEVAL | completed | total_time=%.2f ms iterations=%d", total_time, len(logs))

    return {
        "initial_query": original_query,
        "final_query": retrieval_query,
        "answer": best_answer,
        "chunks": list(collected_map.values()),
        "logs": [log.model_dump() for log in logs],
        "total_ms": total_time,
    }
