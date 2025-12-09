from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.agents.active_retrieval import active_retrieval
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Active retrieval agent is required for MARAG") from exc

try:
    from src.llm_orchestrator import generate_answer, generate_structured_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator is required for MARAG") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("MARAG")


class MaragPlan(BaseModel):
    roles: List[str] = Field(..., min_length=1)
    researcher_iterations: PositiveInt


class AnalystBullet(BaseModel):
    bullet_id: PositiveInt
    chunk_id: str
    summary: str


class AnalystSummary(BaseModel):
    bullets: List[AnalystBullet] = Field(default_factory=list, min_length=1)


def _planner(role_sequence: Optional[Sequence[str]], iterations: int) -> MaragPlan:
    roles = list(role_sequence) if role_sequence else ["Researcher", "Analyst", "Synthesizer"]
    if "Researcher" not in [role.title() for role in roles]:
        roles.insert(0, "Researcher")
    if "Analyst" not in [role.title() for role in roles]:
        roles.append("Analyst")
    if "Synthesizer" not in [role.title() for role in roles]:
        roles.append("Synthesizer")
    canonical = [role.title() for role in roles]
    logger.info("MARAG | planner | roles=%s iterations=%d", canonical, iterations)
    return MaragPlan(roles=canonical, researcher_iterations=max(1, iterations))


def _default_researcher(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    provider: str,
    llm_model: str | None,
    embedding_provider: str,
    embedding_model: str,
    k: int,
    threshold: float,
    use_hybrid: bool,
    lexical_weight: float,
    use_reranker: bool,
    reranker_model: str | None,
    rerank_top_k: Optional[int],
    temperature: float,
    iterations: int,
    sufficiency_threshold: float,
) -> Dict[str, Any]:
    result = active_retrieval(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        index_map=index_map,
        provider=provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        k=k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        lexical_weight=lexical_weight,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        rerank_top_k=rerank_top_k,
        temperature=temperature,
        max_iterations=iterations,
        sufficiency_threshold=sufficiency_threshold,
    )
    logger.info("MARAG | researcher | iterations=%d unique_chunks=%d", len(result["logs"]), len(result["chunks"]))
    return result


def _default_analyst(
    query: str,
    chunks: List[Dict[str, Any]],
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], AnalystSummary]:
    dedup: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        dedup[chunk_id] = chunk
    dedup_chunks = list(dedup.values())

    if not dedup_chunks:
        raise RuntimeError("MARAG analyst received empty context.")

    context_text = "\n\n".join(
        f"[{idx + 1}] chunk_id={chunk.get('chunk_id')} :: {chunk.get('text', '')}"
        for idx, chunk in enumerate(dedup_chunks[:10])
    )
    prompt = (
        "You are an analyst agent. Review the provided context snippets and produce bullet summaries.\n"
        "Each bullet must reference the chunk_id it came from.\n"
        "Return JSON with an array 'bullets', where each item has 'bullet_id', 'chunk_id', and 'summary'.\n\n"
        "Original query:\n{query}\n\n"
        "Context snippets:\n{context}\n"
    ).format(query=query, context=context_text)

    summary = generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=AnalystSummary,
    )
    logger.info("MARAG | analyst | bullets=%d", len(summary.bullets))
    return dedup_chunks, summary


def _default_synthesizer(
    query: str,
    bullets: AnalystSummary,
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> str:
    bullet_text = "\n".join(
        f"[{bullet.bullet_id}] (chunk {bullet.chunk_id}) {bullet.summary}" for bullet in bullets.bullets
    )
    prompt = (
        "You are the synthesizer agent. Using the analyst bullet points as verified evidence, craft the final answer.\n"
        "Reference bullet IDs when making claims (e.g., [1], [2]).\n\n"
        "Question:\n{query}\n\n"
        "Analyst bullets:\n{bullets}\n"
    ).format(query=query, bullets=bullet_text)

    answer = generate_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )
    logger.info("MARAG | synthesizer | answer_length=%d", len(answer))
    return answer.strip()


def run_marag(
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
    researcher_iterations: int = 2,
    sufficiency_threshold: float = 0.8,
    role_sequence: Optional[Sequence[str]] = None,
    researcher_handler: Optional[Callable[..., Dict[str, Any]]] = None,
    analyst_handler: Optional[Callable[..., Tuple[List[Dict[str, Any]], AnalystSummary]]] = None,
    synthesizer_handler: Optional[Callable[[str, AnalystSummary], str]] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    logger.info("MARAG | start | query_len=%d", len(query))
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if embedding_provider is None or embedding_model is None:
        raise ValueError("embedding_provider and embedding_model are required for MARAG.")

    plan = _planner(role_sequence, researcher_iterations)
    logs: List[Dict[str, Any]] = [
        {"agent": "Planner", "message": f"Roles: {plan.roles}", "iterations": plan.researcher_iterations}
    ]

    researcher_result: Dict[str, Any] = {"chunks": chunks, "logs": []}
    dedup_chunks: List[Dict[str, Any]] = chunks
    analyst_summary = AnalystSummary(bullets=[AnalystBullet(bullet_id=1, chunk_id="fallback", summary="No data")])

    if "Researcher" in plan.roles:
        handler = researcher_handler or _default_researcher
        researcher_result = handler(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            k=k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
            temperature=temperature,
            iterations=plan.researcher_iterations,
            sufficiency_threshold=sufficiency_threshold,
        )
        dedup_chunks = researcher_result.get("chunks", chunks)
        logs.append(
            {
                "agent": "Researcher",
                "message": f"Iterations: {len(researcher_result.get('logs', []))}",
                "new_chunks": len(dedup_chunks),
            }
        )

    if "Analyst" in plan.roles:
        handler = analyst_handler or _default_analyst
        dedup_chunks, analyst_summary = handler(
            query=query,
            chunks=dedup_chunks,
            provider=provider,
            llm_model=llm_model,
            temperature=temperature,
        )
        logs.append({"agent": "Analyst", "message": f"Bullets generated: {len(analyst_summary.bullets)}"})

    answer: str = ""
    if "Synthesizer" in plan.roles:
        handler = synthesizer_handler or (
            lambda q, summary: _default_synthesizer(q, summary, provider, llm_model, temperature)
        )
        answer = handler(query, analyst_summary)
        logs.append({"agent": "Synthesizer", "message": f"Answer tokens≈{len(answer.split())}"})

    total_time = (time.perf_counter() - start) * 1000
    logger.success("MARAG | completed | total_time=%.2f ms", total_time)
    return {
        "query": query,
        "answer": answer,
        "plan": plan.model_dump(),
        "bullets": [bullet.model_dump() for bullet in analyst_summary.bullets],
        "chunks": dedup_chunks,
        "logs": logs,
        "time_ms": total_time,
        "provider": provider,
        "model": llm_model or "default",
    }
