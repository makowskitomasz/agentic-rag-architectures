from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any

import numpy as np
from pydantic import BaseModel

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.config import get_embedding_config
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedding configuration is required for chain-of-verification") from exc

try:
    from src.embedder import generate_query_embedding
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedder module is required for chain-of-verification") from exc

try:
    from src.retriever import retrieve
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Retriever module is required for chain-of-verification") from exc

try:
    from src.reranker import rerank_chunks
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Reranker module is required for chain-of-verification") from exc

try:
    from src.llm_orchestrator import generate_answer, generate_structured_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator module is required for chain-of-verification") from exc

try:
    from src.models.chain_verification_models import (
        ChainVerificationOutput,
        VerificationPlan,
        VerificationStatement,
        VerificationStatus,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Chain-of-verification models are required") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("CHAIN_VERIFY")


class StatementAssessment(BaseModel):
    status: VerificationStatus
    reasoning: str


def _generate_statements(
    query: str,
    answer: str,
    provider: str,
    llm_model: str | None,
    temperature: float,
    max_statements: int,
) -> List[str]:
    prompt = (
        "Extract between 2 and {max_statements} factual statements from the given answer that require verification. "
        "Statements must be concise and cover distinct claims.\n\n"
        "Question:\n{query}\n\n"
        "Answer:\n{answer}\n"
    ).format(max_statements=max_statements, query=query, answer=answer)

    plan = generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=VerificationPlan,
    )
    return plan.statements[:max_statements]


def _retrieve_evidence(
    statement: str,
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    index_map: Dict[str, int],
    embedding_provider: str | None,
    embedding_model: str | None,
    k: int,
    threshold: float,
    use_hybrid: bool,
    lexical_weight: float,
    use_reranker: bool,
    reranker_model: str | None,
    rerank_top_k: Optional[int],
) -> List[Dict[str, Any]]:
    embedding_config = get_embedding_config(embedding_provider)
    statement_embedding = generate_query_embedding(
        statement,
        provider=embedding_config.provider,
        model=embedding_model or embedding_config.model,
    )

    retrieved_chunks = retrieve(
        query_embedding=statement_embedding,
        embeddings=embeddings,
        index_map=index_map,
        chunks=chunks,
        k=k,
        threshold=threshold,
        query_text=statement,
        use_hybrid=use_hybrid,
        lexical_weight=lexical_weight,
    )
    if use_reranker and retrieved_chunks:
        target_top_k = rerank_top_k if rerank_top_k is not None else k
        reranked_chunks = rerank_chunks(
            query=statement,
            chunks=retrieved_chunks,
            model_name=reranker_model,
            top_k=target_top_k,
        )
        return reranked_chunks
    return retrieved_chunks


def _assess_statement(
    statement_id: int,
    statement: str,
    query: str,
    answer: str,
    evidence_chunks: List[Dict[str, Any]],
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> StatementAssessment:
    context = "\n\n".join(chunk.get("text", "") for chunk in evidence_chunks) or "No evidence retrieved."
    prompt = (
        "You verify whether a statement is supported by the evidence.\n"
        "Possible statuses: verified, contradicted, insufficient.\n"
        "Respond in JSON using keys 'status' and 'reasoning'.\n\n"
        "Original question:\n{query}\n\n"
        "Initial answer:\n{answer}\n\n"
        "Statement #{statement_id}:\n{statement}\n\n"
        "Evidence:\n{context}\n"
    ).format(statement_id=statement_id, statement=statement, context=context, query=query, answer=answer)

    return generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=StatementAssessment,
    )


def _aggregate_final_answer(
    query: str,
    initial_answer: str,
    statements: List[VerificationStatement],
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> str:
    summary_lines = [
        f"- Statement #{stmt.statement_id}: {stmt.status.upper()} — {stmt.text}\n  Reasoning: {stmt.reasoning}"
        for stmt in statements
    ]
    summary_text = "\n".join(summary_lines)
    prompt = (
        "You are producing the final answer after verification.\n"
        "Rephrase the original answer, explicitly noting which statements are verified, contradicted, or insufficient.\n"
        "Be concise and avoid inventing new claims.\n\n"
        "Original question:\n{query}\n\n"
        "Initial answer:\n{answer}\n\n"
        "Verification summary:\n{summary}\n"
    ).format(query=query, answer=initial_answer, summary=summary_text)

    return generate_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
    ).strip()


def chain_of_verification(
    query: str,
    initial_answer: str,
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
    max_statements: int = 3,
    iterations: int = 1,
) -> ChainVerificationOutput:
    start = time.perf_counter()
    logger.info("CHAIN_VERIFY | start | statements=%d iterations=%d", max_statements, iterations)
    statements_text = _generate_statements(
        query=query,
        answer=initial_answer,
        provider=provider,
        llm_model=llm_model,
        temperature=temperature,
        max_statements=max(2, max_statements),
    )
    statements = [
        VerificationStatement(statement_id=idx + 1, text=text) for idx, text in enumerate(statements_text)
    ]

    total_iterations = 0
    evidences: Dict[int, List[Dict[str, Any]]] = {stmt.statement_id: [] for stmt in statements}

    for iteration in range(iterations):
        total_iterations += 1
        logger.info("CHAIN_VERIFY | iteration %d/%d", iteration + 1, iterations)
        for stmt in statements:
            if stmt.status == "verified":
                continue
            evidence = _retrieve_evidence(
                statement=stmt.text,
                embeddings=embeddings,
                chunks=chunks,
                index_map=index_map,
                embedding_provider=embedding_provider or provider,
                embedding_model=embedding_model,
                k=k,
                threshold=threshold,
                use_hybrid=use_hybrid,
                lexical_weight=lexical_weight,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
                rerank_top_k=rerank_top_k,
            )
            evidences[stmt.statement_id] = evidence
            assessment = _assess_statement(
                statement_id=stmt.statement_id,
                statement=stmt.text,
                query=query,
                answer=initial_answer,
                evidence_chunks=evidence,
                provider=provider,
                llm_model=llm_model,
                temperature=temperature,
            )
            stmt.status = assessment.status
            stmt.reasoning = assessment.reasoning
            stmt.supporting_chunks = evidence
            stmt.iterations += 1
        if all(stmt.status != "insufficient" for stmt in statements):
            break

    refined_answer = _aggregate_final_answer(
        query=query,
        initial_answer=initial_answer,
        statements=statements,
        provider=provider,
        llm_model=llm_model,
        temperature=temperature,
    )

    metadata = {
        "total_ms": (time.perf_counter() - start) * 1000,
        "iterations": total_iterations,
    }
    logger.success("CHAIN_VERIFY | completed | total_ms=%.2f", metadata["total_ms"])
    return ChainVerificationOutput(
        original_answer=initial_answer,
        refined_answer=refined_answer,
        statements=statements,
        iterations=total_iterations,
        metadata=metadata,
    )
