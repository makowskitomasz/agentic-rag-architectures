from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import numpy as np

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

try:
    from src.retriever import retrieve
except ImportError as exc:  
    raise RuntimeError("Retriever module is required") from exc

try:
    from src.llm_orchestrator import generate_answer, generate_structured_answer
except ImportError as exc:  
    raise RuntimeError("LLM orchestrator module is required") from exc

try:
    from src.reranker import rerank_chunks
except ImportError as exc:  
    raise RuntimeError("Reranker module is required") from exc

try:
    from src.models.self_reflective_models import CritiqueModel, SelfReflectiveRagOutput
except ImportError as exc:  
    raise RuntimeError("Models module is required") from exc

try:
    from src.agents.active_retrieval import active_retrieval
except ImportError:
    active_retrieval = None

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("REFLECT")


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    return "\n\n".join(f"[{idx + 1}] {chunk.get('text', '').strip()}" for idx, chunk in enumerate(chunks))


def _critique_answer(
    answer: str,
    chunks: List[Dict[str, Any]],
    provider: str,
    model: str | None,
    temperature: float,
) -> CritiqueModel:
    prompt = (
        "You are a critique agent. Analyze the answer strictly based on the provided context.\n"
        "Identify:\n\n"
        "1. missing_context: which parts of the context were not mentioned but should be\n"
        "2. conflicts: contradictions between the answer and the context\n"
        "3. logic_issues: logical inconsistencies or incoherent reasoning\n"
        "4. hallucinations: statements not grounded in the context\n"
        "5. precision_warnings: vague or imprecise claims\n"
        "6. language_problems: unclear or poorly structured parts\n"
        "7. reasoning_gaps: steps that skip necessary justification\n\n"
        "Return ONLY valid JSON:\n"
        "{{...}}\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}"
    ).format(context=_format_context(chunks), answer=answer)

    try:
        critique = generate_structured_answer(
            query=prompt,
            context_chunks=[],
            provider=provider,
            model=model,
            temperature=temperature,
            response_model=CritiqueModel,
        )
        logger.info("REFLECT | critique parsed successfully.")
        return critique
    except Exception as exc:
        logger.warning("REFLECT | critique structured call failed; returning empty model. error=%s", exc)
        return CritiqueModel()


def _refine_answer(
    query: str,
    initial_answer: str,
    critique: CritiqueModel,
    chunks: List[Dict[str, Any]],
    provider: str,
    model: str | None,
    temperature: float,
) -> str:
    prompt = (
        "You are a refinement agent. Improve the initial answer using:\n\n"
        "- structured critique,\n"
        "- original context,\n"
        "- the user query.\n\n"
        "Rewrite the answer so that it is:\n\n"
        "- fully grounded in context,\n"
        "- logically consistent,\n"
        "- precise,\n"
        "- free from hallucinations,\n"
        "- better structured.\n\n"
        "Return ONLY the refined answer text.\n\n"
        "Structured Critique:\n{critique}\n\n"
        "Context:\n{context}\n\n"
        "Initial Answer:\n{initial}\n"
    ).format(
        critique=critique.model_dump_json(indent=2, ensure_ascii=False),
        context=_format_context(chunks),
        initial=initial_answer,
    )

    refined = generate_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=model,
        temperature=temperature,
    )
    return refined.strip()


def self_reflect_rag(
    query: str,
    chunks: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    provider: str = "openai",
    llm_model: str | None = None,
    temperature: float = 1.0,
    k: int = 5,
    threshold: float = 0.5,
    use_hybrid: bool = True,
    lexical_weight: float = 0.5,
    use_reranker: bool = False,
    reranker_model: str | None = None,
    rerank_top_k: int | None = None,
    use_active_retrieval: bool = False,
    active_iterations: int = 3,
    active_sufficiency_threshold: float = 0.8,
) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    logger.info("REFLECT | start | query_len=%d chunks=%d", len(query), len(chunks))

    retrieval_start = time.perf_counter()
    if use_active_retrieval:
        if active_retrieval is None:
            raise RuntimeError("Active retrieval agent is not available.")
        if embedding_provider is None or embedding_model is None:
            raise ValueError("embedding_provider and embedding_model are required when use_active_retrieval=True.")
        active_result = active_retrieval(
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
            max_iterations=active_iterations,
            sufficiency_threshold=active_sufficiency_threshold,
            query_embedding=query_embedding,
        )
        retrieved_chunks = active_result["chunks"]
        logger.info("REFLECT | active retrieval | iterations=%d", len(active_result["logs"]))
        active_time = active_result.get("total_ms", 0.0)
    else:
        retrieved_chunks = retrieve(
            query_embedding=query_embedding,
            embeddings=embeddings,
            index_map=index_map,
            chunks=chunks,
            k=k,
            threshold=threshold,
            query_text=query,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
        )
    retrieval_time = (time.perf_counter() - retrieval_start) * 1000
    logger.info(
        "REFLECT | retrieval | retrieved=%d time=%.2f ms threshold=%.2f",
        len(retrieved_chunks),
        retrieval_time,
        threshold,
    )

    reranked_chunks = retrieved_chunks
    rerank_time = 0.0
    if use_reranker and retrieved_chunks:
        rerank_start = time.perf_counter()
        target_top_k = rerank_top_k if rerank_top_k is not None else k
        reranked_chunks = rerank_chunks(
            query=query,
            chunks=retrieved_chunks,
            model_name=reranker_model,
            top_k=target_top_k,
        )
        rerank_time = (time.perf_counter() - rerank_start) * 1000
        logger.info(
            "REFLECT | rerank | reordered=%d time=%.2f ms model=%s",
            len(reranked_chunks),
            rerank_time,
            reranker_model or "default",
        )

    timings: Dict[str, float] = {
        "initial_ms": 0.0,
        "critique_ms": 0.0,
        "refined_ms": 0.0,
        "rerank_ms": rerank_time,
    }
    if use_active_retrieval:
        timings["active_ms"] = active_time

    start = time.perf_counter()
    initial_answer = generate_answer(
        query=query,
        context_chunks=reranked_chunks,
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )
    timings["initial_ms"] = (time.perf_counter() - start) * 1000
    logger.info("REFLECT | initial answer | time=%.2f ms", timings["initial_ms"])

    start = time.perf_counter()
    critique = _critique_answer(
        answer=initial_answer,
        chunks=reranked_chunks,
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )
    timings["critique_ms"] = (time.perf_counter() - start) * 1000
    logger.info("REFLECT | critique phase | time=%.2f ms", timings["critique_ms"])

    start = time.perf_counter()
    refined_answer = _refine_answer(
        query=query,
        initial_answer=initial_answer,
        critique=critique,
        chunks=reranked_chunks,
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )
    timings["refined_ms"] = (time.perf_counter() - start) * 1000
    logger.info("REFLECT | refinement | time=%.2f ms", timings["refined_ms"])

    final_tokens = len(refined_answer.split())
    logger.info("REFLECT | final answer | length=%d tokens≈%d", len(refined_answer), final_tokens)

    total_time = (time.perf_counter() - pipeline_start) * 1000
    logger.success(
        "REFLECT | summary | total=%.2f ms initial=%.2f critique=%.2f refined=%.2f retrieval=%.2f",
        total_time,
        timings["initial_ms"],
        timings["critique_ms"],
        timings["refined_ms"],
        retrieval_time,
    )

    result = SelfReflectiveRagOutput(
        query=query,
        initial_answer=initial_answer,
        critique=critique,
        refined_answer=refined_answer,
        retrieved_chunks=reranked_chunks,
        timings={**timings, "retrieval_ms": retrieval_time, "total_ms": total_time},
        provider=provider,
        model=llm_model or "default",
    )
    return result.model_dump()


if __name__ == "__main__":  
    dummy_chunks = [
        {"chunk_id": "c1", "text": "Acceleration relies on hip drive and shin angles.", "score": 0.9, "metadata": {}},
        {"chunk_id": "c2", "text": "Cadence increases toward maximal velocity.", "score": 0.85, "metadata": {}},
    ]
    dummy_embeddings = np.random.rand(2, 384)
    dummy_index = {chunk["chunk_id"]: idx for idx, chunk in enumerate(dummy_chunks)}
    dummy_query_embedding = np.random.rand(384)

    result = self_reflect_rag(
        query="How should sprinters manage acceleration?",
        chunks=dummy_chunks,
        query_embedding=dummy_query_embedding,
        embeddings=dummy_embeddings,
        index_map=dummy_index,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
