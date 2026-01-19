from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("RAG")

try:
    from src.retriever import retrieve
except ImportError as exc:  
    raise RuntimeError("Retriever module is required") from exc

try:
    from src.reranker import rerank_chunks
except ImportError as exc:  
    raise RuntimeError("Reranker module is required") from exc

try:
    from src.llm_orchestrator import generate_answer
except ImportError as exc:  
    raise RuntimeError("LLM orchestrator module is required") from exc

try:
    from src.embedder import generate_query_embedding
except ImportError as exc:  
    raise RuntimeError("Embedder module is required for query embeddings") from exc

try:
    from src.config import get_embedding_config, get_llm_config
except ImportError as exc:  
    raise RuntimeError("Configuration module is required for RAG pipeline") from exc

try:
    from src.agents.chain_of_verification import chain_of_verification
except ImportError:
    chain_of_verification = None

try:
    from src.agents.active_retrieval import active_retrieval
except ImportError:
    active_retrieval = None

try:
    from src.agents.marag import run_marag
except ImportError:
    run_marag = None

try:
    from src.agents.madam_rag import run_madam_rag
except ImportError:
    run_madam_rag = None


def rag(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    provider: str | None = None,
    embedding_model: str | None = None,
    llm_model: str | None = None,
    k: int = 5,
    threshold: float = 0.5,
    use_hybrid: bool = True,
    lexical_weight: float = 0.5,
    use_reranker: bool = False,
    reranker_model: str | None = None,
    rerank_top_k: int | None = None,
    use_chain_of_verification: bool = False,
    verification_iterations: int = 1,
    verification_statements: int = 3,
    use_active_retrieval: bool = False,
    active_iterations: int = 3,
    active_sufficiency_threshold: float = 0.8,
    use_marag: bool = False,
    marag_iterations: int = 2,
    marag_roles: List[str] | None = None,
    marag_sufficiency_threshold: float = 0.8,
    use_madam_rag: bool = False,
    madam_followup_rounds: int = 0,
    use_routing_agent: bool = False,
) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings matrix must be 2-dimensional")
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    pipeline_start = time.perf_counter()
    logger.info("RAG | start | query_len=%d chunks=%d", len(query), len(chunks))

    embedding_config = get_embedding_config(provider)
    base_embedding_provider = embedding_config.provider
    base_embedding_model = embedding_model or embedding_config.model
    llm_config = get_llm_config(provider)

    if use_routing_agent:
        from src.agents.routing_rag import routing_rag as run_routing_agent
        from src.agents.self_reflective_rag import self_reflect_rag
        from src.agents.query_decomposition_rag import query_decomposition_rag
        from src.agents.marag import run_marag
        from src.agents.madam_rag import run_madam_rag

        def _exec_rag(profile, **kwargs):
            return rag(
                query=query,
                chunks=chunks,
                embeddings=embeddings,
                index_map=index_map,
                provider=base_embedding_provider,
                embedding_model=base_embedding_model,
                llm_model=llm_model or llm_config.model,
                k=k,
                threshold=threshold,
                use_hybrid=use_hybrid,
                lexical_weight=lexical_weight,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
                rerank_top_k=rerank_top_k,
                use_chain_of_verification=kwargs.get("use_chain_of_verification", False),
                verification_iterations=kwargs.get("verification_iterations", verification_iterations),
                verification_statements=kwargs.get("verification_statements", verification_statements),
                use_active_retrieval=kwargs.get("use_active_retrieval", False),
                active_iterations=kwargs.get("active_iterations", active_iterations),
                active_sufficiency_threshold=kwargs.get("active_sufficiency_threshold", active_sufficiency_threshold),
                use_marag=False,
                use_madam_rag=False,
                use_routing_agent=False,
            )

        def _exec_vanilla(profile, decision):
            return _exec_rag(profile)

        def _exec_chain(profile, decision):
            return _exec_rag(
                profile,
                use_chain_of_verification=True,
                verification_iterations=max(1, decision.iterations),
                verification_statements=max(2, decision.iterations + 1),
            )

        def _exec_active(profile, decision):
            return _exec_rag(
                profile,
                use_active_retrieval=True,
                active_iterations=max(1, decision.iterations),
                active_sufficiency_threshold=0.75,
            )

        def _exec_self_reflective(profile, decision):
            query_embedding = generate_query_embedding(
                query,
                provider=base_embedding_provider,
                model=base_embedding_model,
            )
            return self_reflect_rag(
                query=query,
                chunks=chunks,
                query_embedding=query_embedding,
                embeddings=embeddings,
                index_map=index_map,
                embedding_provider=base_embedding_provider,
                embedding_model=base_embedding_model,
                provider=llm_config.provider,
                llm_model=llm_model,
            )

        def _exec_query_decomp(profile, decision):
            return query_decomposition_rag(
                query=query,
                chunks=chunks,
                embeddings=embeddings,
                index_map=index_map,
                provider=llm_config.provider,
                llm_model=llm_model,
                embedding_provider=base_embedding_provider,
                embedding_model=base_embedding_model,
                k=k,
                threshold=threshold,
            )

        def _exec_marag(profile, decision):
            if run_marag is None:
                raise RuntimeError("MARAG agent is not available.")
            return run_marag(
                query=query,
                chunks=chunks,
                embeddings=embeddings,
                index_map=index_map,
                provider=llm_config.provider,
                llm_model=llm_model,
                embedding_provider=base_embedding_provider,
                embedding_model=base_embedding_model,
                k=k,
                threshold=threshold,
                use_hybrid=use_hybrid,
                lexical_weight=lexical_weight,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
                rerank_top_k=rerank_top_k,
                researcher_iterations=max(1, decision.iterations),
            )

        def _exec_madam(profile, decision):
            if run_madam_rag is None:
                raise RuntimeError("MADAM-RAG agent is not available.")
            return run_madam_rag(
                query=query,
                chunks=chunks,
                embeddings=embeddings,
                index_map=index_map,
                provider=llm_config.provider,
                llm_model=llm_model,
                embedding_provider=base_embedding_provider,
                embedding_model=base_embedding_model,
                k=k,
                threshold=threshold,
                use_hybrid=use_hybrid,
                lexical_weight=lexical_weight,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
                rerank_top_k=rerank_top_k,
                temperature=1.0,
                followup_rounds=min(2, decision.followup_rounds),
            )

        executors = {
            "vanilla": _exec_vanilla,
            "chain_verification": _exec_chain,
            "active_retrieval": _exec_active,
            "self_reflective": _exec_self_reflective,
            "query_decomposition": _exec_query_decomp,
            "marag": _exec_marag,
            "madam_rag": _exec_madam,
        }

        return run_routing_agent(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            llm_provider=llm_config.provider,
            llm_model=llm_model or llm_config.model,
            executors=executors,
        )

    if use_marag:
        from src.agents.marag import run_marag
        if run_marag is None:
            raise RuntimeError("MARAG agent is not available in this environment.")
        marag_result = run_marag(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=llm_config.provider,
            llm_model=llm_model or llm_config.model,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_model or embedding_config.model,
            k=k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
            temperature=1.0,
            researcher_iterations=marag_iterations,
            sufficiency_threshold=marag_sufficiency_threshold,
            role_sequence=marag_roles,
        )
        return marag_result
    if use_madam_rag:
        from src.agents.madam_rag import run_madam_rag
        if run_madam_rag is None:
            raise RuntimeError("MADAM-RAG agent is not available in this environment.")
        madam_result = run_madam_rag(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=llm_config.provider,
            llm_model=llm_model or llm_config.model,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_model or embedding_config.model,
            k=k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
            temperature=1.0,
            followup_rounds=madam_followup_rounds,
        )
        return madam_result

    embedding_start = time.perf_counter()
    query_embedding = generate_query_embedding(
        query,
        provider=embedding_config.provider,
        model=embedding_model or embedding_config.model,
    )
    embedding_elapsed = (time.perf_counter() - embedding_start) * 1000
    logger.info(
        "RAG | embedding | provider=%s model=%s time=%.2f ms",
        embedding_config.provider,
        embedding_model or embedding_config.model,
        embedding_elapsed,
    )

    active_metadata = None
    retrieval_start = time.perf_counter()
    if use_active_retrieval:
        from src.agents.active_retrieval import active_retrieval
        if active_retrieval is None:
            raise RuntimeError("Active retrieval agent is not available in this environment.")
        active_metadata = active_retrieval(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=llm_config.provider,
            llm_model=llm_model or llm_config.model,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_model or embedding_config.model,
            k=k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
            temperature=1.0,
            max_iterations=active_iterations,
            sufficiency_threshold=active_sufficiency_threshold,
            query_embedding=query_embedding,
        )
        retrieved_chunks = active_metadata["chunks"]
        retrieval_elapsed = active_metadata.get("total_ms", (time.perf_counter() - retrieval_start) * 1000)
        logger.info(
            "RAG | active retrieval | iterations=%d time=%.2f ms",
            len(active_metadata.get("logs", [])),
            retrieval_elapsed,
        )
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
        retrieval_elapsed = (time.perf_counter() - retrieval_start) * 1000
        logger.info(
            "RAG | retrieve | retrieved=%d time=%.2f ms threshold=%.2f",
            len(retrieved_chunks),
            retrieval_elapsed,
            threshold,
        )

    reranked_chunks = retrieved_chunks
    rerank_elapsed = 0.0
    if use_reranker and retrieved_chunks:
        rerank_start = time.perf_counter()
        target_top_k = rerank_top_k if rerank_top_k is not None else k
        reranked_chunks = rerank_chunks(
            query=query,
            chunks=retrieved_chunks,
            model_name=reranker_model,
            top_k=target_top_k,
        )
        rerank_elapsed = (time.perf_counter() - rerank_start) * 1000
        logger.info(
            "RAG | rerank | reordered=%d time=%.2f ms model=%s",
            len(reranked_chunks),
            rerank_elapsed,
            reranker_model or "default",
        )

    llm_start = time.perf_counter()
    answer = generate_answer(
        query=query,
        context_chunks=reranked_chunks,
        provider=llm_config.provider,
        model=llm_model or llm_config.model,
        temperature=1.0,
    )
    llm_elapsed = (time.perf_counter() - llm_start) * 1000
    logger.info(
        "RAG | llm | provider=%s model=%s time=%.2f ms",
        llm_config.provider,
        llm_model or llm_config.model,
        llm_elapsed,
    )

    verification_result = None
    verification_elapsed = 0.0
    if use_chain_of_verification:
        if chain_of_verification is None:
            raise RuntimeError("Chain-of-verification agent is not available in this environment.")
        verification_start = time.perf_counter()
        verification_result = chain_of_verification(
            query=query,
            initial_answer=answer,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=llm_config.provider,
            llm_model=llm_model or llm_config.model,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_model or embedding_config.model,
            k=k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_k=rerank_top_k,
            temperature=1.0,
            max_statements=verification_statements,
            iterations=max(1, verification_iterations),
        )
        verification_elapsed = (time.perf_counter() - verification_start) * 1000
        logger.info(
            "RAG | chain_of_verification | statements=%d time=%.2f ms",
            len(verification_result.statements),
            verification_elapsed,
        )
        answer = verification_result.refined_answer

    answer_tokens = len(answer.split())
    total_elapsed = (time.perf_counter() - pipeline_start) * 1000

    logger.info("RAG | answer | length=%d tokens≈%d", len(answer), answer_tokens)
    logger.success("RAG | complete | total_time=%.2f ms", total_elapsed)

    return {
        "query": query,
        "answer": answer,
        "chunks": reranked_chunks,
        "tokens_estimated": answer_tokens,
        "time_ms": total_elapsed,
        "verification": verification_result.model_dump() if verification_result else None,
        "verification_time_ms": verification_elapsed,
        "active_retrieval": active_metadata,
        "provider": llm_config.provider,
        "model": llm_model or llm_config.model,
    }


if __name__ == "__main__":  
    chunks_path = Path("data/future_poland/processed/chunks.json")
    embeddings_path = Path("embeddings/future_poland/embeddings.npy")
    index_path = Path("embeddings/future_poland/embedding_index.json")

    if chunks_path.exists() and embeddings_path.exists() and index_path.exists():
        loaded_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embedding_matrix = np.load(embeddings_path)
        index_mapping = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        loaded_chunks = [
            {"chunk_id": "c1", "text": "Sample context about sprinting mechanics."},
            {"chunk_id": "c2", "text": "Acceleration phases require coordination."},
        ]
        embedding_matrix = np.random.rand(2, 768)
        index_mapping = {chunk["chunk_id"]: idx for idx, chunk in enumerate(loaded_chunks)}

    response = rag(
        query="How does acceleration influence sprinting mechanics?",
        chunks=loaded_chunks,
        embeddings=embedding_matrix,
        index_map=index_mapping,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))
