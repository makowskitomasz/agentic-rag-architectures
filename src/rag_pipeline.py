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

    retrieval_start = time.perf_counter()
    retrieved_chunks = retrieve(
        query_embedding=query_embedding,
        embeddings=embeddings,
        index_map=index_map,
        chunks=chunks,
        k=k,
        threshold=threshold,
    )
    retrieval_elapsed = (time.perf_counter() - retrieval_start) * 1000
    logger.info(
        "RAG | retrieve | retrieved=%d time=%.2f ms threshold=%.2f",
        len(retrieved_chunks),
        retrieval_elapsed,
        threshold,
    )

    llm_config = get_llm_config(provider)
    llm_start = time.perf_counter()
    answer = generate_answer(
        query=query,
        context_chunks=retrieved_chunks,
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

    answer_tokens = len(answer.split())
    total_elapsed = (time.perf_counter() - pipeline_start) * 1000

    logger.info("RAG | answer | length=%d tokens≈%d", len(answer), answer_tokens)
    logger.info("RAG | complete | total_time=%.2f ms", total_elapsed)

    return {
        "query": query,
        "answer": answer,
        "chunks": retrieved_chunks,
        "tokens_estimated": answer_tokens,
        "time_ms": total_elapsed,
        "provider": llm_config.provider,
        "model": llm_model or llm_config.model,
    }


if __name__ == "__main__":  
    chunks_path = Path("data/processed/chunks.json")
    embeddings_path = Path("embeddings/embeddings.npy")
    index_path = Path("embeddings/embedding_index.json")

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
