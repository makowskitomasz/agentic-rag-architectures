from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

logger = _get_logger(__name__) if callable(_get_logger) else None
if logger is None:
    import logging

    logger = logging.getLogger("RERANK")


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MODEL_CACHE: Dict[Tuple[str, int], Any] = {}


def _load_cross_encoder(model_name: str, max_length: int) -> Any:
    cache_key = (model_name, max_length)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("sentence-transformers package is required for cross-encoder reranking") from exc

    logger.info("RERANK | loading cross-encoder model=%s max_length=%d", model_name, max_length)
    model = CrossEncoder(model_name, max_length=max_length)
    _MODEL_CACHE[cache_key] = model
    return model


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: str | None = None,
    top_k: int | None = None,
    max_length: int = 512,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    if not query.strip() or not chunks:
        return chunks

    chosen_model = model_name or DEFAULT_RERANKER_MODEL
    logger.info("RERANK | start | chunks=%d model=%s top_k=%s", len(chunks), chosen_model, top_k or "all")

    if top_k is not None and top_k <= 0:
        logger.warning("RERANK | top_k <= 0 provided; returning no chunks.")
        return []

    try:
        model = _load_cross_encoder(chosen_model, max_length=max_length)
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        start = time.perf_counter()
        scores = model.predict(pairs, batch_size=batch_size)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:  # pragma: no cover - runtime/torch failures
        logger.warning("RERANK | failed | returning original chunks | error=%s", exc)
        return chunks

    score_array = np.array(scores, dtype=float).reshape(-1)
    indices = np.argsort(score_array)[::-1]
    if top_k is not None and top_k > 0:
        indices = indices[: min(top_k, len(indices))]

    if indices.size == 0:
        logger.warning("RERANK | no scores available for reranking.")
        return []

    reranked: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indices):
        chunk = dict(chunks[int(idx)])
        metadata = dict(chunk.get("metadata") or {})
        metadata["pre_rerank_score"] = chunk.get("score")
        metadata["rerank_score"] = float(score_array[idx])
        metadata["rerank_rank"] = rank
        chunk["metadata"] = metadata
        chunk["score"] = float(score_array[idx])
        reranked.append(chunk)

    logger.info(
        "RERANK | completed | returned=%d time=%.2f ms score_range=(%.4f, %.4f)",
        len(reranked),
        elapsed_ms,
        float(score_array[indices[-1]]),
        float(score_array[indices[0]]),
    )
    return reranked


if __name__ == "__main__":
    dummy_chunks = [
        {"chunk_id": "c1", "text": "Acceleration relies on hip drive.", "score": 0.58, "metadata": {}},
        {"chunk_id": "c2", "text": "Top speed requires relaxation.", "score": 0.55, "metadata": {}},
    ]
    try:
        ranked = rerank_chunks("How do sprinters accelerate?", dummy_chunks, top_k=2)
        for chunk in ranked:
            print(chunk["chunk_id"], chunk["score"])
    except RuntimeError as exc:
        print(exc)
