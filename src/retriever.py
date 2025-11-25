from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

logger = _get_logger(__name__) if callable(_get_logger) else None
if logger is None:  
    import logging

    logger = logging.getLogger("RETRIEVE")


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": {"rank": self.rank, "similarity": self.score},
        }


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([])
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("Query embedding norm is zero; cannot compute cosine similarity.")
    matrix_norms = np.linalg.norm(matrix, axis=1)
    valid = matrix_norms != 0
    similarities = np.zeros(len(matrix), dtype=float)
    similarities[valid] = np.dot(matrix[valid], query) / (matrix_norms[valid] * query_norm)
    return similarities


def retrieve(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    chunks: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings matrix must be 2-dimensional.")
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be 1-dimensional.")
    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("Query embedding dimension does not match embeddings matrix.")
    if k <= 0:
        raise ValueError("k must be positive.")

    logger.info("RETRIEVE | start | vectors=%d k=%d threshold=%.2f", len(embeddings), k, threshold)
    logger.debug(
        "RETRIEVE | embedding shapes | query=%s embeddings=%s",
        query_embedding.shape,
        embeddings.shape,
    )

    similarities = _cosine_similarity(query_embedding, embeddings)
    if similarities.size == 0:
        logger.warning("RETRIEVE | similarity computation returned empty array.")
        return []

    logger.debug(
        "RETRIEVE | similarity stats | min=%.4f max=%.4f mean=%.4f",
        float(np.min(similarities)),
        float(np.max(similarities)),
        float(np.mean(similarities)),
    )

    filtered_indices = np.where(similarities >= threshold)[0]
    logger.info(
        "RETRIEVE | threshold filtering | threshold=%.2f passed=%d",
        threshold,
        filtered_indices.size,
    )

    if filtered_indices.size == 0:
        logger.warning("RETRIEVE | no chunks passed threshold.")
        return []

    sorted_indices = filtered_indices[np.argsort(similarities[filtered_indices])[::-1]]
    top_indices = sorted_indices[:k]

    id_to_chunk: Dict[str, Dict[str, Any]] = {chunk["chunk_id"]: chunk for chunk in chunks}
    results: List[RetrievalResult] = []
    selected_metadata: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_indices):
        chunk_id = next((cid for cid, c_idx in index_map.items() if c_idx == int(idx)), None)
        if not chunk_id:
            continue
        chunk_data = id_to_chunk.get(chunk_id, {"text": ""})
        score = float(similarities[idx])
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=chunk_data.get("text", ""),
                score=score,
                rank=rank,
            )
        )
        selected_metadata.append({"chunk_id": chunk_id, "score": round(score, 4)})

    logger.info("RETRIEVE | top_k selected | %s", selected_metadata)
    return [result.to_dict() for result in results]


if __name__ == "__main__":  
    demo_embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.5], [0.0, 0.4, 0.1]])
    demo_chunks = [
        {"chunk_id": "c1", "text": "First chunk."},
        {"chunk_id": "c2", "text": "Second chunk."},
        {"chunk_id": "c3", "text": "Third chunk."},
    ]
    demo_index = {chunk["chunk_id"]: idx for idx, chunk in enumerate(demo_chunks)}
    query = np.array([0.2, 0.1, 0.4])
    retrieved = retrieve(query, demo_embeddings, demo_index, demo_chunks, k=2, threshold=0.0)
    print(retrieved)
