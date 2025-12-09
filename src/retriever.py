from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

logger = _get_logger(__name__) if callable(_get_logger) else None
if logger is None:  
    import logging

    logger = logging.getLogger("RETRIEVE")


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_']+")


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int
    dense_score: float | None = None
    lexical_score: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "rank": self.rank,
            "similarity": self.score,
        }
        if self.dense_score is not None:
            metadata["dense_score"] = self.dense_score
        if self.lexical_score is not None:
            metadata["lexical_score"] = self.lexical_score
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": metadata,
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


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _bm25_scores(
    query: str,
    documents: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    if not documents:
        return np.array([])
    tokenized_docs = [_tokenize(doc) for doc in documents]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0

    query_tokens = _tokenize(query)
    if not query_tokens:
        return np.zeros(len(documents), dtype=float)

    doc_freq: Dict[str, int] = {}
    doc_term_freqs: List[Counter[str]] = []
    for tokens in tokenized_docs:
        tf = Counter(tokens)
        doc_term_freqs.append(tf)
        for term in tf:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = np.zeros(len(documents), dtype=float)
    num_docs = len(documents)
    query_counter = Counter(query_tokens)
    for term, q_freq in query_counter.items():
        freq_in_docs = doc_freq.get(term, 0)
        if freq_in_docs == 0:
            continue
        idf = math.log((num_docs - freq_in_docs + 0.5) / (freq_in_docs + 0.5) + 1)
        for idx, term_counts in enumerate(doc_term_freqs):
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (doc_lengths[idx] / avgdl if avgdl > 0 else 0))
            scores[idx] += idf * ((freq * (k1 + 1)) / denom) * q_freq
    return scores


def _normalize_cosine(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    normalized = (scores + 1.0) / 2.0
    return np.clip(normalized, 0.0, 1.0)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    max_val = float(np.max(scores))
    min_val = float(np.min(scores))
    if math.isclose(max_val, min_val):
        if max_val <= 0:
            return np.zeros_like(scores)
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)


def retrieve(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    chunks: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = 0.5,
    query_text: str | None = None,
    use_hybrid: bool = True,
    lexical_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings matrix must be 2-dimensional.")
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be 1-dimensional.")
    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("Query embedding dimension does not match embeddings matrix.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if not (0.0 <= lexical_weight <= 1.0):
        raise ValueError("lexical_weight must be between 0 and 1.")

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

    dense_norm = _normalize_cosine(similarities)
    lexical_scores = np.zeros(len(embeddings), dtype=float)
    lexical_enabled = bool(use_hybrid and query_text)
    if lexical_enabled:
        ordered_chunk_texts: List[str] = []
        ordered_indices: List[int] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id"))
            idx = index_map.get(chunk_id)
            if idx is None:
                continue
            ordered_chunk_texts.append(chunk.get("text", ""))
            ordered_indices.append(idx)
        if ordered_chunk_texts:
            bm25_scores = _bm25_scores(query_text or "", ordered_chunk_texts)
            for score, idx in zip(bm25_scores, ordered_indices):
                lexical_scores[idx] = score
            lexical_enabled = bool(np.any(lexical_scores))
        else:
            lexical_enabled = False
    logger.info("RETRIEVE | hybrid enabled=%s weight=%.2f", lexical_enabled, lexical_weight if lexical_enabled else 0.0)

    if lexical_enabled:
        lexical_norm = _normalize_scores(lexical_scores)
        combined_scores = lexical_weight * lexical_norm + (1.0 - lexical_weight) * dense_norm
        ranking_scores = combined_scores
    else:
        ranking_scores = dense_norm

    filtered_indices = np.where(ranking_scores >= threshold)[0]
    logger.info(
        "RETRIEVE | threshold filtering | threshold=%.2f passed=%d",
        threshold,
        filtered_indices.size,
    )

    if filtered_indices.size == 0:
        logger.warning("RETRIEVE | no chunks passed threshold.")
        return []

    sorted_indices = filtered_indices[np.argsort(ranking_scores[filtered_indices])[::-1]]
    top_indices = sorted_indices[:k]

    id_to_chunk: Dict[str, Dict[str, Any]] = {chunk["chunk_id"]: chunk for chunk in chunks}
    index_to_chunk_id: Dict[int, str] = {idx: chunk_id for chunk_id, idx in index_map.items()}
    results: List[RetrievalResult] = []
    selected_metadata: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_indices):
        chunk_id = index_to_chunk_id.get(int(idx))
        if not chunk_id:
            continue
        chunk_data = id_to_chunk.get(chunk_id, {"text": ""})
        score = float(ranking_scores[idx])
        dense_score = float(similarities[idx])
        lexical_score = float(lexical_scores[idx]) if lexical_enabled else None
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=chunk_data.get("text", ""),
                score=score,
                rank=rank,
                dense_score=dense_score,
                lexical_score=lexical_score,
            )
        )
        selected_metadata.append(
            {
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "dense": round(dense_score, 4),
                "lexical": round(lexical_score, 4) if lexical_score is not None else None,
            }
        )

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
    retrieved = retrieve(
        query_embedding=query,
        embeddings=demo_embeddings,
        index_map=demo_index,
        chunks=demo_chunks,
        k=2,
        threshold=0.0,
        query_text="second chunk content",
    )
    print(retrieved)
