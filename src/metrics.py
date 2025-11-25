from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

LOGGER = _get_logger(__name__) if callable(_get_logger) else logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"[A-Za-z]+")
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def _tokenize(text: str) -> List[str]:
    return [match.lower() for match in WORD_PATTERN.findall(text)]


def _chunk_text(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("text", "") or "")


def extract_keywords(question: str) -> List[str]:
    seen = set()
    keywords: List[str] = []
    for token in _tokenize(question):
        if len(token) < 4 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def keyword_hits(text: str, keywords: Sequence[str]) -> bool:
    if not keywords:
        return False
    text_tokens = set(_tokenize(text))
    return any(keyword in text_tokens for keyword in keywords)


def precision_recall_k(
    query: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    all_chunks: Sequence[Dict[str, Any]],
    k: int,
) -> Tuple[float, float]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    keywords = extract_keywords(query)
    if not keywords:
        return 0.0, 0.0

    top_k = list(retrieved_chunks[:k])
    relevant_in_topk = sum(1 for chunk in top_k if keyword_hits(_chunk_text(chunk), keywords))
    total_relevant = sum(1 for chunk in all_chunks if keyword_hits(_chunk_text(chunk), keywords))

    precision = relevant_in_topk / float(k)
    recall = relevant_in_topk / float(total_relevant) if total_relevant > 0 else 0.0
    LOGGER.debug(
        "Keyword precision/recall@%d: relevant_top=%d total_relevant=%d precision=%.3f recall=%.3f",
        k,
        relevant_in_topk,
        total_relevant,
        precision,
        recall,
    )
    return precision, recall


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Cosine similarity expects 1D vectors.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Cosine similarity requires equal vector lengths.")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _similarity_against_answer(
    chunk_ids: Iterable[str],
    answer_embedding: np.ndarray,
    embeddings: np.ndarray,
    index_map: Dict[str, int],
) -> Dict[str, float]:
    similarities: Dict[str, float] = {}
    for chunk_id in chunk_ids:
        idx = index_map.get(chunk_id)
        if idx is None or idx >= embeddings.shape[0]:
            continue
        chunk_vector = embeddings[idx]
        similarities[chunk_id] = cosine(answer_embedding, chunk_vector)
    return similarities


def semantic_precision_recall_k(
    answer_embedding: np.ndarray,
    retrieved_chunks: Sequence[Dict[str, Any]],
    all_chunks: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    k: int,
    threshold: float = 0.55,
) -> Tuple[float, float]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings matrix must be 2-dimensional.")
    if answer_embedding.ndim != 1:
        raise ValueError("Answer embedding must be 1-dimensional.")
    if embeddings.shape[1] != answer_embedding.shape[0]:
        raise ValueError("Embedding dimensionality mismatch.")

    all_ids = [str(chunk.get("chunk_id", "")) for chunk in all_chunks if chunk.get("chunk_id")]
    similarities = _similarity_against_answer(all_ids, answer_embedding, embeddings, index_map)
    relevant_chunks = {cid for cid, score in similarities.items() if score >= threshold}
    total_relevant = len(relevant_chunks)

    top_ids = [
        str(chunk.get("chunk_id", ""))
        for chunk in list(retrieved_chunks[:k])
        if chunk.get("chunk_id")
    ]
    relevant_top = sum(1 for cid in top_ids if similarities.get(cid, 0.0) >= threshold)

    precision = relevant_top / float(k)
    recall = relevant_top / float(total_relevant) if total_relevant > 0 else 0.0

    LOGGER.debug(
        "Semantic precision/recall@%d threshold=%.2f relevant_top=%d total_relevant=%d precision=%.3f recall=%.3f",
        k,
        threshold,
        relevant_top,
        total_relevant,
        precision,
        recall,
    )
    return precision, recall


def grounding_score(answer: str, chunks: Sequence[Dict[str, Any]]) -> float:
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_text = " ".join(_chunk_text(chunk) for chunk in chunks)
    context_tokens = _tokenize(context_text)
    if not context_tokens:
        return 0.0
    answer_counts: Dict[str, int] = {}
    for token in answer_tokens:
        answer_counts[token] = answer_counts.get(token, 0) + 1
    context_counts: Dict[str, int] = {}
    for token in context_tokens:
        context_counts[token] = context_counts.get(token, 0) + 1
    overlap = 0
    for token, count in answer_counts.items():
        overlap += min(count, context_counts.get(token, 0))
    score = overlap / float(len(answer_tokens))
    LOGGER.debug("Grounding score: overlap=%d tokens=%d score=%.3f", overlap, len(answer_tokens), score)
    return score


def estimate_tokens(text: str) -> int:
    return len(text.split())


def _demo() -> None:
    LOGGER.setLevel(logging.INFO)
    question = "How does forward lean improve sprint acceleration efficiency?"
    chunks = [
        {"chunk_id": "c1", "text": "Forward lean lowers the center of mass and aids acceleration."},
        {"chunk_id": "c2", "text": "Cadence increases to 4.8 strides per second within 30 meters."},
        {"chunk_id": "c3", "text": "Recovery drills emphasize knee drive and dorsiflexion."},
    ]
    query_retrieved = chunks[:2]
    precision, recall = precision_recall_k(question, query_retrieved, chunks, k=2)
    LOGGER.info("Keyword precision=%.2f recall=%.2f", precision, recall)

    answer_embedding = np.array([0.2, 0.3, 0.5], dtype=float)
    embeddings = np.array([[0.2, 0.3, 0.5], [0.1, 0.3, 0.4], [0.9, 0.1, 0.0]], dtype=float)
    index_map = {"c1": 0, "c2": 1, "c3": 2}
    sem_precision, sem_recall = semantic_precision_recall_k(
        answer_embedding,
        query_retrieved,
        chunks,
        embeddings,
        index_map,
        k=2,
    )
    LOGGER.info("Semantic precision=%.2f recall=%.2f", sem_precision, sem_recall)

    grounding = grounding_score("Forward lean improves early acceleration efficiency.", query_retrieved)
    LOGGER.info("Grounding score=%.2f", grounding)
    LOGGER.info("Answer token estimate=%d", estimate_tokens("Forward lean improves early acceleration efficiency."))


if __name__ == "__main__":
    _demo()
