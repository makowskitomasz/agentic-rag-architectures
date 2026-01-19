from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

logger = _get_logger(__name__) if callable(_get_logger) else None
if logger is None:  
    import logging

    logger = logging.getLogger("EMBED")

Chunk = Dict[str, Any]

try:
    from src.config import get_embedding_config
except ImportError as exc:  
    raise RuntimeError("Configuration module is required for embedding generation") from exc


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _embed_batch_openai(batch: List[str], model: str, api_key: str) -> List[List[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:  
        raise RuntimeError("openai package is required for OpenAI embeddings") from exc

    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(model=model, input=batch)
    except Exception as exc:  
        raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

    return [item.embedding for item in response.data]


def _embed_batch_gemini(batch: List[str], model: str, api_key: str) -> List[List[float]]:
    try:
        import google.generativeai as genai
    except ImportError as exc:  
        raise RuntimeError("google-generativeai package is required for Gemini embeddings") from exc

    genai.configure(api_key=api_key)

    embeddings: List[List[float]] = []
    try:
        for text in batch:
            result = genai.embed_content(model=model, content=text, task_type="retrieval_document")
            embeddings.append(result["embedding"])
    except Exception as exc:  
        raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

    return embeddings


def _batch_iterable(items: List[Chunk], batch_size: int) -> Iterable[List[Chunk]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _log_embedding_stats(
    provider_key: str,
    model: str,
    num_chunks: int,
    batch_size: int,
    num_batches: int,
    embedding_dim: int | None,
) -> None:
    logger.info("EMBED | provider=%s model=%s", provider_key, model)
    logger.info(
        "EMBED | chunks=%d batch_size=%d batches=%d embedding_dim=%s",
        num_chunks,
        batch_size,
        num_batches,
        embedding_dim if embedding_dim is not None else "unknown",
    )


def generate_embeddings(
    chunks: List[Chunk],
    provider: str | None = None,
    model: str | None = None,
    batch_size: int = 16,
    output_dir: str = "../embeddings",
) -> Tuple[np.ndarray, Dict[str, int]]:
    if not chunks:
        return np.empty((0, 0), dtype=float), {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    config = get_embedding_config(provider)
    provider_key = config.provider
    chosen_model = model or config.model

    logger.info("EMBED | start | chunks=%d", len(chunks))

    vectors: List[List[float]] = []
    index_map: Dict[str, int] = {}

    if provider_key == "openai":
        def embedder(texts: List[str], mdl: str) -> List[List[float]]:
            logger.debug("EMBED | OpenAI batch size=%d", len(texts))
            return _embed_batch_openai(texts, mdl, config.api_key)
    elif provider_key == "gemini":
        def embedder(texts: List[str], mdl: str) -> List[List[float]]:
            logger.debug("EMBED | Gemini batch size=%d", len(texts))
            return _embed_batch_gemini(texts, mdl, config.api_key)
    else:  
        raise ValueError(f"Unsupported provider: {provider_key}")

    batches = list(_batch_iterable(chunks, batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        texts = [item.get("text", "") for item in batch]
        embeddings = embedder(texts, chosen_model)
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding provider returned mismatched batch size")
        for vec, chunk in zip(embeddings, batch):
            index = len(vectors)
            vectors.append(vec)
            chunk_id = str(chunk.get("chunk_id"))
            index_map[chunk_id] = index
        logger.info("EMBED | processed batch %d/%d (%d chunks)", batch_index, len(batches), len(batch))

    embedding_matrix = np.array(vectors, dtype=float)
    embedding_dim = embedding_matrix.shape[1] if embedding_matrix.size else None
    _log_embedding_stats(provider_key, chosen_model, len(chunks), batch_size, len(batches), embedding_dim)

    output_path = Path(output_dir)
    _ensure_output_dir(output_path)
    embeddings_file = output_path / "embeddings.npy"
    index_file = output_path / "embedding_index.json"

    np.save(embeddings_file, embedding_matrix)
    index_file.write_text(json.dumps(index_map, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("EMBED | saved embeddings -> %s", embeddings_file)
    logger.info("EMBED | saved index map -> %s", index_file)

    return embedding_matrix, index_map


def generate_query_embedding(
    query: str,
    provider: str | None = None,
    model: str | None = None,
) -> np.ndarray:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    config = get_embedding_config(provider)
    provider_key = config.provider
    chosen_model = model or config.model

    if provider_key == "openai":
        embedder = _embed_batch_openai
    elif provider_key == "gemini":
        embedder = _embed_batch_gemini
    else:  
        raise ValueError(f"Unsupported provider: {provider_key}")

    logger.info("EMBED | query embedding | provider=%s model=%s", provider_key, chosen_model)
    vector = embedder([query], chosen_model, config.api_key)[0]
    return np.array(vector, dtype=float)


if __name__ == "__main__":  
    chunks_path = Path("data/future_poland/processed/chunks.json")
    if chunks_path.exists():
        loaded_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    else:
        loaded_chunks = []
    matrix, mapping = generate_embeddings(loaded_chunks, provider="openai") if loaded_chunks else (np.empty((0, 0)), {})
    print({"chunks": len(loaded_chunks), "matrix_shape": matrix.shape, "mapping_size": len(mapping)})
