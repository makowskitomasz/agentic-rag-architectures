from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None


logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger(__name__)

Document = Dict[str, Any]
Chunk = Dict[str, Any]


def _create_chunks_for_document(doc: Document, chunk_size: int, overlap: int) -> List[Chunk]:
    text = doc.get("text", "")
    words = text.split()
    if not words:
        return []

    window = chunk_size
    step = chunk_size - overlap
    chunks: List[Chunk] = []
    for start in range(0, len(words), step):
        end = start + window
        word_slice = words[start:end]
        if not word_slice:
            break
        chunk_text = " ".join(word_slice)
        chunks.append(
            {
                "chunk_id": str(uuid4()),
                "doc_id": doc.get("id"),
                "filename": doc.get("filename"),
                "text": chunk_text,
                "index": len(chunks),
                "word_count": len(word_slice),
            }
        )
        if end >= len(words):
            break
    return chunks


def _write_chunks_to_file(chunks: Iterable[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = list(chunks)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 400,
    overlap: int = 50,
    output_path: str = "../data/processed/chunks.json",
) -> List[Chunk]:
    if not documents:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    all_chunks: List[Chunk] = []
    for doc in documents:
        doc_chunks = _create_chunks_for_document(doc, chunk_size, overlap)
        all_chunks.extend(doc_chunks)

    output_file = Path(output_path)
    _write_chunks_to_file(all_chunks, output_file)

    logger.info("Saved %d chunks to %s", len(all_chunks), output_file)
    return all_chunks


if __name__ == "__main__":
    example_docs = [
        {
            "id": "example",
            "filename": "example.md",
            "text": "Sample text for chunking demonstration.",
            "path": "data/raw/example.md",
        }
    ]
    chunks = chunk_documents(example_docs, chunk_size=5, overlap=1, output_path="data/processed/_temp_chunks.json")
    print(len(chunks))
