"""Utilities for loading markdown documents for retrieval workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None


logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger(__name__)

Document = Dict[str, Any]


def _read_file(file_path: Path) -> str:
    """Return the UTF-8 text contained in the provided file path."""

    return file_path.read_text(encoding="utf-8")


def load_markdown_files(path: str = "data/raw") -> List[Document]:
    """Load markdown files from *path* and return structured document records."""

    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path '{directory}' is not a directory.")

    markdown_files = sorted(directory.glob("*.md"))
    if not markdown_files:
        logger.warning("No markdown files found in %s", directory)
        return []

    documents: List[Document] = []
    for file_path in markdown_files:
        text = _read_file(file_path)
        documents.append(
            {
                "id": str(uuid4()),
                "filename": file_path.name,
                "text": text,
                "path": str(file_path.resolve()),
            }
        )

    return documents


if __name__ == "__main__":
    docs = load_markdown_files()
    print(len(docs))
