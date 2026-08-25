"""Deterministic local document loading and chunking."""

from __future__ import annotations

from pathlib import Path

from modules.retrieval.models import DocumentChunk


def load_documents(directory: str | Path) -> list[tuple[str, str]]:
    """Load supported text files in stable path order."""
    root = Path(directory)
    if not root.exists():
        return []
    documents = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append((path.name, text))
    return documents


def chunk_text(
    text: str,
    document_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Split text deterministically by characters while retaining order."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(DocumentChunk(
                chunk_id=f"{document_name}:{chunk_index}",
                document_name=document_name,
                chunk_index=chunk_index,
                text=chunk,
            ))
            chunk_index += 1
        start += chunk_size - overlap
    return chunks


def chunk_documents(
    documents: list[tuple[str, str]], chunk_size: int = 500, overlap: int = 50
) -> list[DocumentChunk]:
    """Chunk all loaded documents in deterministic document and chunk order."""
    chunks: list[DocumentChunk] = []
    for document_name, text in documents:
        chunks.extend(chunk_text(text, document_name, chunk_size, overlap))
    return chunks