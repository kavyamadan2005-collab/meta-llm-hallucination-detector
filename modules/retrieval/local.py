"""Persistent local FAISS semantic retrieval."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from modules.retrieval.chunking import chunk_documents, load_documents
from modules.retrieval.embeddings import SentenceTransformerEmbedder
from modules.retrieval.interfaces import Embedder
from modules.retrieval.models import DocumentChunk, Evidence

logger = logging.getLogger(__name__)


class LocalFaissRetriever:
    """Build and query a normalized inner-product FAISS index over local documents."""

    source_name = "local"

    def __init__(
        self,
        index_directory: str | Path = "data/index",
        embedder: Embedder | None = None,
        min_similarity: float = 0.0,
    ) -> None:
        self.index_directory = Path(index_directory)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.min_similarity = min_similarity
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []
        self._load_persisted()

    @property
    def index_path(self) -> Path:
        return self.index_directory / "evidence.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_directory / "metadata.json"

    def build(self, documents_directory: str | Path = "data/documents", chunk_size: int = 500, overlap: int = 50) -> int:
        """Load, chunk, embed, index, and persist local documents once."""
        self.chunks = chunk_documents(load_documents(documents_directory), chunk_size, overlap)
        if not self.chunks:
            self.index = None
            self._persist()
            return 0
        vectors = self.embedder.encode([chunk.text for chunk in self.chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(self.chunks):
            raise ValueError("embedder must return one two-dimensional vector per chunk")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._persist()
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        """Return thresholded local evidence ranked by cosine similarity."""
        if not query.strip() or self.index is None or self.index.ntotal == 0:
            return []
        limit = max(1, top_k or 5)
        query_vector = self.embedder.encode([query])
        faiss.normalize_L2(query_vector)
        scores, positions = self.index.search(query_vector, min(limit, self.index.ntotal))
        timestamp = datetime.now(timezone.utc)
        evidence: list[Evidence] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or float(score) < self.min_similarity:
                continue
            chunk = self.chunks[int(position)]
            evidence.append(Evidence(
                evidence_id=f"local:{chunk.chunk_id}",
                claim_or_query=query,
                text=chunk.text,
                source_title=chunk.document_name,
                source_url=f"local://{chunk.document_name}",
                relevance_score=float(score),
                retrieval_timestamp=timestamp,
            ))
        return evidence

    def _persist(self) -> None:
        self.index_directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        elif self.index_path.exists():
            self.index_path.unlink()
        chunk_records = [
            chunk.model_dump() if hasattr(chunk, "model_dump") else chunk.dict()
            for chunk in self.chunks
        ]
        self.metadata_path.write_text(
            json.dumps(chunk_records, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _load_persisted(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return
        try:
            self.index = faiss.read_index(str(self.index_path))
            records = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk(**record) for record in records]
            if self.index.ntotal != len(self.chunks):
                logger.warning("Ignoring FAISS index with unsynchronized metadata")
                self.index = None
                self.chunks = []
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            logger.warning("Could not load persisted retrieval index: %s", type(exc).__name__)
            self.index = None
            self.chunks = []