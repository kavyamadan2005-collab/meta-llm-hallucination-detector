"""Replaceable retrieval and embedding interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from modules.retrieval.models import Evidence


class Retriever(Protocol):
    """Interface shared by local, external, and hybrid retrievers."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        """Return available evidence ranked by relevance."""


class Embedder(Protocol):
    """Minimal embedding interface used by the FAISS index."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a two-dimensional float array."""