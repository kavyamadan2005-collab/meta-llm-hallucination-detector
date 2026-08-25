"""Hybrid local and external evidence retrieval."""

from __future__ import annotations

import logging

from modules.retrieval.external import ExternalRetrieverError
from modules.retrieval.interfaces import Retriever
from modules.retrieval.models import Evidence

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine retrievers while retaining source identity and removing duplicates."""

    def __init__(self, retrievers: list[Retriever]) -> None:
        self.retrievers = retrievers

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        """Return deduplicated evidence in retriever order, limited to top_k."""
        results: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        for retriever in self.retrievers:
            try:
                retrieved = retriever.retrieve(query, top_k)
            except ExternalRetrieverError:
                logger.warning("Optional external retrieval failed; continuing with available evidence")
                continue
            for evidence in retrieved:
                key = (evidence.source_url, evidence.text.strip())
                if key not in seen:
                    seen.add(key)
                    results.append(evidence)
        results.sort(key=lambda evidence: evidence.relevance_score, reverse=True)
        return results[:top_k] if top_k else results