"""Composable multi-source evidence retrieval."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from modules.retrieval.interfaces import Retriever
from modules.retrieval.models import Evidence

logger = logging.getLogger(__name__)

_SEMANTIC_WEIGHT = 0.75
_SOURCE_QUALITY_WEIGHT = 0.25
_HIGH_AUTHORITY_DOMAINS = {
    "who.int", "nature.com", "science.org", "sciencemag.org", "reuters.com",
    "apnews.com", "pubmed.ncbi.nlm.nih.gov", "nih.gov", "nasa.gov",
}
_LOW_AUTHORITY_DOMAINS = {
    "quora.com", "reddit.com", "answers.com", "ask.com", "yahoo.com",
}


class MultiSourceRetriever:
    """Query independent sources and combine their evidence defensively."""

    source_name = "multi-source"

    def __init__(self, retrievers: list[Retriever]) -> None:
        self.retrievers = retrievers

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not query.strip():
            return []
        evidence: list[Evidence] = []
        for retriever in self.retrievers:
            try:
                source_top_k = getattr(retriever, "top_k", top_k)
                evidence.extend(retriever.retrieve(query, source_top_k))
            except Exception as exc:
                logger.warning(
                    "Retrieval source %s failed; continuing with other sources: %s",
                    getattr(retriever, "source_name", type(retriever).__name__),
                    type(exc).__name__,
                )
        deduplicated: dict[tuple[str, str], Evidence] = {}
        for item in evidence:
            key = (self._source_identity(item), self._normalize(item.text))
            existing = deduplicated.get(key)
            if existing is None or item.relevance_score > existing.relevance_score:
                deduplicated[key] = item
        ranked = sorted(deduplicated.values(), key=self._ranking_score, reverse=True)
        return ranked[:top_k] if top_k else ranked

    @classmethod
    def _ranking_score(cls, item: Evidence) -> float:
        """Balance semantic relevance with source authority transparently."""
        return (
            _SEMANTIC_WEIGHT * item.relevance_score
            + _SOURCE_QUALITY_WEIGHT * cls.source_quality_score(item.source_url)
        )

    @staticmethod
    def source_quality_score(source_url: str) -> float:
        """Return a deterministic authority score without discarding weak sources."""
        hostname = (urlparse(source_url).hostname or "").casefold().removeprefix("www.")
        if hostname.endswith(".gov") or hostname.endswith(".edu"):
            return 1.0
        if hostname in _HIGH_AUTHORITY_DOMAINS or any(
            hostname.endswith(f".{domain}") for domain in _HIGH_AUTHORITY_DOMAINS
        ):
            return 0.9
        if hostname in _LOW_AUTHORITY_DOMAINS or any(
            hostname.endswith(f".{domain}") for domain in _LOW_AUTHORITY_DOMAINS
        ):
            return 0.2
        return 0.5

    @staticmethod
    def _source_identity(item: Evidence) -> str:
        return item.source_url.casefold()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()