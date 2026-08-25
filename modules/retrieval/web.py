"""Provider-independent web search retrieval adapters."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

import httpx

from modules.retrieval.models import Evidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    """Provider-neutral web result with enough data to build evidence."""

    title: str
    url: str
    text: str
    domain: str | None = None


class WebSearchProvider(Protocol):
    """Minimal interface for pluggable web search implementations."""

    def search(self, query: str, top_k: int) -> list[WebSearchResult]:
        """Return provider results without inventing missing content."""


class TavilyWebSearchProvider:
    """Search Tavily's official API without coupling callers to its SDK."""

    source_name = "tavily"

    def __init__(self, api_key: str | None, api_url: str = "https://api.tavily.com/search",
                 timeout: float = 10.0, top_k: int = 5,
                 client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required when WEB_SEARCH_PROVIDER=tavily")
        self._api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.top_k = top_k
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, top_k: int) -> list[WebSearchResult]:
        """Return only meaningful Tavily result fields, never generated answers."""
        if not query.strip() or top_k <= 0:
            return []
        try:
            response = self._client.post(
                self.api_url,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": min(top_k, self.top_k),
                    "search_depth": "basic",
                    "include_answer": False,
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if not isinstance(results, list):
                return []
            return [
                WebSearchResult(
                    title=item["title"].strip(),
                    url=item["url"].strip(),
                    text=item["content"].strip(),
                    domain=urlparse(item["url"]).netloc or None,
                )
                for item in results
                if self._valid(item)
            ]
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.warning("Tavily web search unavailable: %s", type(exc).__name__)
            return []

    @staticmethod
    def _valid(result: object) -> bool:
        return (
            isinstance(result, dict)
            and isinstance(result.get("title"), str)
            and isinstance(result.get("url"), str)
            and isinstance(result.get("content"), str)
            and bool(result["title"].strip())
            and bool(result["url"].strip())
            and bool(result["content"].strip())
        )

    def close(self) -> None:
        self._client.close()


class DevelopmentWebSearchProvider:
    """Deterministic, clearly labeled web results for local testing only."""

    source_name = "development-web"
    _RESULTS = {
        "capital of france": WebSearchResult(
            title="[Development web data] Capital of France",
            url="https://development.invalid/capital-of-france",
            text="[Development web data] The capital of France is Paris.",
            domain="development.invalid",
        ),
        "chandrayaan-3": WebSearchResult(
            title="[Development web data] Chandrayaan-3",
            url="https://development.invalid/chandrayaan-3",
            text="[Development web data] Chandrayaan-3 landed on the Moon in 2023.",
            domain="development.invalid",
        ),
        "photosynthesis": WebSearchResult(
            title="[Development web data] Photosynthesis",
            url="https://development.invalid/photosynthesis",
            text="[Development web data] Photosynthesis uses light energy to help plants make food.",
            domain="development.invalid",
        ),
    }

    def search(self, query: str, top_k: int) -> list[WebSearchResult]:
        if not query.strip() or top_k <= 0:
            return []
        normalized = query.casefold()
        return [result for key, result in self._RESULTS.items() if key in normalized][:top_k]


class WebSearchRetriever:
    """Adapt a web search provider to the common evidence retriever contract."""

    source_name = "web-search"

    def __init__(self, provider: WebSearchProvider, top_k: int = 5, embedder: object | None = None) -> None:
        self.provider = provider
        self.top_k = top_k
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not query.strip():
            return []
        try:
            if top_k == 0:
                return []
            results = self.provider.search(query, max(1, top_k if top_k is not None else self.top_k))
            timestamp = datetime.now(timezone.utc)
            return [
                self._evidence(query, result, timestamp)
                for result in results
                if self._valid(result)
            ]
        except Exception as exc:
            logger.warning("Web search provider unavailable: %s", type(exc).__name__)
            return []

    def _evidence(self, query: str, result: WebSearchResult, timestamp: datetime) -> Evidence:
        return Evidence(
            evidence_id=f"web:{result.url}",
            claim_or_query=query,
            text=result.text.strip(),
            source_title=result.title.strip(),
            source_url=result.url.strip(),
            relevance_score=self._score(query, result.text),
            retrieval_timestamp=timestamp,
        )

    def _score(self, query: str, text: str) -> float:
        if self._embedder is not None:
            vectors = self._embedder.encode([query, text])
            query_norm = float((vectors[0] @ vectors[0]) ** 0.5)
            text_norm = float((vectors[1] @ vectors[1]) ** 0.5)
            if query_norm and text_norm:
                return max(0.0, min(1.0, float(vectors[0] @ vectors[1]) / (query_norm * text_norm)))
            return 0.0
        query_words = set(re.findall(r"[a-z0-9]+", query.casefold()))
        text_words = set(re.findall(r"[a-z0-9]+", text.casefold()))
        return len(query_words & text_words) / max(1, len(query_words))

    @staticmethod
    def _valid(result: object) -> bool:
        return (
            isinstance(result, WebSearchResult)
            and bool(result.title.strip())
            and bool(result.url.strip())
            and bool(result.text.strip())
        )