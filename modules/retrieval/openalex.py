"""Evidence retrieval through the official OpenAlex works API."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from modules.retrieval.models import Evidence

logger = logging.getLogger(__name__)


class OpenAlexRetriever:
    """Retrieve scholarly works without requiring an OpenAlex API key."""

    source_name = "openalex"

    def __init__(self, api_url: str = "https://api.openalex.org/works", api_key: str | None = None,
                 timeout: float = 10.0, top_k: int = 5, client: httpx.Client | None = None,
                 embedder: Any | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.top_k = top_k
        self._client = client or httpx.Client(timeout=timeout)
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not query.strip():
            return []
        params: dict[str, Any] = {"search": query, "per-page": max(1, top_k or self.top_k)}
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            response = self._client.get(self.api_url, params=params)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if not isinstance(results, list):
                raise ValueError("OpenAlex results must be a list")
            timestamp = datetime.now(timezone.utc)
            evidence: list[Evidence] = []
            for item in results:
                if self._valid(item):
                    try:
                        evidence.append(self._evidence(query, item, timestamp))
                    except (TypeError, ValueError, KeyError):
                        logger.warning("Skipping malformed OpenAlex result")
            return evidence
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.warning("OpenAlex retrieval unavailable: %s", type(exc).__name__)
            return []

    def _evidence(self, query: str, result: dict[str, Any], timestamp: datetime) -> Evidence:
        title = result["title"].strip()
        text = self._abstract(result) or title
        return Evidence(
            evidence_id=f"openalex:{result.get('id', title).rstrip('/').split('/')[-1]}",
            claim_or_query=query,
            text=text,
            source_title=title,
            source_url=result["id"],
            relevance_score=self._score(query, text, result.get("relevance_score")),
            retrieval_timestamp=timestamp,
        )

    def _score(self, query: str, text: str, api_score: Any) -> float:
        if self._embedder is not None:
            vectors = self._embedder.encode([query, text])
            query_norm = float((vectors[0] @ vectors[0]) ** 0.5)
            text_norm = float((vectors[1] @ vectors[1]) ** 0.5)
            if query_norm and text_norm:
                return max(0.0, min(1.0, float(vectors[0] @ vectors[1]) / (query_norm * text_norm)))
            return 0.0
        try:
            return max(0.0, min(1.0, float(api_score)))
        except (TypeError, ValueError):
            query_words = set(re.findall(r"[a-z0-9]+", query.casefold()))
            text_words = set(re.findall(r"[a-z0-9]+", text.casefold()))
            return len(query_words & text_words) / max(1, len(query_words))

    @staticmethod
    def _abstract(result: dict[str, Any]) -> str:
        inverted = result.get("abstract_inverted_index")
        if not isinstance(inverted, dict):
            return ""
        words: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            if isinstance(word, str) and isinstance(positions, list):
                words.extend((position, word) for position in positions if isinstance(position, int))
        return " ".join(word for _, word in sorted(words))

    @staticmethod
    def _valid(result: Any) -> bool:
        return (
            isinstance(result, dict)
            and isinstance(result.get("title"), str)
            and bool(result["title"].strip())
            and isinstance(result.get("id"), str)
            and bool(result["id"].strip())
        )

    def close(self) -> None:
        self._client.close()