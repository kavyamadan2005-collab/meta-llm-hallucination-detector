"""External retrieval from Wikipedia's public search API."""

from __future__ import annotations

import logging
import html
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from modules.retrieval.models import Evidence

logger = logging.getLogger(__name__)


class ExternalRetrieverError(RuntimeError):
    """Raised when an external knowledge source cannot be queried."""


class WikipediaRetriever:
    """Retrieve Wikipedia search results through its public JSON API."""

    endpoint = "https://en.wikipedia.org/w/api.php"
    source_name = "wikipedia"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 10.0, top_k: int = 5,
                 embedder: Any | None = None, *, api_url: str = endpoint,
                 user_agent: str = "Meta-LLM-Hallucination-Detection/0.1 (local development)") -> None:
        self.endpoint = api_url.rstrip("/")
        self.user_agent = user_agent
        self.top_k = top_k
        self._embedder = embedder
        self._client = client or httpx.Client(timeout=timeout)
        self._cache: dict[tuple[str, int], list[Evidence]] = {}

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        """Return available search extracts without inventing missing content."""
        if not query.strip():
            return []
        limit = max(1, top_k or self.top_k)
        cache_key = (query.strip().casefold(), limit)
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        try:
            response = self._client.get(self.endpoint, headers={"User-Agent": self.user_agent}, params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": limit, "format": "json", "utf8": 1,
            })
            response.raise_for_status()
            search_results = response.json().get("query", {}).get("search", [])
            if not isinstance(search_results, list):
                return []
            timestamp = datetime.now(timezone.utc)
            evidence = [
                self._evidence(query, result, timestamp, passage_index, passage)
                for result in search_results if self._valid(result)
                for passage_index, passage in enumerate(self._passages(result["snippet"]))
            ]
            self._cache[cache_key] = evidence
            return list(evidence)
        except httpx.TimeoutException:
            logger.warning("Wikipedia request timed out; continuing without external evidence")
            return []
        except httpx.NetworkError:
            logger.warning("Wikipedia connection failed; continuing without external evidence")
            return []
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 403 or status_code == 429 or 500 <= status_code <= 599:
                logger.warning(
                    "Wikipedia returned HTTP %s; continuing without external evidence",
                    status_code,
                )
                return []
            raise ExternalRetrieverError("Wikipedia returned an HTTP error") from exc
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            raise ExternalRetrieverError("Wikipedia response could not be processed") from exc

    @staticmethod
    def _valid(result: Any) -> bool:
        if not isinstance(result, dict) or not isinstance(result.get("title"), str) or not isinstance(result.get("snippet"), str):
            return False
        return bool(html.unescape(re.sub(r"<[^>]+>", "", result["snippet"])).strip())

    def _evidence(self, query: str, result: dict[str, Any], timestamp: datetime,
                  passage_index: int, passage: str) -> Evidence:
        title = result["title"]
        text = passage
        return Evidence(
            evidence_id=f"wikipedia:{title.casefold()}:{passage_index}",
            claim_or_query=query,
            text=text,
            source_title=title,
            source_url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            relevance_score=self._score(query, text),
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
    def _passages(snippet: str) -> list[str]:
        cleaned = html.unescape(re.sub(r"<[^>]+>", "", snippet)).strip()
        return [passage.strip() for passage in re.split(r"(?<=[.!?])\s+", cleaned) if passage.strip()]