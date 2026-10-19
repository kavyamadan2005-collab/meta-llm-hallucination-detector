"""Dependency construction for the end-to-end analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from backend.core.config import Settings, get_settings
from modules.claims import ClaimExtractor
from modules.graph import ClaimGraph
from modules.llm import LLMProvider, create_llm_provider
from modules.retrieval import (
    CuratedFactRetriever,
    DevelopmentWebSearchProvider,
    LocalFaissRetriever,
    MultiSourceRetriever,
    OpenAlexRetriever,
    TavilyWebSearchProvider,
    WebSearchRetriever,
    WikipediaRetriever,
)
from modules.retrieval.embeddings import SentenceTransformerEmbedder
from modules.retrieval.interfaces import Retriever
from modules.verification import RuleBasedVerifier, VerificationService


@dataclass
class AnalysisComponents:
    """Concrete services used by the analysis endpoint."""

    provider: LLMProvider
    extractor: ClaimExtractor
    retriever: Retriever
    verifier: VerificationService
    retrieval_top_k: int


def create_analysis_components(settings: Settings | None = None) -> AnalysisComponents:
    """Build the pipeline from application settings for production use."""
    configured = settings or get_settings()
    embedder = SentenceTransformerEmbedder()
    local_retriever = LocalFaissRetriever(
        index_directory=configured.retrieval_index_path,
        embedder=embedder,
        min_similarity=configured.retrieval_min_similarity,
    )
    sources: dict[str, Retriever] = {
        "curated": CuratedFactRetriever(),
        "local": local_retriever,
        "wikipedia": WikipediaRetriever(
            api_url=configured.wikipedia_api_url,
            user_agent=configured.wikipedia_user_agent,
            timeout=configured.wikipedia_timeout_seconds,
            top_k=configured.wikipedia_top_k,
            embedder=embedder,
        ),
        "openalex": OpenAlexRetriever(
            api_url=configured.openalex_api_url,
            api_key=configured.openalex_api_key,
            timeout=configured.openalex_timeout_seconds,
            top_k=configured.openalex_top_k,
            embedder=embedder,
        ),
    }
    if configured.web_search_enabled:
        if configured.web_search_provider == "development":
            web_provider = DevelopmentWebSearchProvider()
        elif configured.web_search_provider == "tavily":
            web_provider = TavilyWebSearchProvider(
                api_key=configured.tavily_api_key,
                api_url=configured.tavily_api_url,
                timeout=configured.tavily_timeout_seconds,
                top_k=configured.tavily_top_k,
            )
        else:
            raise ValueError(f"Unsupported web search provider: {configured.web_search_provider}")
        sources["web_search"] = WebSearchRetriever(
            web_provider, top_k=configured.retrieval_top_k, embedder=embedder
        )
    retriever = MultiSourceRetriever([
        sources[name] for name in configured.retrieval_enabled_sources if name in sources
    ])
    verifier = VerificationService(
        retriever=retriever,
        verifier=RuleBasedVerifier(min_similarity=configured.retrieval_min_similarity),
    )
    return AnalysisComponents(
        provider=create_llm_provider(configured),
        extractor=ClaimExtractor(),
        retriever=retriever,
        verifier=verifier,
        retrieval_top_k=configured.retrieval_top_k,
    )