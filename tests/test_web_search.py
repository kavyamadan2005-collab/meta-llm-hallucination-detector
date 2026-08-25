"""Deterministic tests for the provider-independent web search adapter."""

from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.analysis import AnalysisComponents
from backend.main import app, get_analysis_components
from modules.claims import Claim
from modules.llm import DevelopmentLLMProvider, LLMResponse
from modules.retrieval import (
    DevelopmentWebSearchProvider,
    MultiSourceRetriever,
    WebSearchResult,
    WebSearchRetriever,
    TavilyWebSearchProvider,
)
from modules.verification import RuleBasedVerifier, VerificationService


def test_development_web_provider_returns_marked_result() -> None:
    results = DevelopmentWebSearchProvider().search("What is the capital of France?", 5)

    assert len(results) == 1
    assert isinstance(results[0], WebSearchResult)
    assert results[0].text.startswith("[Development web data]")


def test_web_search_retriever_converts_and_scores_evidence() -> None:
    provider = Mock()
    provider.search.return_value = [WebSearchResult(
        title="France", url="https://example.test/france", text="The capital of France is Paris.",
    )]

    evidence = WebSearchRetriever(provider).retrieve("What is the capital of France?", top_k=1)

    assert evidence[0].source_title == "France"
    assert evidence[0].source_url == "https://example.test/france"
    assert evidence[0].text == "The capital of France is Paris."
    assert evidence[0].relevance_score > 0
    provider.search.assert_called_once_with("What is the capital of France?", 1)


def test_web_search_retriever_handles_empty_results() -> None:
    provider = Mock()
    provider.search.return_value = []

    assert WebSearchRetriever(provider).retrieve("unknown") == []


def test_web_search_retriever_handles_provider_failure() -> None:
    provider = Mock()
    provider.search.side_effect = RuntimeError("unavailable")

    assert WebSearchRetriever(provider).retrieve("unknown") == []


def test_multisource_retriever_accepts_web_provider() -> None:
    web = WebSearchRetriever(DevelopmentWebSearchProvider())

    evidence = MultiSourceRetriever([web]).retrieve("photosynthesis")

    assert len(evidence) == 1
    assert evidence[0].source_url == "https://development.invalid/photosynthesis"


def test_analyze_uses_development_web_evidence() -> None:
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question="What is the capital of France?",
        response_text="The capital of France is Paris.",
        model="test-model",
        timestamp="2026-01-01T00:00:00Z",
    )
    extractor = Mock()
    extractor.extract.return_value = [Claim(
        claim_id="C1", claim_text="The capital of France is Paris.",
        original_sentence="The capital of France is Paris.", sentence_index=0,
    )]
    retriever = MultiSourceRetriever([WebSearchRetriever(DevelopmentWebSearchProvider())])
    components = AnalysisComponents(
        provider=provider,
        extractor=extractor,
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier(min_similarity=0.0)),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post(
            "/analyze", json={"question": "What is the capital of France?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    evidence = response.json()["verification_results"][0]["evidence"][0]
    assert evidence["source_url"] == "https://development.invalid/capital-of-france"
    assert evidence["text"].startswith("[Development web data]")


def test_web_search_configuration_defaults_to_development() -> None:
    from backend.core.config import Settings

    settings = Settings()

    assert settings.web_search_enabled is True
    assert settings.web_search_provider == "development"


def tavily_response(payload: object, status_code: int = 200) -> Mock:
    response = Mock()
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(status_code),
        )
    return response


def test_tavily_provider_converts_successful_results_without_exposing_key() -> None:
    client = Mock(spec=httpx.Client)
    client.post.return_value = tavily_response({"results": [{
        "title": "France", "url": "https://example.test/france", "content": "Paris is the capital of France.",
    }]})

    results = TavilyWebSearchProvider("secret-key", client=client).search("capital of France", 2)

    assert results == [WebSearchResult(
        title="France", url="https://example.test/france", text="Paris is the capital of France.",
        domain="example.test",
    )]
    request = client.post.call_args.kwargs
    assert request["json"]["query"] == "capital of France"
    assert request["json"]["max_results"] == 2


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
def test_tavily_provider_returns_empty_for_http_failures(status_code: int) -> None:
    client = Mock(spec=httpx.Client)
    client.post.return_value = tavily_response({}, status_code)

    assert TavilyWebSearchProvider("secret-key", client=client).search("query", 5) == []


def test_tavily_provider_returns_empty_for_empty_and_malformed_results() -> None:
    client = Mock(spec=httpx.Client)
    provider = TavilyWebSearchProvider("secret-key", client=client)
    client.post.return_value = tavily_response({"results": []})
    assert provider.search("query", 5) == []
    client.post.return_value = tavily_response({"results": [{"title": "Missing content", "url": "https://example.test"}]})
    assert provider.search("query", 5) == []


def test_tavily_provider_returns_empty_for_timeout() -> None:
    client = Mock(spec=httpx.Client)
    client.post.side_effect = httpx.TimeoutException("timed out")

    assert TavilyWebSearchProvider("secret-key", client=client).search("query", 5) == []


def test_tavily_provider_returns_empty_for_connection_failure() -> None:
    client = Mock(spec=httpx.Client)
    client.post.side_effect = httpx.ConnectError("offline")

    assert TavilyWebSearchProvider("secret-key", client=client).search("query", 5) == []


def test_tavily_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        TavilyWebSearchProvider(None)


def test_tavily_configuration_is_loaded_without_exposing_key(monkeypatch) -> None:
    from backend.core.config import Settings

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_URL", "https://tavily.test/search")
    monkeypatch.setenv("TAVILY_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("TAVILY_TOP_K", "3")
    monkeypatch.setenv("TAVILY_API_KEY", "test-only-key")
    settings = Settings.from_environment()

    assert settings.web_search_provider == "tavily"
    assert settings.tavily_api_url == "https://tavily.test/search"
    assert settings.tavily_timeout_seconds == 12
    assert settings.tavily_top_k == 3


def test_runtime_components_wire_tavily_into_multi_source_retriever(monkeypatch) -> None:
    from backend.api import analysis
    from backend.core.config import Settings

    tavily_client = Mock(spec=httpx.Client)
    tavily_client.post.return_value = tavily_response({"results": [{
        "title": "France", "url": "https://example.test/france",
        "content": "The capital of France is Paris.",
    }]})
    settings = Settings(
        web_search_provider="tavily",
        tavily_api_key="test-only-key",
        retrieval_enabled_sources=["web_search"],
    )
    embedder = Mock()
    embedder.encode.return_value = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr(analysis, "SentenceTransformerEmbedder", lambda: embedder)
    monkeypatch.setattr(analysis, "LocalFaissRetriever", lambda **kwargs: Mock())
    monkeypatch.setattr(analysis, "WikipediaRetriever", lambda **kwargs: Mock())
    monkeypatch.setattr(analysis, "OpenAlexRetriever", lambda **kwargs: Mock())
    monkeypatch.setattr(
        analysis,
        "TavilyWebSearchProvider",
        lambda **kwargs: TavilyWebSearchProvider(client=tavily_client, **kwargs),
    )
    monkeypatch.setattr(analysis, "ClaimExtractor", lambda: Mock())
    monkeypatch.setattr(analysis, "create_llm_provider", lambda settings: Mock())

    components = analysis.create_analysis_components(settings)
    evidence = components.retriever.retrieve("capital of France", top_k=1)

    assert len(components.retriever.retrievers) == 1
    assert evidence[0].source_url == "https://example.test/france"
    tavily_client.post.assert_called_once()


def test_tavily_result_flows_from_development_llm_through_verifier() -> None:
    tavily_client = Mock(spec=httpx.Client)
    tavily_client.post.return_value = tavily_response({"results": [{
        "title": "France", "url": "https://example.test/france", "content": "The capital of France is Paris.",
    }]})
    retriever = MultiSourceRetriever([
        WebSearchRetriever(TavilyWebSearchProvider("secret-key", client=tavily_client))
    ])
    extractor = Mock()
    extractor.extract.return_value = [Claim(
        claim_id="C1", claim_text="The capital of France is Paris.",
        original_sentence="The capital of France is Paris.", sentence_index=0,
    )]
    components = AnalysisComponents(
        provider=DevelopmentLLMProvider(),
        extractor=extractor,
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier(min_similarity=0.0)),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post(
            "/analyze", json={"question": "What is the capital of France?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    evidence = response.json()["verification_results"][0]["evidence"][0]
    assert evidence["source_url"] == "https://example.test/france"
    assert "secret-key" not in response.text
