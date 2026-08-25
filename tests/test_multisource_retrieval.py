"""Deterministic tests for heterogeneous retrieval sources."""

from unittest.mock import Mock

import httpx

from backend.core.config import Settings
from modules.retrieval import MultiSourceRetriever, OpenAlexRetriever, WikipediaRetriever


def test_multisource_retriever_combines_ranks_and_deduplicates() -> None:
    first = Mock()
    second = Mock()
    first.source_name = "local"
    second.source_name = "wikipedia"
    from modules.retrieval.models import Evidence
    first.retrieve.return_value = [Evidence(evidence_id="one", claim_or_query="q", text="Same fact", source_title="A", source_url="local://a", relevance_score=0.4)]
    second.retrieve.return_value = [
        Evidence(evidence_id="two", claim_or_query="q", text=" same   fact ", source_title="B", source_url="https://b", relevance_score=0.9),
        Evidence(evidence_id="duplicate", claim_or_query="q", text="SAME FACT", source_title="B", source_url="https://b", relevance_score=0.3),
        Evidence(evidence_id="three", claim_or_query="q", text="Other fact", source_title="B", source_url="https://b", relevance_score=0.7),
    ]

    result = MultiSourceRetriever([first, second]).retrieve("q", top_k=2)

    assert [item.relevance_score for item in result] == [0.9, 0.7]
    assert [item.source_title for item in result] == ["B", "B"]


def test_multisource_retriever_keeps_successful_sources_when_one_fails() -> None:
    failed = Mock()
    successful = Mock()
    failed.retrieve.side_effect = RuntimeError("offline")
    from modules.retrieval.models import Evidence
    evidence = Evidence(evidence_id="local:1", claim_or_query="q", text="Local fact", source_title="Local", source_url="local://a", relevance_score=0.8)
    successful.retrieve.return_value = [evidence]

    assert MultiSourceRetriever([failed, successful]).retrieve("q") == [evidence]


def test_multisource_ranking_prefers_authoritative_source_with_comparable_relevance() -> None:
    authoritative = Mock()
    weak = Mock()
    from modules.retrieval.models import Evidence
    authoritative.retrieve.return_value = [Evidence(
        evidence_id="gov", claim_or_query="q", text="Government evidence",
        source_title="Government", source_url="https://data.gov/fact", relevance_score=0.8,
    )]
    weak.retrieve.return_value = [Evidence(
        evidence_id="quora", claim_or_query="q", text="Forum evidence",
        source_title="Forum", source_url="https://quora.com/question", relevance_score=0.81,
    )]

    result = MultiSourceRetriever([weak, authoritative]).retrieve("q")

    assert result[0].source_url == "https://data.gov/fact"


def test_multisource_ranking_preserves_highly_relevant_weak_domain() -> None:
    authoritative = Mock()
    weak = Mock()
    from modules.retrieval.models import Evidence
    authoritative.retrieve.return_value = [Evidence(
        evidence_id="edu", claim_or_query="q", text="Broad evidence",
        source_title="University", source_url="https://university.edu/fact", relevance_score=0.7,
    )]
    weak.retrieve.return_value = [Evidence(
        evidence_id="forum", claim_or_query="q", text="Precise evidence",
        source_title="Forum", source_url="https://quora.com/precise", relevance_score=0.99,
    )]

    result = MultiSourceRetriever([authoritative, weak]).retrieve("q")

    assert result[0].source_url == "https://quora.com/precise"


def test_multisource_ranking_preserves_evidence_source_urls() -> None:
    source = Mock()
    from modules.retrieval.models import Evidence
    evidence = Evidence(
        evidence_id="source", claim_or_query="q", text="A fact",
        source_title="Source", source_url="https://example.edu/fact", relevance_score=0.8,
    )
    source.retrieve.return_value = [evidence]

    result = MultiSourceRetriever([source]).retrieve("q")

    assert result[0].source_url == evidence.source_url


def test_multisource_retriever_returns_empty_when_all_sources_fail() -> None:
    source = Mock()
    source.retrieve.side_effect = RuntimeError("offline")

    assert MultiSourceRetriever([source, source]).retrieve("q") == []


def test_openalex_retriever_maps_abstract_and_provenance() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"results": [{
        "id": "https://openalex.org/W1", "title": "Moon study", "publication_year": 2023,
        "authorships": [{"author": {"display_name": "A Researcher"}}],
        "abstract_inverted_index": {"The": [0], "Moon": [1], "has": [2], "craters": [3]},
    }]}
    client.get.return_value = response

    evidence = OpenAlexRetriever(client=client).retrieve("Moon", top_k=1)

    assert len(evidence) == 1
    assert evidence[0].source_title == "Moon study"
    assert evidence[0].source_url == "https://openalex.org/W1"
    assert evidence[0].text == "The Moon has craters"
    assert evidence[0].relevance_score > 0


def test_openalex_retriever_handles_empty_results() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"results": []}
    client.get.return_value = response

    assert OpenAlexRetriever(client=client).retrieve("unknown") == []


def test_openalex_retriever_handles_timeout() -> None:
    client = Mock(spec=httpx.Client)
    client.get.side_effect = httpx.TimeoutException("timed out")

    assert OpenAlexRetriever(client=client).retrieve("unknown") == []


def test_retrieval_source_configuration_is_loaded() -> None:
    settings = Settings(
        retrieval_enabled_sources=["local", "openalex"],
        wikipedia_api_url="https://wiki.test/api",
        openalex_api_url="https://openalex.test/works",
    )

    assert settings.retrieval_enabled_sources == ["local", "openalex"]
    assert settings.wikipedia_api_url == "https://wiki.test/api"
    assert settings.openalex_api_key is None