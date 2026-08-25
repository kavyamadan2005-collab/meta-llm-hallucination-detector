"""Focused tests for local, external, and hybrid evidence retrieval."""

from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from modules.retrieval import HybridRetriever, LocalFaissRetriever, WikipediaRetriever
from modules.retrieval.chunking import chunk_text, load_documents
from modules.retrieval.external import ExternalRetrieverError
from modules.retrieval.models import Evidence


class TestEmbedder:
    """Small deterministic embedder used instead of downloading a model."""

    dimension = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([
            [1.0, 0.0] if "moon" in text.lower() or "query" in text.lower() else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)


def test_document_loading_and_deterministic_chunking(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "b.md").write_text("second document", encoding="utf-8")
    (documents / "a.txt").write_text("0123456789", encoding="utf-8")
    (documents / "ignored.json").write_text("ignore", encoding="utf-8")

    loaded = load_documents(documents)
    chunks = chunk_text(loaded[0][1], loaded[0][0], chunk_size=4, overlap=1)

    assert [name for name, _ in loaded] == ["a.txt", "b.md"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.text for chunk in chunks] == ["0123", "3456", "6789", "9"]
    assert all(chunk.document_name == "a.txt" for chunk in chunks)


def test_faiss_build_search_persistence_and_metadata(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "facts.txt").write_text("The Moon has craters.", encoding="utf-8")
    index_path = tmp_path / "index"
    embedder = TestEmbedder()
    retriever = LocalFaissRetriever(index_path, embedder=embedder, min_similarity=0.5)

    assert retriever.build(documents, chunk_size=100, overlap=0) == 1
    evidence = retriever.retrieve("Moon query", top_k=1)
    reloaded = LocalFaissRetriever(index_path, embedder=embedder, min_similarity=0.5)

    assert len(evidence) == 1
    assert evidence[0].text == "The Moon has craters."
    assert evidence[0].source_title == "facts.txt"
    assert evidence[0].source_url == "local://facts.txt"
    assert reloaded.retrieve("Moon query", top_k=1)[0].evidence_id == evidence[0].evidence_id


def test_faiss_empty_index_and_threshold_return_no_evidence(tmp_path: Path) -> None:
    documents = tmp_path / "empty"
    documents.mkdir()
    retriever = LocalFaissRetriever(tmp_path / "index", embedder=TestEmbedder(), min_similarity=1.1)

    assert retriever.build(documents) == 0
    assert retriever.retrieve("query") == []


def test_evidence_serialization() -> None:
    evidence = Evidence(
        evidence_id="local:1", claim_or_query="claim", text="fact",
        source_title="Notes", source_url="local://Notes", relevance_score=0.8,
    )

    evidence_data = evidence.model_dump() if hasattr(evidence, "model_dump") else evidence.dict()
    assert evidence_data["evidence_id"] == "local:1"
    assert evidence_data["text"] == "fact"


def test_wikipedia_retriever_uses_mocked_http_response() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"query": {"search": [{"title": "Moon", "snippet": "A natural satellite."}]}}
    client.get.return_value = response

    evidence = WikipediaRetriever(client=client).retrieve("Moon", top_k=1)

    assert evidence[0].source_title == "Moon"
    assert evidence[0].source_url.endswith("/Moon")
    assert evidence[0].text == "A natural satellite."
    client.get.assert_called_once()


@pytest.mark.parametrize("status_code", [403, 429, 500])
def test_wikipedia_retriever_ignores_expected_http_failures(status_code: int) -> None:
    client = Mock(spec=httpx.Client)
    client.get.return_value = httpx.Response(status_code, request=httpx.Request("GET", "https://example.test"))

    assert WikipediaRetriever(client=client).retrieve("Moon") == []


def test_wikipedia_retriever_ignores_timeout() -> None:
    client = Mock(spec=httpx.Client)
    client.get.side_effect = httpx.TimeoutException("timed out")

    assert WikipediaRetriever(client=client).retrieve("Moon") == []


def test_wikipedia_retriever_ignores_connection_failure() -> None:
    client = Mock(spec=httpx.Client)
    client.get.side_effect = httpx.ConnectError("offline")

    assert WikipediaRetriever(client=client).retrieve("Moon") == []


def test_hybrid_retriever_keeps_local_evidence_when_wikipedia_is_unavailable() -> None:
    local = Mock()
    local_evidence = Evidence(
        evidence_id="local:1", claim_or_query="Moon", text="Local fact",
        source_title="Notes", source_url="local://Notes", relevance_score=0.8,
    )
    local.retrieve.return_value = [local_evidence]
    external_client = Mock(spec=httpx.Client)
    external_client.get.return_value = httpx.Response(
        403, request=httpx.Request("GET", "https://example.test")
    )

    evidence = HybridRetriever([
        local, WikipediaRetriever(client=external_client)
    ]).retrieve("Moon", top_k=5)

    assert evidence == [local_evidence]


def test_hybrid_retriever_keeps_local_evidence_when_external_response_is_invalid() -> None:
    local = Mock()
    local_evidence = Evidence(
        evidence_id="local:1", claim_or_query="Moon", text="Local fact",
        source_title="Notes", source_url="local://Notes", relevance_score=0.8,
    )
    local.retrieve.return_value = [local_evidence]
    external_client = Mock(spec=httpx.Client)
    external_client.get.return_value = httpx.Response(
        400, request=httpx.Request("GET", "https://example.test")
    )

    evidence = HybridRetriever([
        local, WikipediaRetriever(client=external_client)
    ]).retrieve("Moon", top_k=5)

    assert evidence == [local_evidence]


def test_hybrid_retriever_deduplicates_same_source_and_text() -> None:
    local = Mock()
    external = Mock()
    duplicate = Evidence(
        evidence_id="local:1", claim_or_query="claim", text="same fact",
        source_title="Notes", source_url="https://example.test/fact", relevance_score=0.7,
    )
    unique = Evidence(
        evidence_id="external:1", claim_or_query="claim", text="same fact",
        source_title="External", source_url="https://example.test/fact", relevance_score=0.9,
    )
    local.retrieve.return_value = [duplicate]
    external.retrieve.return_value = [unique]

    evidence = HybridRetriever([local, external]).retrieve("claim", top_k=5)

    assert len(evidence) == 1
    assert evidence[0].source_title == "Notes"