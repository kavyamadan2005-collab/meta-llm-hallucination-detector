"""Focused tests for the end-to-end analysis endpoint."""

from datetime import datetime, timezone
from unittest.mock import Mock

from fastapi.testclient import TestClient

from backend.api.analysis import AnalysisComponents
from backend.main import app, get_analysis_components
from modules.claims import Claim
from modules.llm import LLMResponse
from modules.retrieval.models import Evidence
from modules.verification import RuleBasedVerifier, VerificationService


def make_components() -> AnalysisComponents:
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question="What happened?", response_text="ISRO launched Chandrayaan-3 in 2023. The Moon has craters.",
        model="test-model", timestamp=datetime.now(timezone.utc),
    )
    extractor = Mock()
    extractor.extract.return_value = [
        Claim(claim_id="C1", claim_text="ISRO launched Chandrayaan-3 in 2023.", original_sentence="ISRO launched Chandrayaan-3 in 2023.", sentence_index=0),
        Claim(claim_id="C2", claim_text="The Moon has craters.", original_sentence="The Moon has craters.", sentence_index=1),
    ]
    retriever = Mock()
    retriever.retrieve.return_value = [Evidence(
        evidence_id="E1", claim_or_query="query", text="Evidence text.", source_title="Test source",
        source_url="https://example.test/source", relevance_score=0.9,
    )]
    return AnalysisComponents(
        provider=provider,
        extractor=extractor,
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )


def test_analyze_orchestrates_pipeline_and_serializes_graph_and_verification() -> None:
    components = make_components()
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": "What happened?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What happened?"
    assert payload["llm_response"]["model"] == "test-model"
    assert len(payload["claims"]) == 2
    assert set(payload["graph"]) == {"nodes", "edges", "claims"}
    assert len(payload["verification_results"]) == 2
    assert payload["verification_results"][0]["evidence"][0]["source_url"] == "https://example.test/source"
    components.provider.generate.assert_called_once_with("What happened?")


def test_analyze_rejects_empty_question() -> None:
    response = TestClient(app).post("/analyze", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "question must not be empty"
