"""Focused tests for the end-to-end analysis endpoint."""

from datetime import datetime, timezone
from unittest.mock import Mock

import spacy
from fastapi.testclient import TestClient

from backend.api.analysis import AnalysisComponents
from backend.main import app, get_analysis_components
from modules.claims import Claim, ClaimExtractor
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
    components.provider.generate.assert_called_once_with(
        "What happened?",
        evidence=components.retriever.retrieve.return_value,
    )


def test_analyze_remains_compatible_with_legacy_generation_providers() -> None:
    components = make_components()

    class LegacyProvider:
        def generate(self, question: str) -> LLMResponse:
            return LLMResponse(
                question=question,
                response_text="A legacy provider response.",
                model="legacy-test-model",
                timestamp=datetime.now(timezone.utc),
            )

    components.provider = LegacyProvider()
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post(
            "/analyze",
            json={"question": "What happened?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["llm_response"]["response_text"] == (
        "A legacy provider response."
    )


def test_analyze_supplies_question_evidence_before_generating_a_list_answer() -> None:
    question = "Name five regions that contain active volcanoes."
    evidence = Evidence(
        evidence_id="volcano-regions",
        claim_or_query=question,
        text="The listed regions with active volcanoes are A, B, and C.",
        source_title="Geographic reference",
        source_url="https://example.test/volcano-regions",
        relevance_score=0.9,
    )
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question=question,
        response_text="The retrieved evidence lists regions A, B, and C; it does not establish five.",
        model="test-model",
        timestamp=datetime.now(timezone.utc),
    )

    class ListEvidenceRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [evidence]

    retriever = ListEvidenceRetriever()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    components = AnalysisComponents(
        provider=provider,
        extractor=ClaimExtractor(nlp=nlp),
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    provider.generate.assert_called_once_with(question, evidence=[evidence])
    assert response.json()["llm_response"]["response_text"] == (
        "The retrieved evidence lists regions A, B, and C; it does not establish five."
    )


def test_analyze_corrects_a_list_question_using_retrieved_evidence() -> None:
    question = "Name the 5 states that start with letter A in India."
    evidence = Evidence(
        evidence_id="indian-states-a",
        claim_or_query=question,
        text=(
            "Only three Indian states start with A: Andhra Pradesh, "
            "Arunachal Pradesh, and Assam."
        ),
        source_title="State reference",
        source_url="https://example.test/indian-states",
        relevance_score=0.95,
    )
    response_text = (
        "Only three Indian states start with A: Andhra Pradesh, "
        "Arunachal Pradesh, and Assam."
    )

    class EvidenceGroundedProvider:
        def generate(self, prompt: str, evidence: list[Evidence]) -> LLMResponse:
            assert prompt == question
            assert evidence == [evidence_item]
            return LLMResponse(
                question=prompt,
                response_text=response_text,
                model="test-model",
                timestamp=datetime.now(timezone.utc),
            )

    evidence_item = evidence

    class StateListRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [evidence_item]

    retriever = StateListRetriever()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    components = AnalysisComponents(
        provider=EvidenceGroundedProvider(),
        extractor=ClaimExtractor(nlp=nlp),
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_response"]["response_text"] == response_text
    assert payload["verification_results"][0]["verdict"] == "SUPPORTED"
    assert payload["overall_result"]["status"] == "VERIFIED"


def test_analysis_supports_capital_correction_from_direct_low_score_evidence() -> None:
    question = "cspital of india is kanataka"
    response_text = "The capital of India is New Delhi, not Karnataka."
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question=question,
        response_text=response_text,
        model="test-model",
        timestamp=datetime.now(timezone.utc),
    )
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    entity_ruler = nlp.add_pipe("entity_ruler")
    entity_ruler.add_patterns([
        {"label": "GPE", "pattern": "India"},
        {"label": "GPE", "pattern": "New Delhi"},
        {"label": "GPE", "pattern": "Karnataka"},
    ])
    extractor = ClaimExtractor(nlp=nlp)
    source = Evidence(
        evidence_id="india-capital",
        claim_or_query=question,
        text="New Delhi is the capital of India.",
        source_title="Capital reference",
        source_url="https://example.test/india-capital",
        relevance_score=0.46,
    )

    class DirectEvidenceRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [source]

    retriever = DirectEvidenceRetriever()
    components = AnalysisComponents(
        provider=provider,
        extractor=extractor,
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    result = response.json()["verification_results"][0]
    assert result["verdict"] == "SUPPORTED"
    assert result["supporting_evidence"][0]["evidence_id"] == "india-capital"


def test_analysis_splits_and_verifies_each_contrastive_fact() -> None:
    question = "cspital of india is kanataka"
    response_text = (
        "The capital of India is New Delhi, not Karnataka. "
        "Karnataka is a state in southern India, but "
        "New Delhi has been the capital since 1911."
    )
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question=question,
        response_text=response_text,
        model="test-model",
        timestamp=datetime.now(timezone.utc),
    )
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    entity_ruler = nlp.add_pipe("entity_ruler")
    entity_ruler.add_patterns([
        {"label": "GPE", "pattern": "India"},
        {"label": "GPE", "pattern": "New Delhi"},
        {"label": "GPE", "pattern": "Karnataka"},
    ])
    extractor = ClaimExtractor(nlp=nlp)
    evidence_items = [
        Evidence(
            evidence_id="india-capital",
            claim_or_query=question,
            text="New Delhi is the capital of India.",
            source_title="Capital reference",
            source_url="https://example.test/india-capital",
            relevance_score=0.46,
        ),
        Evidence(
            evidence_id="karnataka-state",
            claim_or_query=question,
            text="Karnataka is a state in southern India.",
            source_title="State reference",
            source_url="https://example.test/karnataka",
            relevance_score=0.9,
        ),
        Evidence(
            evidence_id="capital-since-1911",
            claim_or_query=question,
            text="The current capital city is New Delhi, which replaced Calcutta in 1911.",
            source_title="Capital history",
            source_url="https://example.test/capital-history",
            relevance_score=0.46,
        ),
    ]

    class CapitalRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return evidence_items

    retriever = CapitalRetriever()
    components = AnalysisComponents(
        provider=provider,
        extractor=extractor,
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert [claim["claim_text"] for claim in payload["claims"]] == [
        "The capital of India is New Delhi, not Karnataka.",
        "Karnataka is a state in southern India.",
        "New Delhi has been the capital since 1911.",
    ]
    assert [result["verdict"] for result in payload["verification_results"]] == [
        "SUPPORTED",
        "SUPPORTED",
        "SUPPORTED",
    ]


def test_analyze_rejects_empty_question() -> None:
    response = TestClient(app).post("/analyze", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "question must not be empty"


def test_negative_question_preserves_and_verifies_positive_landing_answer() -> None:
    question = "Did Chandrayaan-3 fail to land on the Moon?"
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question=question,
        response_text=(
            "No. Chandrayaan-3 successfully landed on the Moon on August 23, 2023."
        ),
        model="test-model",
        timestamp=datetime.now(timezone.utc),
    )
    retriever = Mock()
    retriever.retrieve.return_value = [
        Evidence(
            evidence_id="E1",
            claim_or_query="query",
            text="Chandrayaan-3 successfully landed on the Moon on August 23, 2023.",
            source_title="Mission report",
            source_url="https://example.test/mission",
            relevance_score=0.9,
        )
    ]
    components = AnalysisComponents(
        provider=provider,
        extractor=ClaimExtractor(),
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_response"]["response_text"].startswith("No.")
    assert payload["overall_result"]["status"] == "VERIFIED"
    result = payload["verification_results"][0]
    assert result["verdict"] == "SUPPORTED"
    assert result["event_fact"]["verdict"] == "SUPPORTED"
    assert result["attributes"][0]["name"] == "date"
    assert result["attributes"][0]["verdict"] == "SUPPORTED"


def test_telephone_analysis_returns_verdicts_consistent_with_retrieved_evidence() -> None:
    question = "Who invented the telephone and when?"
    response_text = (
        "Alexander Graham Bell is credited with inventing the first practical telephone. "
        "He patented the device on March 7, 1876, at his laboratory in Boston, Massachusetts. "
        "It is debated whether Bell was the sole inventor, with claims also associated "
        "with Elisha Gray and Antonio Meucci."
    )
    provider = Mock()
    provider.generate.return_value = LLMResponse(
        question=question,
        response_text=response_text,
        model="test-model",
        timestamp=datetime.now(timezone.utc),
    )
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    extractor = ClaimExtractor(nlp=nlp)
    items = {
        "bell": Evidence(
            evidence_id="telephone:inventor",
            claim_or_query=question,
            text=(
                "Alexander Graham Bell is credited with inventing the first "
                "practical telephone."
            ),
            source_title="Telephone history",
            source_url="https://example.test/telephone-inventor",
            relevance_score=0.9,
        ),
        "patent": Evidence(
            evidence_id="telephone:patent",
            claim_or_query=question,
            text="Bell received a patent for the telephone on March 7, 1876.",
            source_title="Patent history",
            source_url="https://example.test/telephone-patent",
            relevance_score=0.9,
        ),
        "claims": Evidence(
            evidence_id="telephone:claims",
            claim_or_query=question,
            text=(
                "Elisha Gray and Antonio Meucci made separate claims related "
                "to the telephone's invention."
            ),
            source_title="Invention history",
            source_url="https://example.test/telephone-claims",
            relevance_score=0.9,
        ),
    }

    class TelephoneRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            normalized = query.casefold()
            if "alexander" in normalized:
                return [items["bell"]]
            if "boston" in normalized or "march" in normalized:
                return [items["patent"]]
            if "elisha" in normalized or "meucci" in normalized:
                return [items["claims"]]
            return []

    retriever = TelephoneRetriever()
    components = AnalysisComponents(
        provider=provider,
        extractor=extractor,
        retriever=retriever,
        verifier=VerificationService(retriever, RuleBasedVerifier()),
        retrieval_top_k=5,
    )
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    results = payload["verification_results"]
    assert [result["verdict"] for result in results] == [
        "SUPPORTED",
        "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
    ]
    verifier_result = RuleBasedVerifier().verify(
        extractor.extract(response_text)[0],
        [items["bell"]],
    )
    assert results[0]["verdict"] == verifier_result.verdict.value
    assert results[0]["confidence"] == verifier_result.confidence
    assert results[0]["supporting_evidence"][0]["evidence_id"] == "telephone:inventor"
    assert results[1]["verdict"] != "SUPPORTED"
    assert "Boston" not in results[1]["evidence"][0]["text"]
    assert payload["overall_result"]["status"] == "INCONCLUSIVE"
    assert payload["llm_response"]["response_text"] == response_text


def test_analysis_reports_retrieval_failure_instead_of_successful_verdict() -> None:
    components = make_components()
    components.retriever.retrieve.side_effect = RuntimeError("source unavailable")
    app.dependency_overrides[get_analysis_components] = lambda: components
    try:
        response = TestClient(app).post("/analyze", json={"question": "What happened?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "Analysis pipeline unavailable"
