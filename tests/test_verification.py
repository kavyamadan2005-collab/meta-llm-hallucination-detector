"""Deterministic tests for the evidence-based verification baseline."""

from modules.claims import Claim
from modules.retrieval.models import Evidence
from modules.verification import RuleBasedVerifier, Verdict, VerificationService


def claim(text: str, claim_id: str = "C1") -> Claim:
    return Claim(claim_id=claim_id, claim_text=text, original_sentence=text, sentence_index=0)


def evidence(text: str, score: float = 0.9, evidence_id: str = "E1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id, claim_or_query="query", text=text,
        source_title="Test source", source_url="https://example.test/source",
        relevance_score=score,
    )


def test_strong_evidence_supports_claim_and_preserves_provenance() -> None:
    item = evidence("ISRO launched Chandrayaan-3 in 2023.")
    result = RuleBasedVerifier().verify(claim("ISRO launched Chandrayaan-3 in 2023."), [item])

    assert result.verdict == Verdict.SUPPORTED
    assert result.confidence > 0.7
    assert result.evidence[0] is item
    assert result.supporting_evidence[0].source_url == item.source_url
    assert result.contradicting_evidence == []


def test_explicitly_conflicting_year_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("ISRO launched Chandrayaan-3 in 2023."),
        [evidence("ISRO launched Chandrayaan-3 in 2022.")],
    )

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting_evidence
    assert "conflicts" in result.explanation


def test_no_evidence_is_unsupported_with_zero_confidence() -> None:
    result = RuleBasedVerifier().verify(claim("ISRO launched Chandrayaan-3 in 2023."), [])

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.confidence == 0.0
    assert "No sufficiently relevant evidence" in result.explanation


def test_low_similarity_evidence_is_not_treated_as_support() -> None:
    result = RuleBasedVerifier(min_similarity=0.8).verify(
        claim("The Eiffel Tower is located in Australia."),
        [evidence("ISRO launched Chandrayaan-3 in 2023.", score=0.2)],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []
    assert result.confidence == 0.0


def test_ambiguous_evidence_is_unsupported() -> None:
    result = RuleBasedVerifier().verify(
        claim("ISRO launched Chandrayaan-3 in 2023."),
        [evidence("ISRO and Chandrayaan-3 are important topics.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_is_bounded_for_all_verdicts() -> None:
    verifier = RuleBasedVerifier()
    cases = [
        (claim("A launched B in 2023."), [evidence("A launched B in 2023.")]),
        (claim("A launched B in 2023."), [evidence("A launched B in 2022.")]),
        (claim("A launched B in 2023."), []),
    ]

    for test_claim, test_evidence in cases:
        result = verifier.verify(test_claim, test_evidence)
        assert 0.0 <= result.confidence <= 1.0


def test_batch_service_retrieves_and_verifies_in_order() -> None:
    class StubRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [evidence(query, evidence_id=f"evidence:{query}")]

    service = VerificationService(StubRetriever(), RuleBasedVerifier())
    results = service.verify_claims([
        claim("The Moon has craters.", "C1"),
        claim("Water freezes at 0 degrees Celsius.", "C2"),
    ])

    assert [result.claim_id for result in results] == ["C1", "C2"]
    assert all(result.verdict == Verdict.SUPPORTED for result in results)
    assert results[0].evidence[0].evidence_id == "evidence:The Moon has craters."