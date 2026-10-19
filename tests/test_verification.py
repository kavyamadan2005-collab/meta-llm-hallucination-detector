"""Deterministic tests for the evidence-based verification baseline."""

import pytest

from modules.claims import Claim, NamedEntity
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


def test_corrective_not_clause_matches_equivalent_copular_evidence_below_threshold() -> None:
    claim_with_correction = Claim(
        claim_id="C1",
        claim_text="The capital of India is New Delhi, not Karnataka.",
        original_sentence="The capital of India is New Delhi, not Karnataka.",
        sentence_index=0,
        entities=[
            NamedEntity(text="India", label="GPE"),
            NamedEntity(text="New Delhi", label="GPE"),
            NamedEntity(text="Karnataka", label="GPE"),
        ],
    )
    source = evidence(
        "New Delhi is the capital of India.",
        score=0.46,
        evidence_id="direct-capital-fact",
    )

    result = RuleBasedVerifier().verify(claim_with_correction, [source])

    assert result.verdict == Verdict.SUPPORTED
    assert result.supporting_evidence == [source]
    assert result.contradicting_evidence == []


def test_current_role_with_start_year_supports_ongoing_role_claim() -> None:
    role_claim = Claim(
        claim_id="C1",
        claim_text="New Delhi has been the capital since 1911.",
        original_sentence="New Delhi has been the capital since 1911.",
        sentence_index=0,
        entities=[NamedEntity(text="New Delhi", label="GPE")],
    )

    result = RuleBasedVerifier().verify(
        role_claim,
        [
            evidence(
                "The current capital city is New Delhi, which replaced Calcutta in 1911."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_explicit_negation_against_positive_evidence_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("Chandrayaan-3 did not land on the Moon."),
        [evidence("Chandrayaan-3 landed on the Moon in 2023.")],
    )

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting_evidence
    assert result.confidence > 0.65


def test_metadata_date_before_matching_event_is_not_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("India became independent on August 15, 1947."),
        [
            evidence(
                "August 15, 2025 ┬╖ On August 15, 1947, India became free "
                "after two centuries of British rule."
            )
        ],
    )

    assert result.verdict != Verdict.CONTRADICTED


def test_publication_date_before_matching_event_is_not_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("India became independent on August 15, 1947."),
        [
            evidence(
                "Published: February 09, 2010.\n"
                "India became independent on August 15, 1947."
            )
        ],
    )

    assert result.verdict != Verdict.CONTRADICTED


def test_last_updated_date_before_matching_event_is_not_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("India became independent on August 15, 1947."),
        [
            evidence(
                "Last Updated: August 14, 2026.\n"
                "India became independent on August 15, 1947."
            )
        ],
    )

    assert result.verdict != Verdict.CONTRADICTED


def test_conflicting_event_date_with_metadata_is_contradicted() -> None:
    claim_text = "India became independent on August 15, 1947."
    evidence_texts = (
        "India became independent on August 16, 1947. Published: January 2020.",
        "Published: January 2020. India became independent on August 16, 1947.",
    )

    for text in evidence_texts:
        result = RuleBasedVerifier().verify(claim(claim_text), [evidence(text)])
        assert result.verdict == Verdict.CONTRADICTED


def test_matching_event_date_with_metadata_is_not_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The launch occurred on March 5, 2022."),
        [
            evidence(
                "Published: March 6, 2024.\n"
                "The launch occurred on March 5, 2022."
            )
        ],
    )

    assert result.verdict != Verdict.CONTRADICTED


def test_patent_date_is_not_confused_with_first_telephone_call_date() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="Bell made the first telephone call on March 10, 1876.",
        original_sentence="Bell made the first telephone call on March 10, 1876.",
        sentence_index=0,
        entities=[{"text": "Bell", "label": "PERSON"}],
    )
    verifier = RuleBasedVerifier()

    supported_event_date = verifier.verify(
        claim_data,
        [
            evidence(
                "Bell received a patent for the telephone on March 7, 1876. "
                "The first telephone call was made on March 10, 1876."
            )
        ],
    )
    conflicting_event_date = verifier.verify(
        claim_data.model_copy(
            update={"claim_text": "Bell made the first telephone call on March 7, 1876."}
        ),
        [
            evidence(
                "The first telephone call was made on March 10, 1876. "
                "Bell received a patent for the telephone on March 7, 1876."
            )
        ],
    )

    assert supported_event_date.verdict != Verdict.CONTRADICTED
    assert conflicting_event_date.verdict == Verdict.CONTRADICTED


def test_positive_claim_is_supported_by_matching_inflected_evidence() -> None:
    result = RuleBasedVerifier().verify(
        claim("India won the match."),
        [evidence("India won the match.")],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_explicitly_conflicting_year_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("ISRO launched Chandrayaan-3 in 2023."),
        [evidence("ISRO launched Chandrayaan-3 in 2022.")],
    )

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting_evidence
    assert "conflicts" in result.explanation


def test_unrelated_biographical_year_does_not_contradict_patent_year() -> None:
    result = RuleBasedVerifier().verify(
        claim(
            "Alexander Graham Bell was awarded the first U.S. patent for "
            "the invention of the telephone in 1876."
        ),
        [
            evidence(
                "Bell died on August 2, 1922. He was credited with patenting "
                "the first practical telephone. Bell received the first U.S. "
                "patent for the telephone on March 7, 1876."
            )
        ],
    )

    assert result.verdict != Verdict.CONTRADICTED
    assert result.contradicting_evidence == []


def test_conflicting_patent_year_remains_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("Bell received the telephone patent in 1876."),
        [evidence("Bell received the telephone patent in 1877.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_same_entity_with_conflicting_explicit_dates_is_contradicted() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="Vikram touched down on September 7, 2023.",
        original_sentence="Vikram touched down on September 7, 2023.",
        sentence_index=0,
        entities=[{"text": "Vikram", "label": "DOMAIN_ENTITY"}],
    )

    result = RuleBasedVerifier().verify(
        claim_data,
        [evidence("Vikram successfully soft-landed on August 23, 2023.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_same_entity_with_same_explicit_date_is_not_contradicted() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="Vikram touched down on August 23, 2023.",
        original_sentence="Vikram touched down on August 23, 2023.",
        sentence_index=0,
        entities=[{"text": "Vikram", "label": "DOMAIN_ENTITY"}],
    )

    result = RuleBasedVerifier().verify(
        claim_data,
        [evidence("Vikram successfully soft-landed on August 23, 2023.")],
    )

    assert result.verdict != Verdict.CONTRADICTED


def test_same_complete_date_supports_matching_event_without_entities() -> None:
    result = RuleBasedVerifier().verify(
        claim("The launch occurred on March 5, 2022."),
        [evidence("The launch occurred on March 5, 2022.")],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_one_day_mismatch_for_same_event_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The launch occurred on March 5, 2022."),
        [evidence("The launch occurred on March 6, 2022.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_different_dates_for_unrelated_events_are_not_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The launch occurred on March 5, 2022."),
        [evidence("The sample was collected on March 6, 2022.")],
    )

    assert result.verdict != Verdict.CONTRADICTED


def test_year_mismatch_behavior_is_preserved() -> None:
    result = RuleBasedVerifier().verify(
        claim("The launch occurred on March 5, 2022."),
        [evidence("The launch occurred on March 5, 2021.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_safe_predicate_paraphrases_support_matching_objects() -> None:
    verifier = RuleBasedVerifier()

    assert verifier.verify(
        claim("The city increased its population."),
        [evidence("The urban population grew.")],
    ).verdict == Verdict.SUPPORTED
    assert verifier.verify(
        claim("The algorithm sorts the list."),
        [evidence("The procedure orders the list.")],
    ).verdict == Verdict.SUPPORTED


def test_safe_predicate_paraphrases_do_not_support_different_objects() -> None:
    result = RuleBasedVerifier().verify(
        claim("The algorithm sorts the list."),
        [evidence("The procedure orders the table.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED


def test_relation_direction_is_preserved_for_founded_by() -> None:
    verifier = RuleBasedVerifier()

    assert verifier.verify(
        claim("Alpha founded Beta."),
        [evidence("Alpha founded Beta.")],
    ).verdict == Verdict.SUPPORTED
    assert verifier.verify(
        claim("Alpha founded Beta."),
        [evidence("Alpha was founded by Beta.")],
    ).verdict != Verdict.SUPPORTED


def test_different_entities_with_conflicting_explicit_dates_are_not_contradicted() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="Vikram touched down on September 7, 2023.",
        original_sentence="Vikram touched down on September 7, 2023.",
        sentence_index=0,
        entities=[{"text": "Vikram", "label": "DOMAIN_ENTITY"}],
    )

    result = RuleBasedVerifier().verify(
        claim_data,
        [evidence("Pragyan successfully soft-landed on August 23, 2023.")],
    )

    assert result.verdict != Verdict.CONTRADICTED


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


def test_exact_claim_date_requires_evidence_date() -> None:
    result = RuleBasedVerifier().verify(
        claim("Chandrayaan-3 landed on the Moon on August 23, 2023."),
        [evidence("Chandrayaan-3 landed on the Moon in 2023.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []


def test_unrelated_negation_does_not_create_contradiction() -> None:
    result = RuleBasedVerifier().verify(
        claim("India won the match."),
        [evidence("The report was not about India winning the match.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED


def test_unrelated_no_longer_clause_does_not_contradict_supported_claim() -> None:
    result = RuleBasedVerifier().verify(
        claim("In India, Independence Day is celebrated every year."),
        [
            evidence(
                "This day celebrates the date in 1947 when the Indian "
                "Independence Act came into effect, which established India "
                "and Pakistan as separate countries, no longer under British "
                "imperialist rule."
            )
        ],
    )

    assert result.verdict != Verdict.CONTRADICTED
    assert result.contradicting_evidence == []


def test_not_just_phrase_does_not_create_a_false_contradiction() -> None:
    result = RuleBasedVerifier().verify(
        claim("In India, Republic Day is celebrated on January 26."),
        [
            evidence(
                "India's Republic Day, celebrated on January 26 every year, "
                "is not just a national holiday; it marks the adoption of "
                "the Constitution."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED
    assert result.contradicting_evidence == []


def test_causal_tide_claim_matches_causal_evidence() -> None:
    result = RuleBasedVerifier().verify(
        claim(
            "The main causes of ocean tides are the gravitational pull of "
            "the Moon and, to a lesser extent, the Sun on Earth's oceans."
        ),
        [
            evidence(
                "Ocean tides result from the gravitational attraction of the "
                "sun and moon on the oceans of the earth."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED
    assert result.contradicting_evidence == []


def test_chandrayaan_success_is_not_negated_by_unrelated_failures() -> None:
    result = RuleBasedVerifier().verify(
        claim("Chandrayaan-3 successfully soft-landed on the Moon."),
        [
            evidence(
                "Chandrayaan-3 successfully achieved a soft landing near the "
                "Moon's south pole. This followed failed missions by other "
                "countries and an earlier failed attempt by India."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED
    assert result.contradicting_evidence == []


def test_verify_claims_reuses_generation_context_evidence() -> None:
    source = evidence(
        "Only three Indian states start with A: Andhra Pradesh, "
        "Arunachal Pradesh, and Assam.",
        evidence_id="indian-states-a",
    )

    class IrrelevantRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [evidence("Unrelated evidence.", evidence_id="unrelated")]

    service = VerificationService(
        IrrelevantRetriever(),
        RuleBasedVerifier(),
    )
    result = service.verify_claims(
        [claim(source.text)],
        context_evidence=[source],
    )[0]

    assert result.verdict == Verdict.SUPPORTED
    assert source in result.supporting_evidence


def test_direct_list_evidence_can_verify_below_similarity_threshold() -> None:
    claim_text = (
        "There are only three Indian states whose names start with A: "
        "Andhra Pradesh, Arunachal Pradesh, and Assam."
    )
    source = evidence(
        "| States | Andhra Pradesh Arunachal Pradesh Assam Bihar "
        "Chhattisgarh Delhi Goa Gujarat Haryana Himachal Pradesh Jharkhand "
        "Karnataka Kerala Maharashtra Madhya Pradesh Manipur Meghalaya "
        "Mizoram Nagaland Odisha Punjab Rajasthan Sikkim Tamil Nadu "
        "Tripura Telangana Uttar Pradesh Uttarakhand West Bengal "
        "Andaman & Nicobar (UT) Chandigarh (UT) |",
        score=0.64,
    )

    result = RuleBasedVerifier().verify(claim(claim_text), [source])

    assert result.verdict == Verdict.SUPPORTED
    assert source in result.supporting_evidence


def test_compound_claim_requires_all_extracted_entities() -> None:
    compound = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-1 and Chandrayaan-2 landed on the Moon.",
        original_sentence="Chandrayaan-1 and Chandrayaan-2 landed on the Moon.",
        sentence_index=0,
        entities=[
            {"text": "Chandrayaan-1", "label": "DOMAIN_ENTITY"},
            {"text": "Chandrayaan-2", "label": "DOMAIN_ENTITY"},
            {"text": "Moon", "label": "LOC"},
        ],
    )

    result = RuleBasedVerifier().verify(
        compound,
        [evidence("Chandrayaan-1 landed on the Moon.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED


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
    assert results[0].evidence[0].evidence_id == "evidence:Moon has craters"


def test_focused_query_preserves_entities_dates_and_action() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 successfully landed on the Moon on August 23, 2023.",
        original_sentence="Chandrayaan-3 successfully landed on the Moon on August 23, 2023.",
        sentence_index=0,
        entities=[
            {"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"},
            {"text": "Moon", "label": "LOC"},
        ],
    )

    query = VerificationService._build_retrieval_query(claim_data)

    assert all(token in query for token in ("Chandrayaan-3", "Moon", "landed", "August", "23", "2023"))


def test_focused_query_preserves_organisation_and_mission_without_context() -> None:
    claim_data = Claim(
        claim_id="C2",
        claim_text="It was a mission by the Indian Space Research Organisation (ISRO).",
        original_sentence="It was a mission by the Indian Space Research Organisation (ISRO).",
        sentence_index=1,
        entities=[
            {"text": "Indian Space Research Organisation", "label": "ORG"},
            {"text": "ISRO", "label": "ORG"},
        ],
    )

    query = VerificationService._build_retrieval_query(claim_data)

    assert "Indian Space Research Organisation" in query
    assert "ISRO" in query
    assert "mission" in query
    assert "Chandrayaan-3" not in query


def test_focused_query_falls_back_for_empty_or_too_short_result() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="It is.",
        original_sentence="It is.",
        sentence_index=0,
    )

    assert VerificationService._build_retrieval_query(claim_data) == claim_data.claim_text


def test_service_passes_focused_query_and_preserves_original_claim() -> None:
    class CapturingRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            self.queries.append(query)
            return [evidence("Chandrayaan-3 landed on the Moon on August 23, 2023.")]

    retriever = CapturingRetriever()
    claim_data = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 successfully landed on the Moon on August 23, 2023.",
        original_sentence="Chandrayaan-3 successfully landed on the Moon on August 23, 2023.",
        sentence_index=0,
        entities=[{"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"}, {"text": "Moon", "label": "LOC"}],
    )
    service = VerificationService(retriever, RuleBasedVerifier())

    result = service.verify_claim(claim_data)

    assert retriever.queries == [
        "Chandrayaan-3 Moon successfully landed August 23 2023",
        "Chandrayaan-3 Moon successfully landed",
    ]
    assert result.claim_text == claim_data.claim_text


def test_referring_mission_claim_inherits_previous_main_entity() -> None:
    previous = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 successfully landed on the Moon.",
        original_sentence="Chandrayaan-3 successfully landed on the Moon.",
        sentence_index=0,
        entities=[{"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"}, {"text": "Moon", "label": "LOC"}],
    )
    current = Claim(
        claim_id="C2",
        claim_text="The mission's Vikram lander deployed Pragyan.",
        original_sentence="The mission's Vikram lander deployed Pragyan.",
        sentence_index=1,
        entities=[
            {"text": "Vikram", "label": "ORG"},
            {"text": "Pragyan", "label": "DOMAIN_ENTITY"},
        ],
    )

    query = VerificationService._build_retrieval_query(
        current,
        VerificationService._referring_context(current, previous),
    )

    assert query.startswith("Chandrayaan-3")
    assert "Vikram" in query
    assert "Pragyan" in query


def test_referring_mission_statement_inherits_previous_main_entity() -> None:
    previous = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 landed successfully.",
        original_sentence="Chandrayaan-3 landed successfully.",
        sentence_index=0,
        entities=[{"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"}],
    )
    current = Claim(
        claim_id="C2",
        claim_text="The mission was considered a partial success.",
        original_sentence="The mission was considered a partial success.",
        sentence_index=1,
    )

    query = VerificationService._build_retrieval_query(
        current,
        VerificationService._referring_context(current, previous),
    )

    assert "Chandrayaan-3" in query
    assert "mission" in query
    assert "partial" in query
    assert current.claim_text == "The mission was considered a partial success."


def test_unrelated_claim_does_not_inherit_previous_entity() -> None:
    previous = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 landed successfully.",
        original_sentence="Chandrayaan-3 landed successfully.",
        sentence_index=0,
        entities=[{"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"}],
    )
    current = Claim(
        claim_id="C2",
        claim_text="India launched another satellite in 2024.",
        original_sentence="India launched another satellite in 2024.",
        sentence_index=1,
    )

    query = VerificationService._build_retrieval_query(
        current,
        VerificationService._referring_context(current, previous),
    )

    assert "Chandrayaan-3" not in query
    assert "India" in query
    assert "2024" in query


def test_no_previous_claim_keeps_focused_query_behavior() -> None:
    current = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 landed on the Moon.",
        original_sentence="Chandrayaan-3 landed on the Moon.",
        sentence_index=0,
        entities=[{"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"}, {"text": "Moon", "label": "LOC"}],
    )

    assert VerificationService._build_retrieval_query(
        current,
        VerificationService._referring_context(current, None),
    ) == "Chandrayaan-3 Moon landed"


def test_verify_claims_preserves_original_claim_text_with_context() -> None:
    class CapturingRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            self.queries.append(query)
            return []

    retriever = CapturingRetriever()
    previous = Claim(
        claim_id="C1",
        claim_text="Chandrayaan-3 landed successfully.",
        original_sentence="Chandrayaan-3 landed successfully.",
        sentence_index=0,
        entities=[{"text": "Chandrayaan-3", "label": "DOMAIN_ENTITY"}],
    )
    current = claim("The mission was considered a success.", "C2")
    results = VerificationService(retriever, RuleBasedVerifier()).verify_claims([previous, current])

    assert "Chandrayaan-3" in retriever.queries[1]
    assert results[1].claim_text == current.claim_text


def test_generic_referring_expressions_inherit_one_previous_entity() -> None:
    previous = Claim(
        claim_id="C1",
        claim_text="Microsoft released Windows 11.",
        original_sentence="Microsoft released Windows 11.",
        sentence_index=0,
        entities=[{"text": "Microsoft", "label": "ORG"}, {"text": "Windows 11", "label": "PRODUCT"}],
    )
    referring_claims = [
        "The company later introduced a new product.",
        "The organization later announced a change.",
        "The country later changed its policy.",
        "He later published a report.",
        "She later published a report.",
        "Its later release gained attention.",
        "His later report gained attention.",
        "Her later report gained attention.",
        "Their later report gained attention.",
        "The President later issued a statement.",
    ]

    for text in referring_claims:
        current = Claim(
            claim_id="C2",
            claim_text=text,
            original_sentence=text,
            sentence_index=1,
        )
        assert VerificationService._referring_context(current, previous) == "Microsoft"
        assert "Microsoft" in VerificationService._build_retrieval_query(
            current,
            VerificationService._referring_context(current, previous),
        )


def test_existing_mission_context_and_entity_filtering_are_preserved() -> None:
    previous = Claim(
        claim_id="C1",
        claim_text="Mission Alpha launched in 2024.",
        original_sentence="Mission Alpha launched in 2024.",
        sentence_index=0,
        entities=[
            {"text": "2024", "label": "DATE"},
            {"text": "Mission Alpha", "label": "DOMAIN_ENTITY"},
        ],
    )
    current = Claim(
        claim_id="C2",
        claim_text="The mission later completed its objective.",
        original_sentence="The mission later completed its objective.",
        sentence_index=1,
    )

    assert VerificationService._referring_context(current, previous) == "Mission Alpha"


def test_generic_context_does_not_inherit_for_unrelated_or_first_claim() -> None:
    previous = Claim(
        claim_id="C1",
        claim_text="Microsoft released Windows 11.",
        original_sentence="Microsoft released Windows 11.",
        sentence_index=0,
        entities=[{"text": "Microsoft", "label": "ORG"}],
    )
    unrelated = Claim(
        claim_id="C2",
        claim_text="Photosynthesis occurs in plants.",
        original_sentence="Photosynthesis occurs in plants.",
        sentence_index=1,
    )
    first = Claim(
        claim_id="C1",
        claim_text="The company released a product.",
        original_sentence="The company released a product.",
        sentence_index=0,
    )

    assert VerificationService._referring_context(unrelated, previous) is None
    assert VerificationService._referring_context(first, None) is None


def test_context_is_query_only_and_original_claim_text_is_preserved() -> None:
    class CapturingRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            self.queries.append(query)
            return []

    previous = Claim(
        claim_id="C1",
        claim_text="Acme developed a device.",
        original_sentence="Acme developed a device.",
        sentence_index=0,
        entities=[{"text": "Acme", "label": "ORG"}],
    )
    current_text = "The company later improved it."
    current = Claim(
        claim_id="C2",
        claim_text=current_text,
        original_sentence=current_text,
        sentence_index=1,
    )
    retriever = CapturingRetriever()
    results = VerificationService(retriever, RuleBasedVerifier()).verify_claims(
        [previous, current]
    )

    assert "Acme" in retriever.queries[1]
    assert results[1].claim_text == current_text


def test_relation_reversal_is_not_supported() -> None:
    result = RuleBasedVerifier().verify(
        claim("Acme acquired Beta."),
        [evidence("Beta acquired Acme.")],
    )
    assert result.verdict == Verdict.UNSUPPORTED


def test_subject_object_mismatch_is_not_supported() -> None:
    result = RuleBasedVerifier().verify(
        claim("The engine uses hydrogen."),
        [evidence("The engine uses gasoline.")],
    )
    assert result.verdict == Verdict.UNSUPPORTED


def test_predicate_conflict_is_not_supported() -> None:
    result = RuleBasedVerifier().verify(
        claim("The European Union adopted the regulation."),
        [evidence("The European Union rejected the regulation.")],
    )
    assert result.verdict == Verdict.UNSUPPORTED


def test_entity_identity_mismatch_is_not_supported() -> None:
    claim_data = Claim(
        claim_id="C1",
        claim_text="Dr. Lee led the study.",
        original_sentence="Dr. Lee led the study.",
        sentence_index=0,
        entities=[{"text": "Dr. Lee", "label": "PERSON"}],
    )
    result = RuleBasedVerifier().verify(
        claim_data,
        [evidence("Dr. Chen led the study.")],
    )
    assert result.verdict == Verdict.UNSUPPORTED


def test_basic_modal_contradiction_is_detected() -> None:
    result = RuleBasedVerifier().verify(
        claim("The machine cannot produce electricity."),
        [evidence("The machine can produce electricity.")],
    )
    assert result.verdict == Verdict.CONTRADICTED


def test_safe_paraphrase_equivalences_are_supported() -> None:
    examples = [
        ("The company purchased the startup.", "The firm acquired the young company."),
        ("The scientist discovered the particle.", "The researcher found the particle."),
        ("The team won the championship.", "The team became champions."),
        ("The spacecraft entered orbit.", "The probe was placed into orbit."),
        ("The treaty ended the conflict.", "The agreement brought the war to an end."),
    ]
    for claim_text, evidence_text in examples:
        result = RuleBasedVerifier().verify(claim(claim_text), [evidence(evidence_text)])
        assert result.verdict == Verdict.SUPPORTED


def test_partial_compound_evidence_is_not_full_support() -> None:
    result = RuleBasedVerifier().verify(
        claim("The rover mapped the crater and collected samples."),
        [evidence("The rover mapped the crater.")],
    )
    assert result.verdict == Verdict.UNSUPPORTED


def test_obvious_numeric_count_conflict_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The box contains 12 items."),
        [evidence("The box contains 20 items.")],
    )
    assert result.verdict == Verdict.CONTRADICTED


def test_explicit_list_count_conflict_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The report lists five affected regions."),
        [evidence("The report lists exactly three affected regions.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_positive_list_membership_evidence_contradicts_universal_negative() -> None:
    result = RuleBasedVerifier().verify(
        claim("No states in the country start with A."),
        [evidence("Andhra, Arunachal, and Assam are states that start with A.")],
    )

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting_evidence


def test_positive_member_contradicts_compound_count_and_none_claim() -> None:
    result = RuleBasedVerifier().verify(
        claim(
            "India has 28 states and 8 Union Territories, "
            "but none start with A."
        ),
        [
            evidence(
                "Andhra Pradesh, Arunachal Pradesh, and Assam are "
                "Indian states that start with A."
            )
        ],
    )

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting_evidence


def test_missing_list_evidence_does_not_prove_a_universal_negative() -> None:
    result = RuleBasedVerifier().verify(
        claim("No states in the country start with Z."),
        [evidence("The country has 28 states.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []


def test_refusal_to_find_list_is_not_supported_by_list_membership_evidence() -> None:
    result = RuleBasedVerifier().verify(
        claim("I couldn't find any information on states that start with A."),
        [
            evidence(
                "Andhra Pradesh, Arunachal Pradesh, and Assam are "
                "Indian states that start with A."
            )
        ],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []


def test_partial_list_evidence_does_not_prove_an_explicit_count() -> None:
    result = RuleBasedVerifier().verify(
        claim("The list contains five states."),
        [evidence("Andhra Pradesh is a state in India.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED


def test_uncertain_relation_falls_back_to_existing_overlap_behavior() -> None:
    result = RuleBasedVerifier().verify(
        claim("The policy addressed employment concerns."),
        [evidence("The policy addressed employment concerns.")],
    )
    assert result.verdict == Verdict.SUPPORTED


def test_everest_compound_claim_is_not_supported_by_partial_location_evidence() -> None:
    claim_text = (
        "Mount Everest, located in the Himalayas on the border between Nepal and Tibet, "
        "China, is the highest mountain in the world, with a height of approximately "
        "8,848 meters (29,029 feet) above sea level."
    )
    evidence_text = (
        "Mount Everest is the highest mountain in the world, at 8,848.86 metres "
        "(29,031.7 ft) above sea level and lies on the border between Nepal and China."
    )

    result = RuleBasedVerifier().verify(
        claim(claim_text),
        [evidence(evidence_text, score=0.86)],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []


def test_everest_core_fact_is_supported_by_approximate_measurement_paraphrase() -> None:
    result = RuleBasedVerifier().verify(
        claim(
            "Mount Everest is the world's highest mountain at approximately "
            "8,848 meters above sea level."
        ),
        [
            evidence(
                "Mount Everest is the highest mountain in the world, at "
                "8,848.86 metres above sea level."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_irrelevant_high_relevance_evidence_is_not_support() -> None:
    result = RuleBasedVerifier().verify(
        claim("The Eiffel Tower is located in Australia."),
        [evidence("Australia is home to many native species.", score=0.99)],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []
    assert result.confidence == 0.0


def test_subject_mismatch_with_shared_object_is_not_support() -> None:
    result = RuleBasedVerifier().verify(
        claim("Paris is in France."),
        [evidence("London is in France.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED


def test_explicitly_conflicting_location_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The Eiffel Tower is in France."),
        [evidence("The Eiffel Tower is in Texas.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_conflicting_location_for_ambiguous_place_name_is_unsupported() -> None:
    place_claim = Claim(
        claim_id="C1",
        claim_text="Paris is in France.",
        original_sentence="Paris is in France.",
        sentence_index=0,
        entities=[
            NamedEntity(text="Paris", label="GPE"),
            NamedEntity(text="France", label="GPE"),
        ],
    )
    result = RuleBasedVerifier().verify(
        place_claim,
        [evidence("Paris is in Texas.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []
    assert result.contradicting_evidence == []


def test_conflicting_measurement_for_same_attribute_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("The tower is 300 meters tall."),
        [evidence("The tower is 250 meters tall.")],
    )

    assert result.verdict == Verdict.CONTRADICTED


def test_patenting_does_not_establish_invention() -> None:
    result = RuleBasedVerifier().verify(
        claim("Bell is credited with inventing the first practical telephone."),
        [evidence("Bell is credited with patenting the first practical telephone.")],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert result.supporting_evidence == []
    assert result.contradicting_evidence == []


def test_patent_evidence_with_invention_context_supports_patent_claim() -> None:
    result = RuleBasedVerifier().verify(
        claim(
            "Alexander Graham Bell was awarded the first U.S. patent for "
            "the invention of the telephone in 1876."
        ),
        [
            evidence(
                "Alexander Graham Bell was awarded the first U.S. patent for "
                "the invention of the telephone in 1876."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_unrelated_negation_elsewhere_in_passage_does_not_cancel_support() -> None:
    result = RuleBasedVerifier().verify(
        claim("India won the match."),
        [
            evidence(
                "India won the match. The report was not about the weather."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_conflicting_sources_produce_insufficient_evidence() -> None:
    result = RuleBasedVerifier().verify(
        claim("The door is open."),
        [
            evidence("The door is open.", evidence_id="supporting"),
            evidence("The door is not open.", evidence_id="contradicting"),
        ],
    )

    assert result.verdict == Verdict.UNSUPPORTED
    assert [item.evidence_id for item in result.supporting_evidence] == ["supporting"]
    assert [item.evidence_id for item in result.contradicting_evidence] == ["contradicting"]
    assert result.confidence == 0.0


def test_different_historical_milestones_are_not_date_conflicts() -> None:
    result = RuleBasedVerifier().verify(
        claim("The term bit was coined in 1947 by John Tukey."),
        [
            evidence(
                "The term bit was first used in an article by Claude Shannon "
                "in 1948."
            ),
            evidence("John Tukey coined the term bit in 1947."),
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_compound_claim_can_be_supported_by_separate_statements() -> None:
    result = RuleBasedVerifier().verify(
        claim("The treaty reduced tariffs and increased trade."),
        [
            evidence(
                "The treaty reduced tariffs. The treaty increased trade."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_contrastive_claim_keeps_an_explicit_second_subject() -> None:
    result = RuleBasedVerifier().verify(
        claim(
            "Karnataka is a state in southern India, but "
            "New Delhi is the capital of India."
        ),
        [
            evidence(
                "Karnataka is a state in southern India. "
                "New Delhi is the capital of India."
            )
        ],
    )

    assert result.verdict == Verdict.SUPPORTED


def test_repeated_verification_is_deterministic() -> None:
    verifier = RuleBasedVerifier()
    test_claim = claim("The machine can produce electricity.")
    test_evidence = [evidence("The machine cannot produce electricity.")]

    results = [verifier.verify(test_claim, test_evidence) for _ in range(3)]

    assert [result.verdict for result in results] == [
        Verdict.CONTRADICTED,
        Verdict.CONTRADICTED,
        Verdict.CONTRADICTED,
    ]
    assert len({result.confidence for result in results}) == 1


def test_retrieval_failure_is_not_converted_to_an_evidence_verdict() -> None:
    class FailedRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            raise RuntimeError("source unavailable")

    service = VerificationService(FailedRetriever(), RuleBasedVerifier())

    with pytest.raises(RuntimeError, match="source unavailable"):
        service.verify_claim(claim("The Moon has craters."))


def test_referring_pronoun_is_resolved_for_verification_only() -> None:
    class ContextRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [
                evidence(
                    "Alexander Graham Bell won the match.",
                    evidence_id="context-evidence",
                )
            ]

    previous = Claim(
        claim_id="C1",
        claim_text="Alexander Graham Bell invented the telephone.",
        original_sentence="Alexander Graham Bell invented the telephone.",
        sentence_index=0,
        entities=[NamedEntity(text="Alexander Graham Bell", label="PERSON")],
    )
    referring_claim = claim("He won the match.", "C2")
    results = VerificationService(
        ContextRetriever(),
        RuleBasedVerifier(),
    ).verify_claims([previous, referring_claim])

    assert results[1].verdict == Verdict.SUPPORTED
    assert results[1].claim_text == referring_claim.claim_text