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


def test_explicit_negation_against_positive_evidence_is_contradicted() -> None:
    result = RuleBasedVerifier().verify(
        claim("Chandrayaan-3 did not land on the Moon."),
        [evidence("Chandrayaan-3 landed on the Moon in 2023.")],
    )

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting_evidence
    assert result.confidence > 0.65


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
            self.query = ""

        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            self.query = query
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

    assert retriever.query == "Chandrayaan-3 Moon successfully landed August 23 2023"
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


def test_uncertain_relation_falls_back_to_existing_overlap_behavior() -> None:
    result = RuleBasedVerifier().verify(
        claim("The policy addressed employment concerns."),
        [evidence("The policy addressed employment concerns.")],
    )
    assert result.verdict == Verdict.SUPPORTED