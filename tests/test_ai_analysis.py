"""Focused tests for generation, extraction, and claim graph construction."""

from pathlib import Path
from unittest.mock import Mock

import httpx
import spacy

from backend.core.config import Settings
from modules.claims import Claim, ClaimExtractor
from modules.graph import ClaimGraph
from modules.llm import LLMProviderError, OpenAICompatibleProvider


def provider_settings() -> Settings:
    return Settings(llm_api_key="test-key", llm_base_url="https://llm.test", llm_model="test-model")


def test_provider_returns_structured_response() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"model": "actual-model", "choices": [{"message": {"content": "Answer."}}]}
    client.post.return_value = response

    result = OpenAICompatibleProvider(provider_settings(), client).generate("Question?")

    assert result.question == "Question?"
    assert result.response_text == "Answer."
    assert result.model == "actual-model"
    client.post.assert_called_once()


def test_provider_failure_is_typed_and_does_not_fake_response() -> None:
    client = Mock(spec=httpx.Client)
    client.post.side_effect = httpx.TimeoutException("timed out")

    try:
        OpenAICompatibleProvider(provider_settings(), client).generate("Question?")
    except LLMProviderError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("provider failure should raise LLMProviderError")


def test_claim_extraction_preserves_sentences_and_entities() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "ORG", "pattern": "ISRO"},
        {"label": "DATE", "pattern": "2023"},
    ])

    claims = ClaimExtractor(nlp=nlp).extract("Hello! ISRO launched Chandrayaan-3 in 2023. Is it successful?")

    assert len(claims) == 1
    assert claims[0].claim_id == "C1"
    assert claims[0].original_sentence == "ISRO launched Chandrayaan-3 in 2023."
    assert [(entity.text, entity.label) for entity in claims[0].entities] == [
        ("ISRO", "ORG"), ("Chandrayaan-3", "DOMAIN_ENTITY"), ("2023", "DATE")
    ]


def test_claim_extraction_splits_independent_contrastive_facts() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract(
        "Karnataka is a state in southern India, but "
        "New Delhi has been the capital since 1911."
    )

    assert [claim.claim_text for claim in claims] == [
        "Karnataka is a state in southern India.",
        "New Delhi has been the capital since 1911.",
    ]
    assert all(claim.original_sentence == (
        "Karnataka is a state in southern India, but "
        "New Delhi has been the capital since 1911."
    ) for claim in claims)


def test_claim_extraction_keeps_conjoined_objects_together() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract(
        "The main causes of ocean tides are the gravitational pull of the "
        "sun and the moon on Earth's oceans."
    )

    assert [claim.claim_text for claim in claims] == [
        "The main causes of ocean tides are the gravitational pull of the "
        "sun and the moon on Earth's oceans."
    ]


def test_claim_extraction_removes_numbered_list_artifacts() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract(
        "The key principles of a circular economy are:\n"
        "1. Eliminate waste and pollution.\n"
        "2. Circulate products and materials at their highest value.\n"
        "3. Regenerate natural systems."
    )

    assert [claim.claim_text for claim in claims] == [
        "Eliminate waste and pollution.",
        "Circulate products and materials at their highest value.",
        "Regenerate natural systems.",
    ]


def test_claim_extraction_fallback_does_not_duplicate_spacy_entity() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "PRODUCT", "pattern": "Chandrayaan-3"}])

    claims = ClaimExtractor(nlp=nlp).extract("Chandrayaan-3 uses GPT-4.")

    assert [(entity.text, entity.label) for entity in claims[0].entities] == [
        ("Chandrayaan-3", "PRODUCT"), ("GPT-4", "DOMAIN_ENTITY")
    ]


def test_claim_extraction_filters_standalone_meta_sentences() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract("That is incorrect. BIT was established in 1979.")

    assert [claim.original_sentence for claim in claims] == ["BIT was established in 1979."]


def test_claim_extraction_filters_standalone_correct_judgment() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract("That is correct. BIT was established in 1979.")

    assert [claim.original_sentence for claim in claims] == ["BIT was established in 1979."]


def test_claim_extraction_keeps_meta_judgment_with_factual_content() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract(
        "The statement that BIT was established in 2024 is incorrect."
    )

    assert [claim.original_sentence for claim in claims] == [
        "The statement that BIT was established in 2024 is incorrect."
    ]


def test_claim_extraction_keeps_factual_content_after_meta_judgment() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    claims = ClaimExtractor(nlp=nlp).extract(
        "It is incorrect to say that BIT was established in 2024; "
        "BIT was established in 1979."
    )

    assert [claim.original_sentence for claim in claims] == [
        "It is incorrect to say that BIT was established in 2024; "
        "BIT was established in 1979."
    ]


def test_extracted_chandrayaan_claim_builds_expected_graph() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "ORG", "pattern": "ISRO"},
        {"label": "DATE", "pattern": "2023"},
    ])

    claims = ClaimExtractor(nlp=nlp).extract("ISRO launched Chandrayaan-3 in 2023.")
    graph = ClaimGraph()
    graph.add_claims(claims)

    data = graph.to_dict()
    nodes = {node["text"]: node for node in data["nodes"]}
    edges = {(edge["source"], edge["target"]): edge for edge in data["edges"]}
    mission = nodes["Chandrayaan-3"]["id"]
    organization = nodes["ISRO"]["id"]
    year = nodes["2023"]["id"]
    assert {node["text"] for node in data["nodes"]} == {"ISRO", "Chandrayaan-3", "2023"}
    assert edges[(mission, organization)]["relationship_type"] == "launched_by"
    assert edges[(mission, year)]["relationship_type"] == "launched_in"
    assert edges[(mission, organization)]["claim_ids"] == ["C1"]


def test_claim_graph_deduplicates_entities_and_serializes(tmp_path: Path) -> None:
    claims = [
        Claim(
            claim_id="C1", claim_text="ISRO launched Chandrayaan-3 in 2023.",
            original_sentence="ISRO launched Chandrayaan-3 in 2023.", sentence_index=0,
            entities=[{"text": "ISRO", "label": "ORG"}, {"text": "Chandrayaan-3", "label": "PRODUCT"}, {"text": "2023", "label": "DATE"}],
        ),
        Claim(
            claim_id="C2", claim_text="ISRO launched Chandrayaan-3 in 2023.",
            original_sentence="ISRO launched Chandrayaan-3 in 2023.", sentence_index=1,
            entities=[{"text": "ISRO", "label": "ORG"}, {"text": "Chandrayaan-3", "label": "PRODUCT"}, {"text": "2023", "label": "DATE"}],
        ),
    ]
    graph = ClaimGraph()
    graph.add_claims(claims)

    data = graph.to_dict()
    assert len(data["nodes"]) == 3
    assert {edge["relationship_type"] for edge in data["edges"]} == {"launched_by", "launched_in"}
    assert all(edge["claim_ids"] == ["C1", "C2"] for edge in data["edges"])
    graph.export_graphml(tmp_path / "claims.graphml")
    assert (tmp_path / "claims.graphml").exists()


def test_claim_graph_extracts_directional_capital_and_settlement_relationships() -> None:
    claims = ClaimExtractor().extract(
        "No, Bengaluru is not the capital of Manipur. "
        "Imphal is the capital of Manipur. "
        "Bengaluru is mentioned as a place where some Manipuris have settled."
    )
    graph = ClaimGraph()
    graph.add_claims(claims)

    data = graph.to_dict()
    node_text_by_id = {node["id"]: node["text"] for node in data["nodes"]}
    relationships = {
        (
            node_text_by_id[data["source"]],
            data["relationship_type"],
            node_text_by_id[data["target"]],
        )
        for data in data["edges"]
    }

    assert relationships == {
        ("Bengaluru", "capital_of", "Manipur"),
        ("Imphal", "capital_of", "Manipur"),
        ("Manipuris", "settled_in", "Bengaluru"),
    }