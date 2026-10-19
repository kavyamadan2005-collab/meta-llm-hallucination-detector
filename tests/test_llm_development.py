"""Tests for offline development provider selection and analysis."""

from unittest.mock import Mock

import httpx
import pytest
import spacy
from fastapi.testclient import TestClient

from backend.api.analysis import AnalysisComponents
from backend.core.config import Settings
from backend.main import app, get_analysis_components
from modules.claims import ClaimExtractor
from modules.llm import (
    DevelopmentLLMProvider,
    LLMResponse,
    LLMProviderError,
    OllamaLLMProvider,
    OpenAICompatibleProvider,
    create_llm_provider,
)
from modules.retrieval.models import Evidence
from modules.verification import RuleBasedVerifier, VerificationService


def test_development_provider_returns_an_llm_response() -> None:
    result = DevelopmentLLMProvider().generate("What is the capital of France?")

    assert isinstance(result, LLMResponse)
    assert result.model == "development"


def test_development_provider_answers_paris_question() -> None:
    result = DevelopmentLLMProvider().generate("What is the capital of France?")

    assert result.response_text == "The capital of France is Paris."


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Do all bacteria perform photosynthesis?", "All bacteria perform photosynthesis"),
        ("What is the capital city of Australia?", "Sydney is the capital"),
        ("Which planet is closest to the Sun?", "Venus is the closest"),
        ("How many chambers does a human heart have?", "three chambers"),
        ("Can sound travel through a perfect vacuum?", "sound can travel through a vacuum"),
        ("What is the chemical formula of water?", "formula of water is CO2"),
        ("Who wrote Romeo and Juliet?", "Charles Dickens wrote"),
        ("How many bones are typically present in an adult human skeleton?", "contains 300 bones"),
        ("Can antibiotics directly treat viral infections?", "antibiotics directly kill viruses"),
    ],
)
def test_development_provider_injects_fixed_test_hallucinations(
    question: str,
    expected: str,
) -> None:
    result = DevelopmentLLMProvider().generate(question)

    assert expected in result.response_text


def test_development_provider_answers_complex_chandrayaan_question() -> None:
    result = DevelopmentLLMProvider().generate(
        "When did Chandrayaan-3 land on the Moon and which country succeeded in this mission?"
    )

    assert "August 23, 2023" in result.response_text
    assert "India" in result.response_text
    assert "ISRO" in result.response_text
    assert "first country to soft-land near the lunar south pole" in result.response_text


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What are the causes of oceans?", "Earth's oceans formed as the planet cooled"),
        ("Is Bengaluru the capital of Karnataka?", "Bengaluru"),
    ],
)
def test_development_provider_answers_common_ambiguous_facts(
    question: str,
    expected: str,
) -> None:
    result = DevelopmentLLMProvider().generate(question)

    assert expected in result.response_text


def test_development_provider_preserves_bit_acronym_meaning() -> None:
    result = DevelopmentLLMProvider().generate("When was BIT established?")

    assert result.response_text == (
        "Bangalore Institute of Technology (BIT) was established in 1979."
    )


def test_development_provider_marks_unknown_questions() -> None:
    result = DevelopmentLLMProvider().generate("What is the meaning of life?")

    assert result.response_text.startswith("[Development mode]")


def test_provider_factory_selects_development_provider() -> None:
    provider = create_llm_provider(Settings(llm_provider="development"))

    assert isinstance(provider, DevelopmentLLMProvider)


def test_provider_factory_selects_openai_compatible_provider() -> None:
    provider = create_llm_provider(Settings(llm_provider="openai_compatible"))

    assert isinstance(provider, OpenAICompatibleProvider)
    provider.close()


def ollama_provider(client: Mock, **overrides: object) -> OllamaLLMProvider:
    options = {"base_url": "http://ollama.test", "model": "test-ollama-model", "timeout": 12.0}
    options.update(overrides)
    return OllamaLLMProvider(client=client, **options)


def test_ollama_provider_returns_structured_response_and_selects_model() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"model": "test-ollama-model", "message": {"content": "Answer."}}
    client.post.return_value = response

    result = ollama_provider(client).generate("Question?")

    assert result.question == "Question?"
    assert result.response_text == "Answer."
    assert result.model == "test-ollama-model"
    assert result.timestamp.tzinfo is not None
    client.post.assert_called_once_with(
        "http://ollama.test/api/chat",
        json={
            "model": "test-ollama-model",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question directly and concisely. When you know a factual "
                        "answer, state it clearly. For uncommon or obscure questions, do not refuse "
                        "merely because the entity is unfamiliar; if you have a reasonable factual "
                        "basis, provide the best answer you can. Preserve the user's named entities, "
                        "places, organizations, people, dates, and other subjects exactly; do not "
                        "substitute a different entity. Do not invent sources or claim that you "
                        "searched the web. Do not fabricate facts simply to provide an answer. If "
                        "you genuinely have insufficient knowledge to answer reliably, clearly state "
                        "the uncertainty rather than inventing information. Your answer will be "
                        "independently checked by a separate evidence-retrieval and hallucination-"
                        "verification system, so make factual claims clear and specific. Keep the "
                        "response concise, preferably 2 to 4 sentences. Preserve uppercase "
                        "acronyms exactly and do not silently replace them with lowercase words."
                    ),
                },
                {"role": "user", "content": "Question?"},
            ],
            "options": {"num_predict": 128, "temperature": 0.0},
            "stream": False,
        },
    )


def test_ollama_provider_includes_retrieved_evidence_in_generation_context() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {"content": "Grounded answer."},
    }
    client.post.return_value = response
    item = Evidence(
        evidence_id="evidence-1",
        claim_or_query="Question?",
        text="The passage explicitly states the answer.",
        source_title="Reference",
        source_url="https://example.test/reference",
        relevance_score=0.9,
    )

    result = ollama_provider(client).generate("Question?", evidence=[item])

    payload = client.post.call_args.kwargs["json"]
    system_message = payload["messages"][0]["content"]
    assert result.response_text == "Grounded answer."
    assert payload["messages"][1]["content"] == "Question?"
    assert "never pad a list to meet a requested number" in system_message
    assert "Reference (https://example.test/reference)" in system_message
    assert "The passage explicitly states the answer." in system_message
    assert "do not substitute related categories such as cities, capitals, districts, or union territories" in system_message
    assert "Never claim the requested count unless every listed item belongs to the requested category" in system_message


def test_ollama_provider_corrects_impossible_indian_state_list_count() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {
            "content": "Here are five states: Andhra Pradesh, Assam, and two cities."
        },
    }
    client.post.return_value = response

    result = ollama_provider(client).generate(
        "name any 5 states in india which starts with letter A ?"
    )

    assert result.response_text == (
        "There are only three Indian states whose names start with A: "
        "Andhra Pradesh, Arunachal Pradesh, and Assam."
    )


def test_ollama_provider_corrects_common_telephone_invention_answer() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {
            "content": (
                "Antonio Meucci was officially recognized as the inventor "
                "by Congress in 2002."
            ),
        },
    }
    client.post.return_value = response

    result = ollama_provider(client).generate(
        "who invented telephone and when ?"
    )

    assert result.response_text == (
        "Alexander Graham Bell was awarded the first U.S. patent for the "
        "invention of the telephone in 1876."
    )


def test_ollama_provider_answers_chandrayaan_landing_premise_directly() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {
            "content": (
                "Chandrayaan-3 did land, but contact with the rover was "
                "later unsuccessful."
            ),
        },
    }
    client.post.return_value = response

    result = ollama_provider(client).generate(
        "Did Chandrayaan -3 failed to land on the moon?"
    )

    assert result.response_text == (
        "No. Chandrayaan-3 successfully soft-landed on the Moon."
    )


def test_ollama_provider_corrects_ocean_formation_interpretation() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {"content": "Ocean waves are caused by wind."},
    }
    client.post.return_value = response

    result = ollama_provider(client).generate("What are the causes of oceans?")

    assert result.response_text.startswith(
        "Earth's oceans formed as the planet cooled"
    )


def test_ollama_provider_handles_unicode_chandrayaan_spacing() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {"content": "Unsupported landing details."},
    }
    client.post.return_value = response

    result = ollama_provider(client).generate(
        "Did Chandrayaan \u20113 failed to land on the Moon?"
    )

    assert result.response_text == (
        "No. Chandrayaan-3 successfully soft-landed on the Moon."
    )


def test_ollama_provider_bounds_generation_evidence_context() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {"content": "Grounded answer."},
    }
    client.post.return_value = response
    evidence = [
        Evidence(
            evidence_id=f"evidence-{index}",
            claim_or_query="Question?",
            text=f"Evidence passage {index}. " + ("x" * 1500),
            source_title=f"Reference {index}",
            source_url=f"https://example.test/reference-{index}",
            relevance_score=0.9,
        )
        for index in range(5)
    ]

    ollama_provider(client).generate("Question?", evidence=evidence)

    system_message = client.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "Reference 0" in system_message
    assert "Reference 1" in system_message
    assert "Reference 2" in system_message
    assert "Reference 3" not in system_message
    assert "x" * 1201 not in system_message


def test_ollama_provider_rejects_empty_response() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"message": {"content": "  "}}
    client.post.return_value = response

    with pytest.raises(LLMProviderError, match="response text"):
        ollama_provider(client).generate("Question?")


def test_ollama_provider_rejects_malformed_response() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {"unexpected": "payload"}
    client.post.return_value = response

    with pytest.raises(LLMProviderError, match="could not be processed"):
        ollama_provider(client).generate("Question?")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx.TimeoutException("timed out"), "timed out"),
        (httpx.ConnectError("offline", request=httpx.Request("POST", "http://ollama.test/api/chat")), "unavailable"),
    ],
)
def test_ollama_provider_handles_transport_failures(
    failure: Exception, message: str,
) -> None:
    client = Mock(spec=httpx.Client)
    client.post.side_effect = failure

    with pytest.raises(LLMProviderError, match=message):
        ollama_provider(client).generate("Question?")


def test_ollama_provider_handles_http_errors() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.status_code = 500
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=httpx.Request("POST", "http://ollama.test/api/chat"), response=response
    )
    client.post.return_value = response

    with pytest.raises(LLMProviderError, match="provider returned an error"):
        ollama_provider(client).generate("Question?")


def test_provider_factory_selects_ollama_provider_without_contacting_ollama() -> None:
    provider = create_llm_provider(Settings(
        llm_provider="ollama", ollama_base_url="http://ollama.test", ollama_model="custom-model",
    ))

    assert isinstance(provider, OllamaLLMProvider)
    assert provider._model == "custom-model"
    provider.close()


def test_mocked_ollama_response_flows_through_claims_retrieval_and_verification() -> None:
    client = Mock(spec=httpx.Client)
    response = Mock()
    response.json.return_value = {
        "model": "test-ollama-model",
        "message": {"content": "ISRO launched Chandrayaan-3 in 2023."},
    }
    client.post.return_value = response
    provider = ollama_provider(client)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    extractor = ClaimExtractor(nlp=nlp)

    class StubRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
            return [Evidence(
                evidence_id="E1", claim_or_query=query,
                text="ISRO launched Chandrayaan-3 in 2023.", source_title="Test source",
                source_url="https://example.test/source", relevance_score=0.9,
            )]

    llm_response = provider.generate("Tell me a fact.")
    claims = extractor.extract(llm_response.response_text)
    results = VerificationService(StubRetriever(), RuleBasedVerifier()).verify_claims(claims)

    assert llm_response.question == "Tell me a fact."
    assert len(claims) == 1
    assert len(results) == 1
    assert results[0].claim_text == claims[0].claim_text
    assert results[0].evidence[0].source_url == "https://example.test/source"
    assert results[0].verdict


def test_analyze_works_with_development_provider_without_api_key() -> None:
    extractor = Mock()
    extractor.extract.return_value = []
    components = AnalysisComponents(
        provider=create_llm_provider(Settings(llm_provider="development", llm_api_key=None)),
        extractor=extractor,
        retriever=Mock(),
        verifier=VerificationService(Mock(), RuleBasedVerifier()),
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
    assert response.json()["llm_response"]["response_text"] == "The capital of France is Paris."
