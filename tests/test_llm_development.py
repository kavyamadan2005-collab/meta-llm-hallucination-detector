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


def test_development_provider_answers_chandrayaan_question() -> None:
    result = DevelopmentLLMProvider().generate("When did Chandrayaan-3 land on the Moon?")

    assert result.response_text == "Chandrayaan-3 landed on the Moon in 2023."


def test_development_provider_answers_complex_chandrayaan_question() -> None:
    result = DevelopmentLLMProvider().generate(
        "When did Chandrayaan-3 land on the Moon and which country succeeded in this mission?"
    )

    assert "August 23, 2023" in result.response_text
    assert "India" in result.response_text
    assert "ISRO" in result.response_text
    assert "first country to soft-land near the lunar south pole" in result.response_text


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
                        "response concise, preferably 2 to 4 sentences."
                    ),
                },
                {"role": "user", "content": "Question?"},
            ],
            "stream": False,
        },
    )


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
