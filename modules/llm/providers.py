"""Replaceable provider interfaces and an OpenAI-compatible implementation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

import httpx

from backend.core.config import Settings, get_settings
from modules.llm.models import LLMResponse

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot produce a response."""


class LLMProvider(Protocol):
    """Interface implemented by all response-generation providers."""

    def generate(self, question: str) -> LLMResponse:
        """Generate a response for a user question."""


class DevelopmentLLMProvider:
    """Return deterministic responses for local development and testing."""

    _RESPONSES = {
        "what is the capital of france?": "The capital of France is Paris.",
        "when did chandrayaan-3 land on the moon?": "Chandrayaan-3 landed on the Moon in 2023.",
        "when did chandrayaan-3 land on the moon and which country succeeded in this mission?": (
            "Chandrayaan-3 successfully soft-landed on the Moon on August 23, 2023. "
            "India, through ISRO, successfully carried out the mission, becoming the "
            "first country to soft-land near the lunar south pole."
        ),
    }

    def generate(self, question: str) -> LLMResponse:
        """Generate a predefined response without contacting an external service."""
        if not question.strip():
            raise ValueError("question must not be empty")
        response_text = self._RESPONSES.get(
            question.strip().casefold(),
            "[Development mode] No predefined response is available for this question.",
        )
        return LLMResponse(
            question=question,
            response_text=response_text,
            model="development",
            timestamp=datetime.now(timezone.utc),
        )


class OpenAICompatibleProvider:
    """Call a chat-completions endpoint without coupling callers to its SDK."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or httpx.Client(timeout=self._settings.llm_timeout_seconds)

    def generate(self, question: str) -> LLMResponse:
        """Generate one response, translating transport/API failures clearly."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if not self._settings.llm_api_key:
            raise LLMProviderError("LLM API key is not configured")

        try:
            response = self._client.post(
                f"{self._settings.llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._settings.llm_api_key}"},
                json={
                    "model": self._settings.llm_model,
                    "messages": [{"role": "user", "content": question}],
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            model = payload.get("model", self._settings.llm_model)
            if not isinstance(text, str) or not text.strip():
                raise LLMProviderError("LLM response did not contain response text")
            return LLMResponse(
                question=question,
                response_text=text,
                model=model,
                timestamp=datetime.now(timezone.utc),
            )
        except httpx.TimeoutException as exc:
            logger.warning("LLM request timed out")
            raise LLMProviderError("LLM request timed out") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning("LLM request failed with status %s", exc.response.status_code)
            raise LLMProviderError("LLM provider returned an error") from exc
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("LLM response could not be processed: %s", type(exc).__name__)
            raise LLMProviderError("LLM response could not be processed") from exc

    def close(self) -> None:
        """Close the internally owned HTTP client when used as a long-lived service."""
        self._client.close()


class OllamaLLMProvider:
    """Call Ollama's local chat API without requiring a hosted LLM key."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
        settings: Settings | None = None,
    ) -> None:
        configured = settings or get_settings()
        self._base_url = (base_url or configured.ollama_base_url).rstrip("/")
        self._model = model or configured.ollama_model
        self._timeout = timeout or configured.ollama_timeout_seconds
        self._client = client or httpx.Client(timeout=self._timeout)

    def generate(self, question: str) -> LLMResponse:
        """Generate one response and translate local API failures safely."""
        if not question.strip():
            raise ValueError("question must not be empty")

        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer the user's question concisely in 2 to 4 sentences. "
                                "Give only the information needed to answer the question. "
                                "Avoid unnecessary explanations, repetition, lists, speculation, "
                                "and unrelated facts."
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["message"]["content"]
            response_model = payload.get("model", self._model)
            if not isinstance(text, str) or not text.strip():
                raise LLMProviderError("LLM response did not contain response text")
            if not isinstance(response_model, str) or not response_model.strip():
                raise LLMProviderError("LLM response did not contain a model")
            return LLMResponse(
                question=question,
                response_text=text,
                model=response_model,
                timestamp=datetime.now(timezone.utc),
            )
        except httpx.TimeoutException as exc:
            logger.warning("Ollama request timed out")
            raise LLMProviderError("LLM request timed out") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning("Ollama request failed with status %s", exc.response.status_code)
            raise LLMProviderError("LLM provider returned an error") from exc
        except httpx.ConnectError as exc:
            logger.warning("Ollama is unavailable")
            raise LLMProviderError("LLM provider is unavailable") from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ollama response could not be processed: %s", type(exc).__name__)
            raise LLMProviderError("LLM response could not be processed") from exc

    def close(self) -> None:
        """Close the internally owned HTTP client when used as a long-lived service."""
        self._client.close()


def create_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Construct the configured response-generation provider."""
    configured = settings or get_settings()
    if configured.llm_provider == "development":
        return DevelopmentLLMProvider()
    if configured.llm_provider == "openai_compatible":
        return OpenAICompatibleProvider(configured)
    if configured.llm_provider == "ollama":
        return OllamaLLMProvider(settings=configured)
    raise ValueError(f"Unsupported LLM provider: {configured.llm_provider}")