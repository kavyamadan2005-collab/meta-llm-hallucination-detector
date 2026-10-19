
"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Validated settings for the application and its AI analysis layer."""

    app_name: str = Field(default="Meta-LLM Hallucination Detection")
    environment: str = Field(default="development")
    database_path: str = Field(default="database/app.db")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8501",
        ]
    )

    log_level: str = Field(default="INFO")
    llm_provider: Literal[
        "development", "ollama", "openai_compatible"
    ] = Field(default="development")
    llm_api_key: str | None = Field(default=None)
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    ollama_timeout_seconds: float = Field(default=300.0, gt=0)

    retrieval_enabled_sources: list[str] = Field(
        default_factory=lambda: ["curated", "local", "wikipedia", "web_search"]
    )

    wikipedia_api_url: str = Field(
        default="https://en.wikipedia.org/w/api.php"
    )
    wikipedia_user_agent: str = Field(
        default="Meta-LLM-Hallucination-Detection/0.1 (local development)"
    )
    wikipedia_timeout_seconds: float = Field(default=10.0, gt=0)
    wikipedia_top_k: int = Field(default=5, gt=0)

    openalex_api_url: str = Field(
        default="https://api.openalex.org/works"
    )
    openalex_api_key: str | None = Field(default=None)
    openalex_timeout_seconds: float = Field(default=10.0, gt=0)
    openalex_top_k: int = Field(default=5, gt=0)

    web_search_enabled: bool = Field(default=True)
    web_search_provider: str = Field(default="development")
    tavily_api_key: str | None = Field(default=None)
    tavily_api_url: str = Field(
        default="https://api.tavily.com/search"
    )
    tavily_timeout_seconds: float = Field(default=10.0, gt=0)
    tavily_top_k: int = Field(default=5, gt=0)

    retrieval_index_path: str = Field(default="data/index")
    retrieval_chunk_size: int = Field(default=500, gt=0)
    retrieval_chunk_overlap: int = Field(default=50, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_min_similarity: float = Field(
        default=0.65, ge=-1.0, le=1.0
    )

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build settings from .env and process environment values."""

        load_dotenv()

        origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501",
        )
        enabled_sources = os.getenv(
            "RETRIEVAL_ENABLED_SOURCES",
            "curated,local,wikipedia,web_search",
        )

        parsed_origins = [
            origin.strip()
            for origin in origins.split(",")
            if origin.strip()
        ]

        for required_origin in (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ):
            if required_origin not in parsed_origins:
                parsed_origins.append(required_origin)

        return cls(
            app_name=os.getenv(
                "APP_NAME", "Meta-LLM Hallucination Detection"
            ),
            environment=os.getenv("ENVIRONMENT", "development"),
            database_path=os.getenv(
                "DATABASE_PATH", "database/app.db"
            ),
            cors_origins=parsed_origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            llm_provider=os.getenv("LLM_PROVIDER", "development"),
            llm_api_key=(
                os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            ),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_timeout_seconds=float(
                os.getenv("LLM_TIMEOUT_SECONDS", "30")
            ),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            ollama_timeout_seconds=float(
                os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")
            ),
            retrieval_enabled_sources=[
                source.strip().casefold()
                for source in enabled_sources.split(",")
                if source.strip()
            ],
            wikipedia_api_url=os.getenv(
                "WIKIPEDIA_API_URL",
                "https://en.wikipedia.org/w/api.php",
            ).rstrip("/"),
            wikipedia_user_agent=os.getenv(
                "WIKIPEDIA_USER_AGENT",
                "Meta-LLM-Hallucination-Detection/0.1 (local development)",
            ),
            wikipedia_timeout_seconds=float(
                os.getenv("WIKIPEDIA_TIMEOUT_SECONDS", "10")
            ),
            wikipedia_top_k=int(
                os.getenv("WIKIPEDIA_TOP_K", "5")
            ),
            openalex_api_url=os.getenv(
                "OPENALEX_API_URL",
                "https://api.openalex.org/works",
            ).rstrip("/"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
            openalex_timeout_seconds=float(
                os.getenv("OPENALEX_TIMEOUT_SECONDS", "10")
            ),
            openalex_top_k=int(
                os.getenv("OPENALEX_TOP_K", "5")
            ),
            web_search_enabled=os.getenv(
                "WEB_SEARCH_ENABLED", "true"
            ).casefold() in {"1", "true", "yes", "on"},
            web_search_provider=os.getenv(
                "WEB_SEARCH_PROVIDER", "development"
            ).casefold(),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            tavily_api_url=os.getenv(
                "TAVILY_API_URL",
                "https://api.tavily.com/search",
            ).rstrip("/"),
            tavily_timeout_seconds=float(
                os.getenv("TAVILY_TIMEOUT_SECONDS", "10")
            ),
            tavily_top_k=int(
                os.getenv("TAVILY_TOP_K", "5")
            ),
            retrieval_index_path=os.getenv(
                "RETRIEVAL_INDEX_PATH", "data/index"
            ),
            retrieval_chunk_size=int(
                os.getenv("RETRIEVAL_CHUNK_SIZE", "500")
            ),
            retrieval_chunk_overlap=int(
                os.getenv("RETRIEVAL_CHUNK_OVERLAP", "50")
            ),
            retrieval_top_k=int(
                os.getenv("RETRIEVAL_TOP_K", "5")
            ),
            retrieval_min_similarity=float(
                os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.65")
            ),
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings.from_environment()
