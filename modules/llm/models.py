"""Models shared by LLM providers."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """A provider response retained for downstream analysis and auditing."""

    question: str
    response_text: str = Field(min_length=1)
    model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))