"""API request and response models for end-to-end analysis."""

from typing import Any

from pydantic import BaseModel, Field

from modules.claims.models import Claim
from modules.llm.models import LLMResponse
from modules.verification.models import VerificationResult


class AnalyzeRequest(BaseModel):
    """Question submitted for analysis."""

    question: str = Field(...)


class AnalyzeResponse(BaseModel):
    """Complete output of the current analysis pipeline."""

    question: str
    llm_response: LLMResponse
    claims: list[Claim]
    graph: dict[str, Any]
    verification_results: list[VerificationResult]