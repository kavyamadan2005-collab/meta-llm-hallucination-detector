"""API request and response models for end-to-end analysis."""

from typing import Any, Literal

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
    overall_result: "OverallResult"


class OverallResult(BaseModel):
    """Aggregate outcome derived only from individual claim verdicts."""

    status: Literal["VERIFIED", "HALLUCINATION_DETECTED", "INCONCLUSIVE"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)