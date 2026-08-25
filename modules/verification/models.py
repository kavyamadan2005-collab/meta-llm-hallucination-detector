"""Structured outputs for claim verification."""

from enum import Enum

from pydantic import BaseModel, Field

from modules.retrieval.models import Evidence


class Verdict(str, Enum):
    """Baseline verification outcomes."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class VerificationResult(BaseModel):
    """A deterministic, evidence-preserving claim verification result."""

    claim_id: str
    claim_text: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)