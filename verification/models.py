"""Structured outputs for claim verification."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from modules.retrieval.models import Evidence


class Verdict(str, Enum):
    """Baseline verification outcomes."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "INSUFFICIENT_EVIDENCE"


class EvidenceDiagnostic(BaseModel):
    """Per-evidence details from the existing deterministic verifier checks."""

    evidence_id: str
    relevance_score: float
    accepted_by_threshold: bool
    relation: Literal["support", "contradict"] | None = None
    lexical_overlap: float | None = None
    predicate_overlap: bool | None = None
    entity_coverage: bool | None = None
    date_year_check: str | None = None
    negation_polarity_check: str | None = None
    decision_reason: str


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
    evidence_diagnostics: list[EvidenceDiagnostic] = Field(default_factory=list)