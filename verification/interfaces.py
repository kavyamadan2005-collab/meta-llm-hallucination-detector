"""Replaceable claim verification interfaces."""

from __future__ import annotations

from typing import Protocol, Sequence

from modules.claims.models import Claim
from modules.retrieval.models import Evidence
from modules.verification.models import VerificationResult


class EvidenceVerifier(Protocol):
    """Interface for rule-based, NLI, or future LLM verifiers."""

    def verify(self, claim: Claim, evidence: Sequence[Evidence]) -> VerificationResult:
        """Evaluate a claim against retrieved evidence."""