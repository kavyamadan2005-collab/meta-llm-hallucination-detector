"""Batch retrieval and verification orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from modules.claims.models import Claim
from modules.retrieval.interfaces import Retriever
from modules.verification.interfaces import EvidenceVerifier
from modules.verification.models import VerificationResult


class VerificationService:
    """Keep evidence retrieval separate from the verifier implementation."""

    def __init__(self, retriever: Retriever, verifier: EvidenceVerifier) -> None:
        self.retriever = retriever
        self.verifier = verifier

    def verify_claim(self, claim: Claim, top_k: int | None = None) -> VerificationResult:
        """Retrieve evidence for and verify one claim."""
        evidence = self.retriever.retrieve(claim.claim_text, top_k)
        return self.verifier.verify(claim, evidence)

    def verify_claims(self, claims: Sequence[Claim], top_k: int | None = None) -> list[VerificationResult]:
        """Retrieve and verify claims in input order."""
        return [self.verify_claim(claim, top_k) for claim in claims]