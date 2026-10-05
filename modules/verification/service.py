"""Batch retrieval and verification orchestration."""

from __future__ import annotations

from collections.abc import Sequence
import re

from modules.claims.models import Claim
from modules.retrieval.interfaces import Retriever
from modules.verification.interfaces import EvidenceVerifier
from modules.verification.models import VerificationResult


_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "on", "or", "that", "the", "this", "to",
    "was", "were",
}
_REFERRING_PREFIX = re.compile(
    r"^\s*(?:(?:the|this)\s+[a-z][a-z-]*(?:['’]s)?\b|"
    r"(?:he|she|they|his|her|their|its)\b)",
    re.IGNORECASE,
)
_NON_CONTEXT_LABELS = {"date", "time", "cardinal", "ordinal"}
_NUMERIC_ENTITY = re.compile(r"^\s*[\d\s.,:/-]+\s*$")


class VerificationService:
    """Keep evidence retrieval separate from the verifier implementation."""

    def __init__(self, retriever: Retriever, verifier: EvidenceVerifier) -> None:
        self.retriever = retriever
        self.verifier = verifier

    def verify_claim(self, claim: Claim, top_k: int | None = None) -> VerificationResult:
        """Retrieve evidence for and verify one claim."""
        query = self._build_retrieval_query(claim)
        evidence = self.retriever.retrieve(query, top_k)
        return self.verifier.verify(claim, evidence)

    def verify_claims(self, claims: Sequence[Claim], top_k: int | None = None) -> list[VerificationResult]:
        """Retrieve and verify claims in input order."""
        results: list[VerificationResult] = []
        previous_claim: Claim | None = None
        for claim in claims:
            context_entity = self._referring_context(claim, previous_claim)
            query = self._build_retrieval_query(claim, context_entity)
            evidence = self.retriever.retrieve(query, top_k)
            results.append(self.verifier.verify(claim, evidence))
            previous_claim = claim
        return results

    @staticmethod
    def _build_retrieval_query(claim: Claim, context_entity: str | None = None) -> str:
        """Build a compact query while preserving claim-specific facts."""
        query_parts: list[str] = []
        seen_tokens: set[str] = set()

        entity_texts = [context_entity] if context_entity else []
        entity_texts.extend(entity.text for entity in claim.entities)
        for entity_text in entity_texts:
            entity_text = entity_text.strip()
            if not entity_text:
                continue
            query_parts.append(entity_text)
            seen_tokens.update(
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", entity_text)
            )

        for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", claim.claim_text):
            normalized = token.casefold()
            if normalized in _QUERY_STOP_WORDS or normalized in seen_tokens:
                continue
            query_parts.append(token)
            seen_tokens.add(normalized)

        focused_query = " ".join(query_parts).strip()
        return focused_query if len(query_parts) >= 2 else claim.claim_text

    @staticmethod
    def _referring_context(
        claim: Claim,
        previous_claim: Claim | None,
    ) -> str | None:
        """Return one prior explicit entity for a syntactic referring claim."""
        if previous_claim is None or not _REFERRING_PREFIX.match(claim.claim_text):
            return None
        for entity in previous_claim.entities:
            entity_text = entity.text.strip()
            if (
                entity_text
                and entity.label.casefold() not in _NON_CONTEXT_LABELS
                and not _NUMERIC_ENTITY.fullmatch(entity_text)
            ):
                return entity_text
        return None