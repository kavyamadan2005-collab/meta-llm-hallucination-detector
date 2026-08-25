"""Conservative deterministic evidence-based verification baseline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from modules.claims.models import Claim
from modules.retrieval.models import Evidence
from modules.verification.models import Verdict, VerificationResult

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
}
_NEGATION_WORDS = {"not", "no", "never", "neither", "isn't", "wasn't", "aren't", "weren't"}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_PREDICATES = {
    "are", "aren't", "be", "built", "created", "developed", "founded", "has",
    "have", "is", "isn't", "launched", "located", "made", "was", "wasn't", "were",
}


class RuleBasedVerifier:
    """Verify only when relevance and lexical support are sufficiently strong.

    Confidence is an evidence-quality score, not a probability that a claim is true.
    It combines the best accepted retrieval score with lexical coverage, capped below
    certainty because this baseline has no source authority or entailment model.
    """

    def __init__(self, min_similarity: float = 0.65, min_token_overlap: float = 0.5) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")
        if not 0.0 <= min_token_overlap <= 1.0:
            raise ValueError("min_token_overlap must be between 0 and 1")
        self.min_similarity = min_similarity
        self.min_token_overlap = min_token_overlap

    def verify(self, claim: Claim, evidence: Sequence[Evidence]) -> VerificationResult:
        """Return a verdict without treating retrieval alone as proof."""
        accepted = [item for item in evidence if item.relevance_score >= self.min_similarity]
        supporting: list[Evidence] = []
        contradicting: list[Evidence] = []
        for item in accepted:
            relation = self._relation(claim.claim_text, item.text)
            if relation == "support":
                supporting.append(item)
            elif relation == "contradict":
                contradicting.append(item)

        if supporting and not contradicting:
            verdict = Verdict.SUPPORTED
            strongest = max(supporting, key=lambda item: item.relevance_score)
            explanation = f"The retrieved evidence directly supports the claim (source: {strongest.source_title})."
            confidence = self._confidence(strongest, claim.claim_text)
        elif contradicting and not supporting:
            verdict = Verdict.CONTRADICTED
            strongest = max(contradicting, key=lambda item: item.relevance_score)
            explanation = f"The retrieved evidence conflicts with the claim (source: {strongest.source_title})."
            confidence = self._confidence(strongest, claim.claim_text)
        else:
            verdict = Verdict.UNSUPPORTED
            explanation = (
                "No sufficiently relevant evidence was retrieved to verify this claim."
                if not accepted
                else "The retrieved evidence is ambiguous or insufficient to verify this claim."
            )
            confidence = 0.0 if not accepted else min(
                self._confidence(item, claim.claim_text) for item in accepted
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            evidence=list(evidence),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
        )

    def _relation(self, claim_text: str, evidence_text: str) -> str | None:
        claim_tokens = self._tokens(claim_text)
        evidence_tokens = self._tokens(evidence_text)
        if not claim_tokens or not evidence_tokens:
            return None
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        if overlap < self.min_token_overlap:
            return None
        claim_predicates = claim_tokens & _PREDICATES
        if claim_predicates and not claim_predicates.intersection(evidence_tokens):
            return None
        claim_numbers = set(re.findall(r"\b\d{4}\b", claim_text))
        evidence_numbers = set(re.findall(r"\b\d{4}\b", evidence_text))
        if claim_numbers and evidence_numbers and claim_numbers.isdisjoint(evidence_numbers):
            return "contradict"
        claim_negated = bool(self._tokens(claim_text) & _NEGATION_WORDS)
        evidence_negated = bool(self._tokens(evidence_text) & _NEGATION_WORDS)
        if claim_negated != evidence_negated:
            return "contradict"
        return "support"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token not in _STOP_WORDS}

    def _confidence(self, evidence: Evidence, claim_text: str) -> float:
        overlap = len(self._tokens(claim_text) & self._tokens(evidence.text)) / max(1, len(self._tokens(claim_text)))
        return round(min(0.95, max(0.0, float(evidence.relevance_score)) * (0.5 + 0.5 * overlap)), 3)