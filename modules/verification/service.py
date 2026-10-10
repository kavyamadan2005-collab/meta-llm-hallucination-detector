"""Batch retrieval and verification orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

from modules.claims.models import Claim
from modules.retrieval.interfaces import Retriever
from modules.retrieval.models import Evidence
from modules.verification.interfaces import EvidenceVerifier
from modules.verification.models import (
    AttributeVerification,
    VerificationAssessment,
    VerificationResult,
    Verdict,
)


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
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_DATE_ATTRIBUTE_PATTERN = re.compile(
    rf"(?P<prefix>\b(?:on|in|at)\s+)?"
    rf"(?P<date>\b(?:{'|'.join(_MONTHS)})\s+\d{{1,2}}(?:,?\s+\d{{4}})?)\b",
    re.IGNORECASE,
)


class VerificationService:
    """Keep evidence retrieval separate from the verifier implementation."""

    def __init__(self, retriever: Retriever, verifier: EvidenceVerifier) -> None:
        self.retriever = retriever
        self.verifier = verifier

    def verify_claim(self, claim: Claim, top_k: int | None = None) -> VerificationResult:
        """Retrieve evidence for and verify one claim."""
        return self._verify_with_retrieved_evidence(claim, None, top_k)

    def verify_claims(self, claims: Sequence[Claim], top_k: int | None = None) -> list[VerificationResult]:
        """Retrieve and verify claims in input order."""
        results: list[VerificationResult] = []
        previous_claim: Claim | None = None
        for claim in claims:
            context_entity = self._referring_context(claim, previous_claim)
            results.append(self._verify_with_retrieved_evidence(claim, context_entity, top_k))
            previous_claim = claim
        return results

    def _verify_with_retrieved_evidence(
        self,
        claim: Claim,
        context_entity: str | None,
        top_k: int | None,
    ) -> VerificationResult:
        """Verify the complete claim and, when dated, its event separately."""
        event_claim, date_value = self._without_explicit_date(claim)
        queries = [self._build_retrieval_query(claim, context_entity)]
        if event_claim is not None:
            event_query = self._build_retrieval_query(event_claim, context_entity)
            if event_query not in queries:
                queries.append(event_query)

        evidence = self._combine_evidence(
            item
            for query in queries
            for item in self.retriever.retrieve(query, top_k)
        )
        result = self.verifier.verify(claim, evidence)
        if event_claim is None:
            return result

        event_result = self.verifier.verify(event_claim, evidence)
        event_assessment = VerificationAssessment(
            verdict=event_result.verdict,
            confidence=event_result.confidence,
            explanation=event_result.explanation,
            supporting_evidence=event_result.supporting_evidence,
            contradicting_evidence=event_result.contradicting_evidence,
        )
        if result.verdict == Verdict.SUPPORTED:
            attribute_verdict = Verdict.SUPPORTED
            attribute_explanation = "The retrieved evidence supports the complete claim, including this date."
        elif result.verdict == Verdict.CONTRADICTED and event_result.verdict == Verdict.SUPPORTED:
            attribute_verdict = Verdict.CONTRADICTED
            attribute_explanation = (
                "The event is supported, but the retrieved evidence conflicts with "
                "the date stated in the claim."
            )
        else:
            attribute_verdict = Verdict.INSUFFICIENT_EVIDENCE
            attribute_explanation = (
                "The retrieved evidence does not establish whether this date is correct."
            )
        attribute_assessment = AttributeVerification(
            name="date",
            value=date_value or "",
            verdict=attribute_verdict,
            confidence=result.confidence,
            explanation=attribute_explanation,
            supporting_evidence=(
                result.supporting_evidence if attribute_verdict == Verdict.SUPPORTED else []
            ),
            contradicting_evidence=(
                result.contradicting_evidence if attribute_verdict == Verdict.CONTRADICTED else []
            ),
        )
        return result.model_copy(
            update={
                "event_fact": event_assessment,
                "attributes": [attribute_assessment],
            }
        )

    @staticmethod
    def _without_explicit_date(claim: Claim) -> tuple[Claim | None, str | None]:
        """Remove a calendar-date phrase to form a date-neutral event claim."""
        match = _DATE_ATTRIBUTE_PATTERN.search(claim.claim_text)
        if match is None:
            return None, None
        start = match.start("prefix") if match.group("prefix") else match.start("date")
        event_text = (
            claim.claim_text[:start] + claim.claim_text[match.end("date"):]
        )
        event_text = re.sub(r"\s+([.,;!?])", r"\1", event_text)
        event_text = re.sub(r"[ \t]+", " ", event_text).strip(" \t.,;")
        if not event_text:
            return None, None
        event_claim = claim.model_copy(
            update={
                "claim_text": event_text,
                "entities": [
                    entity for entity in claim.entities
                    if entity.label.casefold() not in _NON_CONTEXT_LABELS
                ],
            }
        )
        return event_claim, match.group("date")

    @staticmethod
    def _combine_evidence(items: Iterable[Evidence]) -> list[Evidence]:
        """Merge repeated retrieval results while retaining source provenance."""
        deduplicated: dict[tuple[str, str], Evidence] = {}
        for item in items:
            normalized_text = re.sub(r"\s+", " ", item.text).strip().casefold()
            source = item.source_url.casefold() or item.evidence_id.casefold()
            key = (source, normalized_text)
            existing = deduplicated.get(key)
            if existing is None or item.relevance_score > existing.relevance_score:
                deduplicated[key] = item
        return list(deduplicated.values())

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