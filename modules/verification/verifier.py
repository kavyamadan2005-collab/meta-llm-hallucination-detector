"""Conservative deterministic evidence-based verification baseline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from modules.claims.models import Claim
from modules.retrieval.models import Evidence
from modules.verification.models import Verdict, VerificationResult


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
}

_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "isn't",
    "wasn't",
    "aren't",
    "weren't",
    "cannot",
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_PREDICATES = {
    "are",
    "aren't",
    "be",
    "built",
    "created",
    "developed",
    "founded",
    "has",
    "have",
    "is",
    "isn't",
    "launch",
    "land",
    "launched",
    "located",
    "locat",
    "made",
    "make",
    "build",
    "create",
    "develop",
    "found",
    "was",
    "wasn't",
    "were",
}

_NEGATION_CUES = _NEGATION_WORDS | {"can", "did", "do", "does"}
_IRREGULAR_FORMS = {
    "won": "win",
    "went": "go",
    "was": "be",
    "were": "be",
}
_SAFE_RELATION_ALIASES = {
    "purchased": "acquire",
    "purchase": "acquire",
    "discovered": "found",
    "discover": "find",
    "won": "win",
    "champions": "win",
    "ended": "end",
    "brought": "bring",
    "uses": "use",
    "used": "use",
    "adopted": "adopt",
    "rejected": "reject",
    "led": "lead",
    "acquired": "acquire",
    "founded": "found",
    "developed": "develop",
    "mentioned": "mention",
    "contains": "contain",
    "entered": "enter",
    "increased": "increase",
    "increase": "increase",
    "grew": "increase",
    "grow": "increase",
    "sorts": "sort",
    "sort": "sort",
    "orders": "sort",
    "order": "sort",
}
_RELATION_WORDS = set(_SAFE_RELATION_ALIASES.values()) | {
    "acquire",
    "found",
    "win",
    "end",
    "use",
    "adopt",
    "reject",
    "lead",
    "develop",
    "mention",
    "contain",
    "produce",
    "open",
    "operate",
    "approve",
    "sign",
    "launch",
    "land",
    "orbit",
    "is",
    "has",
    "enter",
    "increase",
    "sort",
}
_SAFE_PARAPHRASE_PREDICATES = {
    frozenset({"increase", "grow"}),
    frozenset({"sort", "order"}),
}
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_LANDING_TERMS = {"land", "landed", "landing", "soft-land", "soft-landed", "touch", "touched"}


class RuleBasedVerifier:
    """Verify claims using deterministic evidence-based rules.

    Confidence is an evidence-quality score, not a probability that
    a claim is true.

    The verifier combines:
    - retrieval relevance
    - lexical overlap
    - predicate overlap
    - year consistency
    - simple direct-negation detection

    This is a baseline verifier and does not perform full semantic
    entailment or NLI.
    """

    def __init__(
        self,
        min_similarity: float = 0.65,
        min_token_overlap: float = 0.5,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")

        if not 0.0 <= min_token_overlap <= 1.0:
            raise ValueError("min_token_overlap must be between 0 and 1")

        self.min_similarity = min_similarity
        self.min_token_overlap = min_token_overlap

    def verify(
        self,
        claim: Claim,
        evidence: Sequence[Evidence],
    ) -> VerificationResult:
        """Return a verdict without treating retrieval alone as proof."""

        accepted = [
            item
            for item in evidence
            if item.relevance_score >= self.min_similarity
        ]

        supporting: list[Evidence] = []
        contradicting: list[Evidence] = []

        for item in accepted:
            relation = self._relation(
                claim,
                item.text,
            )

            if relation == "support":
                supporting.append(item)

            elif relation == "contradict":
                contradicting.append(item)

        if supporting and not contradicting:
            verdict = Verdict.SUPPORTED

            strongest = max(
                supporting,
                key=lambda item: item.relevance_score,
            )

            explanation = (
                "The retrieved evidence directly supports the claim "
                f"(source: {strongest.source_title})."
            )

            confidence = self._confidence(
                strongest,
                claim.claim_text,
            )

        elif contradicting and not supporting:
            verdict = Verdict.CONTRADICTED

            strongest = max(
                contradicting,
                key=lambda item: item.relevance_score,
            )

            explanation = (
                "The retrieved evidence conflicts with the claim "
                f"(source: {strongest.source_title})."
            )

            confidence = self._confidence(
                strongest,
                claim.claim_text,
            )

        else:
            verdict = Verdict.UNSUPPORTED

            explanation = (
                "No sufficiently relevant evidence was retrieved "
                "to verify this claim."
                if not accepted
                else
                "The retrieved evidence is ambiguous or insufficient "
                "to verify this claim."
            )

            confidence = (
                0.0
                if not accepted
                else min(
                    self._confidence(
                        item,
                        claim.claim_text,
                    )
                    for item in accepted
                )
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

    def _relation(
        self,
        claim: Claim,
        evidence_text: str,
    ) -> str | None:
        """Determine whether evidence supports or contradicts a claim."""

        claim_text = claim.claim_text
        claim_tokens = self._tokens(claim_text)
        evidence_tokens = self._tokens(evidence_text)

        if not claim_tokens or not evidence_tokens:
            return None

        claim_content = self._comparison_tokens(claim_text)
        evidence_content = self._comparison_tokens(evidence_text)

        if self._has_conflicting_explicit_date(claim, evidence_text):
            return "contradict"

        claim_relation = self._relation_signature(claim_text)
        evidence_relation = self._relation_signature(evidence_text)
        aligned_relation = False
        if self._has_partial_compound_support(claim_text, evidence_text):
            return None
        if self._has_reversed_founded_relation(claim_text, evidence_text):
            return None
        if claim_relation and evidence_relation:
            if self._polarity(claim_text) != self._polarity(evidence_text):
                return "contradict"

        if self._is_safe_paraphrase_match(
            claim_text,
            evidence_text,
            claim_relation,
            evidence_relation,
        ):
            return "support"
        if claim_relation and evidence_relation:
            if self._has_conflicting_numeric_argument(
                claim_text,
                evidence_text,
                claim_relation,
                evidence_relation,
            ):
                return "contradict"
            aligned_relation = self._has_aligned_relation(
                claim_text,
                evidence_text,
                claim_relation,
                evidence_relation,
            )
            if (
                self._relation_conflicts(claim_relation, evidence_relation)
                and not aligned_relation
            ):
                return None

        # Measure how much of the claim is covered by the evidence.
        overlap = (
            len(claim_content & evidence_content)
            / len(claim_content)
        )

        if overlap < self.min_token_overlap:
            return None

        # If the claim contains a meaningful predicate,
        # require the evidence to contain that predicate too.
        claim_predicates = {
            self._canonical_token(token)
            for token in claim_tokens
        } & _PREDICATES

        if (
            claim_predicates
            and not claim_predicates.intersection(
                self._canonical_token(token)
                for token in evidence_tokens
            )
            and not aligned_relation
        ):
            return None

        if not self._entities_are_covered(claim, evidence_text):
            return None

        required_date_tokens = self._specific_date_tokens(claim_text)
        if required_date_tokens and not required_date_tokens.issubset(
            self._comparison_tokens(evidence_text)
        ):
            return None

        # If both texts contain years but the years disagree,
        # treat the evidence as contradictory.
        claim_numbers = set(
            re.findall(r"\b\d{4}\b", claim_text)
        )

        evidence_numbers = set(
            re.findall(r"\b\d{4}\b", evidence_text)
        )

        if (
            claim_numbers
            and evidence_numbers
            and claim_numbers.isdisjoint(evidence_numbers)
        ):
            return "contradict"

        # IMPORTANT:
        # Do not simply check whether the word "not" appears
        # anywhere in a long evidence document.
        #
        # Instead, detect simple direct negation patterns.
        claim_negated = self._negated_predicates(claim_text)
        evidence_negated = self._negated_predicates(evidence_text)

        if claim_negated and not claim_negated.intersection(evidence_content):
            return None
        if evidence_negated and not evidence_negated.intersection(claim_content):
            return None

        # A positive claim vs directly negated evidence is contradictory.
        if (
            not claim_negated
            and evidence_negated.intersection(evidence_content)
            and evidence_negated.intersection(claim_content)
        ):
            return "contradict"

        # A negated claim vs positive evidence is contradictory.
        if (
            claim_negated.intersection(claim_content)
            and not evidence_negated
            and claim_negated.intersection(evidence_content)
        ):
            return "contradict"

        return "support"

    def _entities_are_covered(self, claim: Claim, evidence_text: str) -> bool:
        """Require extracted claim entities to appear in supporting evidence."""
        evidence_tokens = self._comparison_tokens(evidence_text)
        return all(
            self._comparison_tokens(entity.text).issubset(evidence_tokens)
            for entity in claim.entities
        )

    @staticmethod
    def _has_reversed_founded_relation(
        claim_text: str,
        evidence_text: str,
    ) -> bool:
        """Reject the active/passive subject-object reversal for founding."""
        claim_match = re.search(
            r"\b([a-z0-9][a-z0-9-]*)\s+founded\s+([a-z0-9][a-z0-9-]*)\b",
            claim_text.casefold(),
        )
        evidence_match = re.search(
            r"\b([a-z0-9][a-z0-9-]*)\s+was\s+founded\s+by\s+"
            r"([a-z0-9][a-z0-9-]*)\b",
            evidence_text.casefold(),
        )
        return bool(
            claim_match
            and evidence_match
            and claim_match.group(1) == evidence_match.group(1)
            and claim_match.group(2) == evidence_match.group(2)
        )

    @classmethod
    def _relation_signature(
        cls,
        text: str,
    ) -> tuple[frozenset[str], str, frozenset[str]] | None:
        """Extract a small subject/predicate/object signature when unambiguous."""
        text = cls._normalize_safe_phrases(text)
        ordered = [
            _SAFE_RELATION_ALIASES.get(token, cls._canonical_token(token))
            for token in _TOKEN_PATTERN.findall(text.casefold())
            if (
                (token not in _STOP_WORDS or token in _RELATION_WORDS)
                and token not in _NEGATION_CUES
            )
        ]
        for index, token in enumerate(ordered):
            if token not in _RELATION_WORDS:
                continue
            subject = frozenset(ordered[:index])
            object_tokens = frozenset(ordered[index + 1:])
            if not subject or not object_tokens:
                continue
            return subject, token, object_tokens
        return None

    @staticmethod
    def _normalize_safe_phrases(text: str) -> str:
        """Normalize only explicitly supported, non-general paraphrases."""
        normalized = text.casefold()
        normalized = re.sub(r"\bwas placed into orbit\b", "entered orbit", normalized)
        normalized = re.sub(
            r"\bbrought the war to an end\b",
            "ended the conflict",
            normalized,
        )
        normalized = re.sub(r"\bbecame champions\b", "won championship", normalized)
        return normalized

    @classmethod
    def _relation_conflicts(
        cls,
        claim_relation: tuple[frozenset[str], str, frozenset[str]],
        evidence_relation: tuple[frozenset[str], str, frozenset[str]],
    ) -> bool:
        """Reject clear subject, object, or predicate mismatches."""
        claim_subject, claim_predicate, claim_object = claim_relation
        evidence_subject, evidence_predicate, evidence_object = evidence_relation

        if claim_predicate == evidence_predicate:
            if (
                claim_subject == evidence_object
                and claim_object == evidence_subject
            ):
                return True
            return (
                bool(claim_subject & evidence_subject)
                and claim_subject != evidence_subject
            ) or (
                bool(claim_subject & evidence_subject)
                and claim_object != evidence_object
                and {
                    token for token in claim_object if not token.isdigit()
                } != {
                    token for token in evidence_object if not token.isdigit()
                }
            )

        return (
            bool(claim_subject & evidence_subject)
            and bool(claim_object & evidence_object)
            and claim_predicate != evidence_predicate
        )

    def _has_aligned_relation(
        self,
        claim_text: str,
        evidence_text: str,
        claim_relation: tuple[frozenset[str], str, frozenset[str]],
        evidence_relation: tuple[frozenset[str], str, frozenset[str]],
    ) -> bool:
        """Allow the fallback to assess compatible signatures with extra context."""
        claim_subject, claim_predicate, claim_object = claim_relation
        evidence_subject, evidence_predicate, evidence_object = evidence_relation
        if claim_predicate != evidence_predicate:
            return False
        if not claim_subject & evidence_subject or not claim_object & evidence_object:
            return False
        object_overlap = len(claim_object & evidence_object) / len(claim_object)
        if object_overlap < max(self.min_token_overlap, 0.6):
            return False
        claim_content = self._comparison_tokens(claim_text)
        evidence_content = self._comparison_tokens(evidence_text)
        overlap = len(claim_content & evidence_content) / max(1, len(claim_content))
        return overlap >= max(self.min_token_overlap, 0.6)

    @staticmethod
    def _polarity(text: str) -> str:
        """Detect only direct sentence-level negation or modal denial."""
        normalized = text.casefold()
        if re.search(
            r"\b(?:is|are|was|were|has|have|can|does|do|did)\s+"
            r"(?:not|no|never)\b|\bcannot\b|\bnever\s+\w+",
            normalized,
        ):
            return "negative"
        return "positive"

    @classmethod
    def _has_partial_compound_support(
        cls,
        claim_text: str,
        evidence_text: str,
    ) -> bool:
        """Keep a multi-clause claim unsupported when evidence covers one clause."""
        if not re.search(r"\band\b", claim_text.casefold()):
            return False
        normalized = claim_text.casefold()
        conjunction = re.search(r"\band\b", normalized)
        if conjunction is None:
            return False
        prefix = normalized[:conjunction.start()]
        if re.search(r"\b(?:between|in|on)\s+[^.!?;]*$", prefix):
            return False
        if re.search(r"\bfrom\s+[^.!?;]*\bto\s+[^.!?;]*$", prefix):
            return False
        claim_parts = [prefix, normalized[conjunction.end():]]
        if len(claim_parts) != 2:
            return False
        second_part = cls._comparison_tokens(claim_parts[1])
        evidence_tokens = cls._comparison_tokens(evidence_text)
        return bool(second_part) and not second_part.issubset(evidence_tokens)

    @classmethod
    def _has_conflicting_numeric_argument(
        cls,
        claim_text: str,
        evidence_text: str,
        claim_relation: tuple[frozenset[str], str, frozenset[str]],
        evidence_relation: tuple[frozenset[str], str, frozenset[str]],
    ) -> bool:
        """Detect an obvious count conflict without altering year/date handling."""
        if claim_relation[1] != "contain" or evidence_relation[1] != "contain":
            return False
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_text))
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))
        return (
            bool(claim_numbers and evidence_numbers)
            and claim_numbers.isdisjoint(evidence_numbers)
            and claim_relation[0] == evidence_relation[0]
        )

    def _has_conflicting_explicit_date(self, claim: Claim, evidence_text: str) -> bool:
        """Detect different dates for the same explicitly shared event."""
        claim_date = self._explicit_date(claim.claim_text)
        evidence_date = self._explicit_date(evidence_text)
        if claim_date is None or evidence_date is None or claim_date == evidence_date:
            return False

        evidence_tokens = self._comparison_tokens(evidence_text)
        shared_entities = [
            entity.text.strip()
            for entity in claim.entities
            if entity.text.strip()
            and entity.label.casefold() in {"domain_entity", "org", "product", "person", "gpe", "loc"}
            and self._comparison_tokens(entity.text).issubset(evidence_tokens)
        ]
        claim_event_terms = self._event_terms(claim.claim_text)
        evidence_event_terms = self._event_terms(evidence_text)
        if shared_entities and claim_event_terms & evidence_event_terms:
            return True

        claim_context = self._date_free_context(claim.claim_text)
        evidence_context = self._date_free_context(evidence_text)
        shared_context = claim_context & evidence_context
        return (
            len(shared_context) >= 2
            and len(shared_context) / max(1, len(claim_context)) >= 0.5
        )

    @staticmethod
    def _explicit_date(text: str) -> tuple[str, str, str | None] | None:
        """Return the first explicit month/day/year date in text."""
        match = re.search(
            rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}})(?:,?\s+\d{{4}})?\b",
            text.casefold(),
        )
        if not match:
            return None
        year_match = re.search(
            rf"\b{match.group(1)}\s+{match.group(2)},?\s+(\d{{4}})\b",
            text.casefold(),
        )
        return (
            match.group(1),
            match.group(2),
            year_match.group(1) if year_match else None,
        )

    @classmethod
    def _date_free_context(cls, text: str) -> set[str]:
        """Return factual context with complete calendar-date tokens removed."""
        normalized = text.casefold()
        normalized = re.sub(
            rf"\b(?:{'|'.join(_MONTHS)})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b",
            " ",
            normalized,
        )
        return cls._comparison_tokens(normalized) - {
            "occur",
            "happen",
        }

    @classmethod
    def _is_safe_paraphrase_match(
        cls,
        claim_text: str,
        evidence_text: str,
        claim_relation: tuple[frozenset[str], str, frozenset[str]] | None,
        evidence_relation: tuple[frozenset[str], str, frozenset[str]] | None,
    ) -> bool:
        """Support only a known predicate equivalence with a shared object."""
        claim_predicate = cls._safe_paraphrase_predicate(claim_text)
        evidence_predicate = cls._safe_paraphrase_predicate(evidence_text)
        if claim_predicate is None or evidence_predicate is None:
            return False
        if claim_predicate == evidence_predicate:
            return False
        if frozenset({claim_predicate, evidence_predicate}) not in _SAFE_PARAPHRASE_PREDICATES:
            return False

        claim_object = claim_relation[2] if claim_relation else frozenset()
        evidence_object = evidence_relation[2] if evidence_relation else frozenset()
        if claim_object and evidence_object:
            return claim_object == evidence_object

        claim_content = cls._comparison_tokens(claim_text)
        evidence_content = cls._comparison_tokens(evidence_text)
        return bool((claim_content - {claim_predicate}) & (evidence_content - {evidence_predicate}))

    @classmethod
    def _safe_paraphrase_predicate(cls, text: str) -> str | None:
        predicate_forms = {
            "increased": "increase",
            "increase": "increase",
            "grew": "grow",
            "grow": "grow",
            "sorts": "sort",
            "sort": "sort",
            "orders": "order",
            "order": "order",
        }
        predicates = [
            predicate_forms[token]
            for token in _TOKEN_PATTERN.findall(text.casefold())
            if token in predicate_forms
        ]
        return predicates[0] if len(predicates) == 1 else None

    @classmethod
    def _event_terms(cls, text: str) -> set[str]:
        """Return small normalized event terms for guarded date comparison."""
        terms = cls._comparison_tokens(text)
        if terms & _LANDING_TERMS:
            terms.add("landing-event")
        return terms

    @classmethod
    def _specific_date_tokens(cls, text: str) -> set[str]:
        """Return month/day tokens when a claim contains a calendar date."""
        normalized = text.casefold()
        date_tokens: set[str] = set()
        for month in _MONTHS:
            if re.search(rf"\b{month}\s+\d{{1,2}}\b", normalized):
                date_tokens.add(month)
        for day in re.findall(rf"\b(?:{'|'.join(_MONTHS)})\s+(\d{{1,2}})\b", normalized):
            date_tokens.add(day)
        return date_tokens

    @staticmethod
    def _has_direct_negation(text: str) -> bool:
        """Detect simple sentence-level negation.

        This avoids treating words such as 'not' anywhere inside
        a long evidence passage as proof that the evidence
        contradicts the claim.
        """

        return bool(RuleBasedVerifier._negated_predicates(text))

    @classmethod
    def _negated_predicates(cls, text: str) -> set[str]:
        """Return the factual words directly governed by sentence negation."""
        normalized = text.casefold()
        matches = re.findall(
            r"\b(?:is|are|was|were|does|do|did|has|have|had|"
            r"can|cannot|could|will|would|should|must)\s+"
            r"(?:not|never)\s+([a-z0-9]+(?:-[a-z0-9]+)*)\b",
            normalized,
        )
        return {cls._canonical_token(token) for token in matches}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Tokenize text and remove common stop words."""

        return {
            token
            for token in _TOKEN_PATTERN.findall(
                RuleBasedVerifier._normalize_safe_phrases(text)
            )
            if token not in _STOP_WORDS
        }

    @classmethod
    def _comparison_tokens(cls, text: str) -> set[str]:
        """Return normalized factual tokens used for overlap comparisons."""
        return {
            cls._canonical_token(token)
            for token in cls._tokens(text)
            if token not in _NEGATION_CUES
        }

    @staticmethod
    def _canonical_token(token: str) -> str:
        """Normalize only simple inflections needed for factual comparison."""
        if token in _SAFE_RELATION_ALIASES:
            return _SAFE_RELATION_ALIASES[token]
        if token in _IRREGULAR_FORMS:
            return _IRREGULAR_FORMS[token]
        if token.endswith("ied") and len(token) > 4:
            return f"{token[:-3]}y"
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    def _confidence(
        self,
        evidence: Evidence,
        claim_text: str,
    ) -> float:
        """Calculate evidence-quality confidence.

        This is NOT a calibrated probability of truth.
        """

        claim_tokens = self._comparison_tokens(claim_text)
        evidence_tokens = self._comparison_tokens(evidence.text)

        overlap = (
            len(claim_tokens & evidence_tokens)
            / max(1, len(claim_tokens))
        )

        return round(
            min(
                0.95,
                max(0.0, float(evidence.relevance_score))
                * (0.5 + 0.5 * overlap),
            ),
            3,
        )