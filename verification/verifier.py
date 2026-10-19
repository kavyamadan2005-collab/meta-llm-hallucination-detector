"""Conservative deterministic evidence-based verification baseline."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

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

            strongest = self._strongest_evidence(supporting, claim.claim_text)

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

            strongest = self._strongest_evidence(contradicting, claim.claim_text)

            explanation = (
                "The retrieved evidence conflicts with the claim "
                f"(source: {strongest.source_title})."
            )

            confidence = self._confidence(
                strongest,
                claim.claim_text,
            )

        else:
            verdict = Verdict.INSUFFICIENT_EVIDENCE

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

    def _strongest_evidence(
        self,
        evidence: Sequence[Evidence],
        claim_text: str,
    ) -> Evidence:
        """Select the same representative evidence regardless of retrieval order."""
        return max(
            evidence,
            key=lambda item: (
                item.relevance_score,
                self._confidence(item, claim_text),
                item.evidence_id,
                item.source_url,
                item.source_title,
                item.text,
            ),
        )

    def _relation(
        self,
        claim: Claim,
        evidence_text: str,
    ) -> str | None:
        """Determine whether evidence supports or contradicts a claim."""

        claim_text = claim.claim_text
        asserted_text = self._contrastive_assertion(claim_text) or claim_text
        claim_tokens = self._tokens(asserted_text)
        evidence_tokens = self._tokens(evidence_text)

        if self._has_discourse_negation(claim_text) or self._has_discourse_negation(evidence_text):
            return None
        if not claim_tokens or not evidence_tokens:
            return None

        claim_content = self._comparison_tokens(asserted_text)
        evidence_content = self._comparison_tokens(evidence_text)
        relevant_evidence = self._most_relevant_evidence_sentences(
            asserted_text,
            evidence_text,
        )

        if self._has_conflicting_explicit_date(claim, evidence_text):
            return "contradict"
        if self._has_conflicting_numeric_context(asserted_text, evidence_text):
            return "contradict"

        claim_relation = self._relation_signature(asserted_text)
        evidence_relation = self._relation_signature(evidence_text)
        aligned_relation = False
        if self._has_person_role_mismatch(claim_relation, evidence_text):
            return None
        if self._has_same_person_different_role(claim_relation, evidence_text):
            return None
        if self._is_role_claim_same_person_other_office(claim, evidence_text):
            return None
        if self._has_partial_compound_support(claim_text, evidence_text):
            return None
        if self._has_reversed_founded_relation(claim_text, evidence_text):
            return None
        if claim_relation and evidence_relation:
            if relevant_evidence:
                evidence_polarities = {
                    self._polarity(sentence)
                    for sentence in relevant_evidence
                }
                if len(evidence_polarities) > 1:
                    return None
                if (
                    self._polarity(asserted_text) not in evidence_polarities
                    and not self._has_contrastive_negation(claim_text)
                ):
                    return "contradict"
            if self._has_conflicting_numeric_argument(
                asserted_text,
                evidence_text,
                claim_relation,
                evidence_relation,
            ):
                return "contradict"
            if self._has_conflicting_role_holder(
                claim_relation,
                evidence_relation,
            ):
                return "contradict"

        if self._is_safe_paraphrase_match(
            asserted_text,
            evidence_text,
            claim_relation,
            evidence_relation,
        ):
            return "support"
        if claim_relation and evidence_relation:
            aligned_relation = self._has_aligned_relation(
                asserted_text,
                evidence_text,
                claim_relation,
                evidence_relation,
            )
            if self._relation_conflicts(claim_relation, evidence_relation):
                if (
                    claim_relation[1] == evidence_relation[1] == "capital_of"
                    and self._relation_arguments_match(
                        claim_relation[0],
                        evidence_relation[0],
                    )
                    and not claim_relation[2] & evidence_relation[2]
                ):
                    return "contradict"
                if not aligned_relation:
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

        if not self._entities_are_covered(claim, evidence_text, asserted_text):
            return None

        required_date_tokens = self._specific_date_tokens(asserted_text)
        if required_date_tokens and not required_date_tokens.issubset(
            self._comparison_tokens(evidence_text)
        ):
            return None

        # If both texts contain years but the years disagree,
        # treat the evidence as contradictory.
        claim_numbers = set(
            re.findall(r"\b\d{4}\b", asserted_text)
        )

        evidence_numbers = set(
            re.findall(r"\b\d{4}\b", evidence_text)
        )
        evidence_numbers -= {
            date[2]
            for date, _, is_metadata in self._explicit_date_candidates(evidence_text)
            if is_metadata and date[2] is not None
        }

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
        claim_negated = self._negated_predicates(asserted_text)
        relevant_evidence_content = set().union(
            *(
                self._comparison_tokens(sentence)
                for sentence in relevant_evidence
            )
        )
        evidence_negated = set().union(
            *(
                self._negated_predicates(sentence)
                for sentence in relevant_evidence
            )
        )

        if claim_negated and not claim_negated.intersection(relevant_evidence_content):
            return None
        if evidence_negated and not evidence_negated.intersection(claim_content):
            return None

        # A positive claim vs directly negated evidence is contradictory.
        if (
            not claim_negated
            and evidence_negated.intersection(relevant_evidence_content)
            and evidence_negated.intersection(claim_content)
        ):
            return "contradict"

        # A negated claim vs positive evidence is contradictory.
        if (
            claim_negated.intersection(claim_content)
            and not evidence_negated
            and claim_negated.intersection(relevant_evidence_content)
        ):
            return "contradict"

        return "support"

    @classmethod
    def _most_relevant_evidence_sentences(
        cls,
        claim_text: str,
        evidence_text: str,
    ) -> list[str]:
        """Keep polarity checks within evidence sentences that best match the claim."""
        claim_tokens = cls._comparison_tokens(claim_text)
        if not claim_tokens:
            return []

        scored_sentences: list[tuple[float, str]] = []
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", evidence_text):
            sentence = sentence.strip()
            sentence_tokens = cls._comparison_tokens(sentence)
            if not sentence_tokens:
                continue
            overlap = len(claim_tokens & sentence_tokens) / len(claim_tokens)
            scored_sentences.append((overlap, sentence))

        if not scored_sentences:
            return []

        best_overlap = max(overlap for overlap, _ in scored_sentences)
        if best_overlap < 0.25:
            return []
        return [
            sentence
            for overlap, sentence in scored_sentences
            if best_overlap - overlap < 0.01
        ]

    def _entities_are_covered(
        self,
        claim: Claim,
        evidence_text: str,
        claim_text: str | None = None,
    ) -> bool:
        """Require extracted claim entities to appear in supporting evidence."""
        evidence_tokens = self._comparison_tokens(evidence_text)
        claim_tokens = self._comparison_tokens(claim_text or claim.claim_text)
        role_claim = bool(
            {
                "prime",
                "president",
                "chief",
                "minister",
                "mayor",
                "governor",
                "chair",
                "leader",
            }
            & claim_tokens
        )
        required_entities = claim.entities
        if role_claim:
            required_entities = [
                entity
                for entity in claim.entities
                if entity.label.casefold() not in {"gpe", "loc"}
            ]
        non_factual_labels = {
            "date",
            "time",
            "cardinal",
            "ordinal",
            "percent",
            "money",
            "quantity",
        }
        for entity in required_entities:
            if entity.label.casefold() in non_factual_labels:
                continue
            entity_tokens = self._comparison_tokens(entity.text)
            if not entity_tokens.issubset(claim_tokens):
                continue
            if entity_tokens.issubset(evidence_tokens):
                continue

            exact_anchors = entity_tokens & evidence_tokens
            unmatched_tokens = entity_tokens - evidence_tokens
            if (not exact_anchors and len(entity_tokens) > 1) or not all(
                len(token) >= 4
                and any(
                    len(candidate) >= 4
                    and SequenceMatcher(None, token, candidate).ratio() >= 0.9
                    for candidate in evidence_tokens
                )
                for token in unmatched_tokens
            ):
                return False
        return True

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
    def _has_person_role_mismatch(
        cls,
        claim_relation: tuple[frozenset[str], str, frozenset[str]] | None,
        evidence_text: str,
    ) -> bool:
        """Reject same-person evidence that mentions a different office or fact."""
        if claim_relation is None or claim_relation[1] != "is":
            return False
        claim_subject, _, claim_object = claim_relation
        if not claim_subject or not claim_object:
            return False

        claim_heads = {
            token for token in claim_object
            if token in {"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"}
        }
        if not claim_heads:
            return False

        evidence_tokens = cls._comparison_tokens(evidence_text)
        if not claim_subject & evidence_tokens:
            return False

        evidence_heads = {
            token
            for token in cls._comparison_tokens(evidence_text)
            if token in {"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"}
        }
        if not evidence_heads:
            return True

        return not claim_heads & evidence_heads

    @classmethod
    def _is_role_claim_same_person_other_office(
        cls,
        claim: Claim,
        evidence_text: str,
    ) -> bool:
        """Ignore evidence that names the same person in a different office."""
        claim_tokens = cls._comparison_tokens(claim.claim_text)
        if not ({"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"} & claim_tokens):
            return False

        person_names = {
            token
            for token in claim_tokens
            if token and token[0].isupper()
        }
        if not person_names:
            return False

        claim_subject = cls._extract_subject_from_role_claim(claim.claim_text)
        if not claim_subject:
            return False

        person_match = re.search(
            r"(?P<person>[A-Z][a-zA-Z.' -]+?)\s+(?:was|is|has been|became|served as|served|assumed office as)\s+"
            r"(?:the\s+)?(?P<role>[a-z][a-z -]+?)(?:\s+of\s+(?P<jurisdiction>[a-z][a-z -]+?))?"
            r"(?:\s+(?:from|on|in|since|for)\b|[.!?,;]|$)",
            evidence_text,
        )
        if person_match is None:
            return False

        evidence_person = cls._comparison_tokens(person_match.group("person"))
        return bool(claim_subject & evidence_person)

    @classmethod
    def _extract_subject_from_role_claim(
        cls,
        claim_text: str,
    ) -> frozenset[str]:
        """Pull the named person from a role-based claim."""
        match = re.search(
            r"(?:the\s+)?(?P<role>[a-z][a-z -]+?)\s+of\s+(?P<jurisdiction>[a-z][a-z -]+?)\s+(?:is|was|has been|became|serves as|served as|assumed office as)\s+(?P<person>[A-Z][A-Za-z.' -]+?)(?:[.!?,;]|$)",
            claim_text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return frozenset(cls._comparison_tokens(match.group("person")))

        match = re.search(
            r"(?P<person>[A-Z][A-Za-z.' -]+?)\s+(?:is|was|has been|became|serves as|served as|assumed office as)\s+(?:the\s+)?(?P<role>[a-z][a-z -]+?)(?:\s+of\s+(?P<jurisdiction>[a-z][a-z -]+?))?(?:[.!?,;]|$)",
            claim_text,
        )
        if match is not None:
            return frozenset(cls._comparison_tokens(match.group("person")))
        return frozenset()

    @classmethod
    def _has_same_person_different_role(
        cls,
        claim_relation: tuple[frozenset[str], str, frozenset[str]] | None,
        evidence_text: str,
    ) -> bool:
        """Neutralize same-person evidence that asserts a different office rather than the claim."""
        if claim_relation is None or claim_relation[1] != "is":
            return False

        claim_subject, _, claim_object = claim_relation
        if not claim_subject or not claim_object:
            return False

        claim_roles = {
            token
            for token in claim_object
            if token in {"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"}
        }
        if not claim_roles:
            return False

        match = re.search(
            r"(?P<person>[a-z][a-z .'-]+?)\s+(?:was|is|has been|became|served as|served|assumed office as)\s+"
            r"(?:the\s+)?(?P<role>[a-z][a-z -]+?)(?:\s+of\s+(?P<jurisdiction>[a-z][a-z -]+?))?"
            r"(?:\s+(?:from|on|in|since|for)\b|[.!?,;]|$)",
            evidence_text.casefold(),
        )
        if match is None:
            return False

        evidence_person = cls._comparison_tokens(match.group("person"))
        if not claim_subject & evidence_person:
            return False

        evidence_roles = {
            token
            for token in cls._comparison_tokens(match.group("role"))
            if token in {"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"}
        }
        return bool(evidence_roles) and evidence_roles != claim_roles

    @classmethod
    def _relation_signature(
        cls,
        text: str,
    ) -> tuple[frozenset[str], str, frozenset[str]] | None:
        """Extract a small subject/predicate/object signature when unambiguous."""
        text = cls._normalize_safe_phrases(text)
        capital_relation = cls._capital_relation_signature(text)
        if capital_relation is not None:
            return capital_relation
        role_relation = cls._role_relation_signature(text)
        if role_relation is not None:
            return role_relation
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

    @classmethod
    def _capital_relation_signature(
        cls,
        text: str,
    ) -> tuple[frozenset[str], str, frozenset[str]] | None:
        """Extract the city-to-jurisdiction fact independently of word order."""
        normalized = text.casefold().strip()
        patterns = (
            r"\b(?P<city>[a-z][a-z .,'-]*?)\s+(?:is|was)\s+"
            r"(?:the\s+)?(?:(?:national|state|provincial)\s+)?"
            r"capital(?:\s+city)?\s+of\s+"
            r"(?P<jurisdiction>[a-z][a-z -]*?)(?:[,.!?;]|$)",
            r"\b(?:the\s+)?(?:(?:national|state|provincial)\s+)?"
            r"capital(?:\s+city)?\s+of\s+"
            r"(?P<jurisdiction>[a-z][a-z -]*?)\s+(?:is|was)\s+"
            r"(?P<city>[a-z][a-z .,'-]*?)(?:[,.!?;]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match is None:
                continue
            city = cls._comparison_tokens(match.group("city"))
            jurisdiction = cls._comparison_tokens(match.group("jurisdiction"))
            if city and jurisdiction:
                return frozenset(city), "capital_of", frozenset(jurisdiction)
        return None

    @classmethod
    def _role_relation_signature(
        cls,
        text: str,
    ) -> tuple[frozenset[str], str, frozenset[str]] | None:
        """Canonicalize common person-role claims, including served-as wording."""
        normalized = text.casefold().strip()
        patterns = (
            (
                r"(?:the\s+)?(?P<role>[a-z][a-z -]+?)\s+of\s+"
                r"(?P<jurisdiction>[a-z][a-z -]+?)\s+"
                r"(?:is|was|has been|became|serves as|served as|assumed office as|was sworn in as)\s+"
                r"(?:the\s+)?(?P<person>[a-z][a-z .'-]+?)(?:[.!?,;]|$)",
                "role",
            ),
            (
                r"(?P<person>[a-z][a-z .'-]+?)\s+"
                r"(?:is|was|has been|became|serves as|served as|assumed office as|was sworn in as)\s+"
                r"(?:the\s+)?(?P<role>[a-z][a-z -]+?)(?:\s+of\s+(?P<jurisdiction>[a-z][a-z -]+?))?(?:[.!?,;]|$)",
                "person",
            ),
        )

        for pattern, side in patterns:
            match = re.search(pattern, normalized)
            if match is None:
                continue
            person = cls._comparison_tokens(match.group("person"))
            role = cls._comparison_tokens(match.group("role"))
            jurisdiction = cls._comparison_tokens(match.groupdict().get("jurisdiction") or "")
            if not person or not role:
                continue
            if side == "role" and not jurisdiction:
                continue
            normalized_role = role | jurisdiction
            return frozenset(person), "is", frozenset(normalized_role)
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
        shared_subject = claim_subject & evidence_subject
        shared_object = claim_object & evidence_object
        subject_overlap = len(shared_subject) / max(
            1,
            min(len(claim_subject), len(evidence_subject)),
        )
        object_overlap = len(shared_object) / max(
            1,
            min(len(claim_object), len(evidence_object)),
        )
        same_subject = bool(shared_subject) and subject_overlap >= 0.75
        same_object = bool(shared_object) and object_overlap >= 0.75

        if claim_predicate == evidence_predicate:
            if (
                claim_subject == evidence_object
                and claim_object == evidence_subject
            ):
                return True
            if same_subject:
                return (
                    claim_object != evidence_object
                    and {
                        token for token in claim_object if not token.isdigit()
                    } != {
                        token for token in evidence_object if not token.isdigit()
                    }
                )
            return bool(shared_subject) and same_object

        return (
            bool(shared_subject)
            and same_object
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
        if not self._relation_arguments_match(claim_subject, evidence_subject):
            return False
        if not claim_object & evidence_object:
            return False
        object_overlap = len(claim_object & evidence_object) / max(
            1,
            min(len(claim_object), len(evidence_object)),
        )
        if object_overlap < max(self.min_token_overlap, 0.35):
            return False
        claim_content = self._comparison_tokens(claim_text)
        evidence_content = self._comparison_tokens(evidence_text)
        overlap = len(claim_content & evidence_content) / max(1, len(claim_content))
        return overlap >= max(self.min_token_overlap, 0.6)

    @staticmethod
    def _relation_arguments_match(
        claim_argument: frozenset[str],
        evidence_argument: frozenset[str],
    ) -> bool:
        """Require substantial shared identity before aligning relation arguments."""
        shared = claim_argument & evidence_argument
        overlap = len(shared) / max(
            1,
            min(len(claim_argument), len(evidence_argument)),
        )
        return bool(shared) and overlap >= 0.75

    @staticmethod
    def _polarity(text: str) -> str:
        """Detect only direct sentence-level negation or modal denial."""
        normalized = text.casefold()
        if re.search(
            r"\b(?:not|never)\s+(?:about|related\s+to|regarding|concerning|mentioning|with)\b",
            normalized,
        ):
            return "positive"
        if re.search(
            r"\b(?:is|are|was|were|has|have|can|does|do|did)\s+"
            r"(?:not(?!\s+only\b)|no|never)\b|\bcannot\b|\bnever\s+\w+",
            normalized,
        ) or re.search(
            r"\b(?:fail|fails|failed|failure)\b(?:\s+\w+){0,4}\s+"
            r"(?:land|launch|complete|reach|win|succeed)\b",
            normalized,
        ):
            return "negative"
        return "positive"

    @staticmethod
    def _has_contrastive_negation(text: str) -> bool:
        """Recognize contrastive alternatives rather than direct negation."""
        return bool(
            re.search(
                r"\b(?:is|are|was|were)\s+not\b[^.!?;]*"
                r"(?:,\s*but\b|;\s*(?:it\s+)?(?:is|are|was|were)\b)",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _contrastive_assertion(text: str) -> str | None:
        """Return the factual alternative asserted after a contrast marker."""
        match = re.search(r",\s*but\s+(.+)$", text, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(
            r";\s*(?:it\s+)?(?:is|are|was|were)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @classmethod
    def _has_partial_compound_support(
        cls,
        claim_text: str,
        evidence_text: str,
    ) -> bool:
        """Keep a multi-clause claim unsupported when evidence covers one clause."""
        if not re.search(r"\band\b", claim_text.casefold()):
            return False
        conjunction = re.search(r"\band\b", claim_text, re.IGNORECASE)
        if conjunction is None:
            return False
        prefix = claim_text[:conjunction.start()]
        if re.search(r"\b(?:between|in|on)\s+[^.!?;]*$", prefix):
            return False
        if re.search(r"\bfrom\s+[^.!?;]*\bto\s+[^.!?;]*$", prefix):
            return False
        claim_parts = [prefix, claim_text[conjunction.end():]]
        if len(claim_parts) != 2:
            return False
        evidence_tokens = cls._comparison_tokens(evidence_text)
        clause_coverage = [
            cls._essential_clause_tokens(part)
            for part in claim_parts
        ]
        if not clause_coverage or any(not tokens for tokens in clause_coverage):
            return False
        if all(tokens.issubset(evidence_tokens) for tokens in clause_coverage):
            return False
        return True

    @classmethod
    def _essential_clause_tokens(cls, clause_text: str) -> set[str]:
        """Return named entities and role markers needed to verify a clause."""
        clause_tokens = cls._comparison_tokens(clause_text)
        role_tokens = {
            token
            for token in clause_tokens
            if token in {"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"}
        }
        named_tokens = {
            token
            for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", clause_text)
            for token in cls._comparison_tokens(phrase)
            if token not in {"prime", "president", "chief", "minister", "mayor", "governor", "chair", "leader"}
        }
        if named_tokens:
            return named_tokens | role_tokens
        return {
            token
            for token in clause_tokens
            if token not in {"of", "the", "state", "country", "nation", "capital", "city", "government"}
        }

    @classmethod
    def _has_conflicting_numeric_argument(
        cls,
        claim_text: str,
        evidence_text: str,
        claim_relation: tuple[frozenset[str], str, frozenset[str]],
        evidence_relation: tuple[frozenset[str], str, frozenset[str]],
    ) -> bool:
        """Detect different numeric arguments for the same relation."""
        if claim_relation[1] != evidence_relation[1]:
            return False
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_text))
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))
        return (
            bool(claim_numbers and evidence_numbers)
            and claim_numbers.isdisjoint(evidence_numbers)
            and bool(claim_relation[0] & evidence_relation[0])
            and bool(claim_relation[2] & evidence_relation[2])
        )

    @classmethod
    def _has_conflicting_numeric_context(
        cls,
        claim_text: str,
        evidence_text: str,
    ) -> bool:
        """Detect different numbers in otherwise matching factual contexts."""
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_text))
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))
        if (
            not claim_numbers
            or not evidence_numbers
            or not claim_numbers.isdisjoint(evidence_numbers)
        ):
            return False
        claim_context = cls._comparison_tokens(claim_text) - claim_numbers
        evidence_context = cls._comparison_tokens(evidence_text) - evidence_numbers
        shared_context = claim_context & evidence_context
        return bool(shared_context) and (
            len(shared_context) / max(1, len(claim_context)) >= 0.5
        )

    @staticmethod
    def _has_conflicting_role_holder(
        claim_relation: tuple[frozenset[str], str, frozenset[str]],
        evidence_relation: tuple[frozenset[str], str, frozenset[str]],
    ) -> bool:
        """Detect different subjects asserting the same role or ownership fact."""
        claim_subject, claim_predicate, claim_object = claim_relation
        evidence_subject, evidence_predicate, evidence_object = evidence_relation
        if claim_predicate != evidence_predicate:
            return False
        if claim_subject & evidence_subject:
            return False
        if claim_object != evidence_object:
            return False
        return claim_predicate in {"be", "is", "have", "hold", "own", "lead"}

    def _has_conflicting_explicit_date(self, claim: Claim, evidence_text: str) -> bool:
        """Detect different dates for the same explicitly shared event."""
        claim_date = self._explicit_date(claim.claim_text)
        evidence_dates = [
            date
            for date, _, is_metadata in self._explicit_date_candidates(evidence_text)
            if not is_metadata
        ]
        if claim_date is None or not evidence_dates:
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
        claim_context = self._date_free_context(claim.claim_text)
        for evidence_date, local_context, is_metadata in self._explicit_date_candidates(evidence_text):
            if is_metadata or evidence_date == claim_date:
                continue
            evidence_event_terms = self._event_terms(local_context)
            local_tokens = self._comparison_tokens(local_context)
            local_shared_entities = [
                entity
                for entity in shared_entities
                if self._comparison_tokens(entity).issubset(local_tokens)
            ]
            if local_shared_entities and claim_event_terms & evidence_event_terms:
                return True
            evidence_context = self._date_free_context(local_context)
            shared_context = claim_context & evidence_context
            if (
                len(shared_context) >= 2
                and len(shared_context) / max(1, len(claim_context)) >= 0.5
            ):
                return True
        return False

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
    def _explicit_date_candidates(
        cls,
        text: str,
    ) -> list[tuple[tuple[str, str, str | None], str, bool]]:
        """Return dates with local context and metadata classification."""
        normalized = text.casefold()
        pattern = re.compile(
            rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b"
        )
        candidates: list[tuple[tuple[str, str, str | None], str, bool]] = []
        for match in pattern.finditer(normalized):
            sentence_start = max(
                normalized.rfind(".", 0, match.start()),
                normalized.rfind("!", 0, match.start()),
                normalized.rfind("?", 0, match.start()),
                normalized.rfind("\n", 0, match.start()),
            ) + 1
            sentence_end_candidates = [
                position
                for position in (
                    normalized.find(".", match.end()),
                    normalized.find("!", match.end()),
                    normalized.find("?", match.end()),
                    normalized.find("\n", match.end()),
                )
                if position >= 0
            ]
            sentence_end = min(sentence_end_candidates, default=len(normalized))
            context = normalized[sentence_start:sentence_end].strip()
            candidates.append(
                (
                    (match.group(1), match.group(2), match.group(3)),
                    context,
                    cls._is_metadata_date(normalized, match.start(), match.end()),
                )
            )
        return candidates

    @staticmethod
    def _is_metadata_date(text: str, start: int, end: int) -> bool:
        """Identify dates used as publication, update, or post metadata."""
        preceding = text[max(0, start - 80):start]
        following = text[end:min(len(text), end + 30)]
        return bool(
            re.search(
                r"(?:published|publication|last\s+updated|updated|posted)"
                r"\s*:?\s*$",
                preceding,
            )
            or re.match(r"\s*[·•|]\s*", following)
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
    def _has_discourse_negation(text: str) -> bool:
        """Reject meta-statements like “not about …” that are not evidence about the claim."""
        return bool(
            re.search(
                r"\b(?:not|never)\s+(?:about|related\s+to|regarding|concerning|mentioning|with)\b",
                text.casefold(),
            )
        )

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
        if re.search(
            r"\b(?:not|never)\s+(?:about|related\s+to|regarding|concerning|mentioning|with)\b",
            normalized,
        ):
            return set()
        matches = re.findall(
            r"\b(?:is|are|was|were|does|do|did|has|have|had|"
            r"can|cannot|could|will|would|should|must)\s+"
            r"(?:not(?!\s+only\b)|never)\s+(?:the\s+)?([a-z0-9]+(?:-[a-z0-9]+)*)\b",
            normalized,
        )
        ignored = {"about", "related", "regarding", "concerning", "mentioning", "with", "in", "on"}
        return {
            cls._canonical_token(token)
            for token in matches
            if cls._canonical_token(token) not in ignored
        }

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