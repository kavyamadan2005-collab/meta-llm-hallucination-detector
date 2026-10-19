
"""spaCy-backed sentence and named-entity claim extraction."""

from __future__ import annotations

import re

import spacy
from spacy.language import Language
from spacy.tokens import Span

from modules.claims.models import Claim


_FILLER_STARTS = (
    "hello",
    "hi",
    "hey",
    "sure",
    "of course",
    "as an ai",
)

_OPINION_STARTS = (
    "i think",
    "i believe",
    "in my opinion",
    "i feel",
)

_STANDALONE_META_SENTENCE = re.compile(
    r"^(?:yes|no|no[, ]+it is not true|no[, ]+that is not true|"
    r"that(?:'s| is) (?:correct|incorrect|true|false)|"
    r"this is (?:correct|incorrect)|"
    r"the answer is (?:correct|incorrect)|"
    r"it is not true|that is not true|that's not true)"
    r"[\s.!]*$",
    re.IGNORECASE,
)

_FALLBACK_PATTERNS = (
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b"),
    re.compile(
        r"\b(?=[A-Za-z0-9._/-]*[A-Za-z])"
        r"(?=[A-Za-z0-9._/-]*\d)"
        r"[A-Za-z][A-Za-z0-9._/-]*\b"
    ),
    re.compile(r"\b[A-Z]{2,6}\b"),
)

_ROLE_ENTITY_PATTERN = re.compile(
    r"\b(?:the\s+)?"
    r"(?P<role>[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+of\s+"
    r"(?P<jurisdiction>[A-Za-z][A-Za-z\s-]*)\s+"
    r"(?:is|was|has been|became|serves as|served as|"
    r"assumed office as)\b",
    re.IGNORECASE,
)

_CONJUNCTIONS = (
    "and",
    "but",
    "while",
    "whereas",
    "additionally",
    "as well as",
)

_CLAUSE_CONJUNCTION = re.compile(
    r"\b(?:and(?!\s+(?:the|a|an)\b)|but|while|whereas|"
    r"additionally|as\s+well\s+as)\b",
    re.IGNORECASE,
)

_LEADING_LIST_MARKER = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s*)")
_STANDALONE_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]*$")

# Recognize common subject + predicate patterns while avoiding splits
# where a conjunction merely joins items in the same clause.
_INDEPENDENT_CLAUSE_PREFIX = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9'â€™\-]*|"
    r"(?:the|a|an|this|that|these|those|he|she|it|they|we|i|you)\b)"
    r"(?:\s+(?:of|the|and|[A-Za-z][A-Za-z0-9'â€™\-]*)){0,5}"
    r"\s+(?:is|are|was|were|has|have|had|can|could|will|would|"
    r"did|does|do|[A-Za-z]+ed|[A-Za-z]+s)\b",
    re.IGNORECASE,
)


class ClaimExtractor:
    """Extract factual candidate clauses while preserving source wording."""

    def __init__(
        self,
        nlp: Language | None = None,
        model_name: str = "en_core_web_sm",
    ) -> None:
        self._nlp = nlp or self._load_model(model_name)

        if not {"parser", "senter", "sentencizer"}.intersection(
            self._nlp.pipe_names
        ):
            self._nlp.add_pipe("sentencizer")

    @staticmethod
    def _load_model(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is unavailable; "
                "install it before extracting claims"
            ) from exc

    def extract(self, response_text: str) -> list[Claim]:
        """Extract factual candidates from an answer."""
        if not response_text.strip():
            return []

        document = self._nlp(response_text)
        claims: list[Claim] = []


        for sentence_index, sentence in enumerate(document.sents):
            original = sentence.text.strip()

            # Remove a leading conversational yes/no before extracting facts.
            original = re.sub(
                r"^(?:No|Yes)[.!]\s+(?=[A-Z])",
                "",
                original,
                flags=re.IGNORECASE,
            ).strip()

            if not original:
                continue

            if not self._is_candidate(original):
                continue

            for fact_text in self._split_independent_clauses(original):
                fact_text = self._clean_list_marker(fact_text)

                if not self._is_candidate(fact_text):
                    continue

                fact_document = self._nlp(fact_text)
                fact_sentence = next(iter(fact_document.sents), None)

                entities = (
                    self._entities_for_sentence(fact_sentence)
                    if fact_sentence is not None
                    else []
                )

                claims.append(
                    Claim(
                        claim_id=f"C{len(claims) + 1}",
                        claim_text=fact_text,
                        original_sentence=original,
                        sentence_index=sentence_index,
                        entities=entities,
                    )
                )

        return claims

    @staticmethod
    def _clean_list_marker(text: str) -> str:
        """Remove numbered-list artifacts that sentence splitting can isolate."""
        cleaned = _LEADING_LIST_MARKER.sub("", text)
        cleaned = _STANDALONE_LIST_MARKER.sub("", cleaned)
        return cleaned.strip()

    @classmethod
    def _split_independent_clauses(
        cls,
        sentence: str,
    ) -> list[str]:
        """Split conjunctions only when both sides state independent facts."""
        for conjunction in _CLAUSE_CONJUNCTION.finditer(sentence):
            left = sentence[:conjunction.start()].strip(" ,;")
            right = sentence[conjunction.end():].strip(" ,;")

            if (
                cls._has_explicit_subject_predicate(left)
                and cls._has_explicit_subject_predicate(right)
            ):
                punctuation = (
                    sentence[-1]
                    if sentence[-1:] in ".!?"
                    else "."
                )

                left = f"{left.rstrip('.!?')}{punctuation}"
                right = f"{right.rstrip('.!?')}{punctuation}"

                return (
                    cls._split_independent_clauses(left)
                    + cls._split_independent_clauses(right)
                )

        return [sentence.strip()]

    @staticmethod
    def _has_explicit_subject_predicate(clause: str) -> bool:
        return bool(
            _INDEPENDENT_CLAUSE_PREFIX.match(clause.strip())
        )

    @staticmethod
    def _entities_for_sentence(
        sentence: Span,
    ) -> list[dict[str, str]]:
        """Merge spaCy entities with conservative fallback entities."""
        entities = [
            {
                "text": entity.text,
                "label": entity.label_,
                "start": entity.start_char,
                "end": entity.end_char,
            }
            for entity in sentence.ents
        ]

        occupied = [
            (entity["start"], entity["end"])
            for entity in entities
        ]

        for role_match in _ROLE_ENTITY_PATTERN.finditer(sentence.text):
            role_text = role_match.group("role").strip()
            start = sentence.start_char + role_match.start("role")
            end = sentence.start_char + role_match.end("role")

            overlaps = any(
                start < occupied_end and end > occupied_start
                for occupied_start, occupied_end in occupied
            )

            if not role_text or overlaps:
                continue

            entities.append(
                {
                    "text": role_text,
                    "label": "ROLE",
                    "start": start,
                    "end": end,
                }
            )
            occupied.append((start, end))

        for pattern in _FALLBACK_PATTERNS:
            for match in pattern.finditer(sentence.text):
                start = sentence.start_char + match.start()
                end = sentence.start_char + match.end()

                overlaps = any(
                    start < occupied_end and end > occupied_start
                    for occupied_start, occupied_end in occupied
                )

                if overlaps:
                    continue

                entities.append(
                    {
                        "text": match.group(),
                        "label": "DOMAIN_ENTITY",
                        "start": start,
                        "end": end,
                    }
                )
                occupied.append((start, end))

        entities.sort(key=lambda entity: entity["start"])

        return [
            {key: entity[key] for key in ("text", "label")}
            for entity in entities
        ]

    @staticmethod
    def _is_candidate(sentence: str) -> bool:
        """Reject empty text, questions, filler, and standalone meta-answers."""
        normalized = re.sub(
            r"\s+",
            " ",
            sentence,
        ).strip().lower()

        if not normalized or normalized.endswith("?"):
            return False

        if normalized.endswith((":", ";")):
            return False

        if normalized.startswith(_FILLER_STARTS):
            return False

        if normalized.startswith(_OPINION_STARTS):
            return False

        if _STANDALONE_META_SENTENCE.fullmatch(normalized):
            return False

        return bool(re.search(r"[A-Za-z0-9]", normalized))
