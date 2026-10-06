"""spaCy-backed sentence and named-entity claim extraction."""

from __future__ import annotations

import re

import spacy
from spacy.language import Language
from spacy.tokens import Span

from modules.claims.models import Claim

_FILLER_STARTS = ("hello", "hi", "hey", "sure", "of course", "as an ai")
_OPINION_STARTS = ("i think", "i believe", "in my opinion", "i feel")
_STANDALONE_META_SENTENCE = re.compile(
    r"^(?:yes|no|that(?:'s| is) (?:correct|incorrect|true|false)|"
    r"this is (?:correct|incorrect)|the answer is (?:correct|incorrect))[\s.!]*$"
)
_FALLBACK_PATTERNS = (
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b"),
    re.compile(r"\b(?=[A-Za-z0-9._/-]*[A-Za-z])(?=[A-Za-z0-9._/-]*\d)[A-Za-z][A-Za-z0-9._/-]*\b"),
    re.compile(r"\b[A-Z]{2,6}\b"),
)


class ClaimExtractor:
    """Extract factual candidate sentences while preserving original text."""

    def __init__(self, nlp: Language | None = None, model_name: str = "en_core_web_sm") -> None:
        self._nlp = nlp or self._load_model(model_name)
        if not {"parser", "senter", "sentencizer"}.intersection(self._nlp.pipe_names):
            self._nlp.add_pipe("sentencizer")

    @staticmethod
    def _load_model(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is unavailable; install it before extracting claims"
            ) from exc

    def extract(self, response_text: str) -> list[Claim]:
        """Return meaningful sentence candidates from an unchanged response."""
        if not response_text.strip():
            return []
        document = self._nlp(response_text)
        claims: list[Claim] = []
        for sentence_index, sentence in enumerate(document.sents):
            original = sentence.text.strip()
            if not self._is_candidate(original):
                continue
            entities = self._entities_for_sentence(sentence)
            claims.append(
                Claim(
                    claim_id=f"C{len(claims) + 1}",
                    claim_text=original,
                    original_sentence=original,
                    sentence_index=sentence_index,
                    entities=entities,
                )
            )
        return claims

    @staticmethod
    def _entities_for_sentence(sentence: Span) -> list[dict[str, str]]:
        """Merge spaCy entities with non-overlapping conservative fallback spans."""
        entities = [
            {"text": entity.text, "label": entity.label_, "start": entity.start_char, "end": entity.end_char}
            for entity in sentence.ents
        ]
        occupied = [(entity["start"], entity["end"]) for entity in entities]
        for pattern in _FALLBACK_PATTERNS:
            for match in pattern.finditer(sentence.text):
                start = sentence.start_char + match.start()
                end = sentence.start_char + match.end()
                if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied):
                    continue
                entities.append(
                    {"text": match.group(), "label": "DOMAIN_ENTITY", "start": start, "end": end}
                )
                occupied.append((start, end))
        entities.sort(key=lambda entity: entity["start"])
        return [{key: entity[key] for key in ("text", "label")} for entity in entities]

    @staticmethod
    def _is_candidate(sentence: str) -> bool:
        normalized = re.sub(r"\s+", " ", sentence).strip().lower()
        if not normalized or normalized.endswith("?"):
            return False
        if normalized.startswith(_FILLER_STARTS) or normalized.startswith(_OPINION_STARTS):
            return False
        if _STANDALONE_META_SENTENCE.fullmatch(normalized):
            return False
        return bool(re.search(r"[A-Za-z0-9]", normalized))