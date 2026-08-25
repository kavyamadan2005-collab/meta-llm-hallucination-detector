"""Pydantic models for explainable claim extraction."""

from pydantic import BaseModel, Field


class NamedEntity(BaseModel):
    """A named entity as recognized in the original sentence."""

    text: str
    label: str


class Claim(BaseModel):
    """A factual candidate retaining its source wording and position."""

    claim_id: str
    claim_text: str = Field(min_length=1)
    original_sentence: str
    sentence_index: int = Field(ge=0)
    entities: list[NamedEntity] = Field(default_factory=list)

    @property
    def named_entities(self) -> list[NamedEntity]:
        """Alias matching the domain terminology used by the extraction contract."""
        return self.entities
