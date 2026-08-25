"""Data models used by evidence retrieval."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A retrieved passage with enough provenance for later verification."""

    evidence_id: str
    claim_or_query: str
    text: str = Field(min_length=1)
    source_title: str
    source_url: str
    relevance_score: float
    retrieval_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """A persisted chunk and its source metadata."""

    chunk_id: str
    document_name: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)