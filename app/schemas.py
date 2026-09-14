"""Pydantic request/response models — the API contract shared with the frontend."""
from datetime import datetime

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to summarize.")


class WikiContext(BaseModel):
    """One piece of retrieved background information."""

    entity: str = Field(..., description="The extracted term this context is for.")
    title: str = Field(..., description="Matched Wikipedia article title.")
    extract: str = Field(..., description="Short summary from Wikipedia.")
    url: str = Field(..., description="Link to the full article.")
    thumbnail: str | None = Field(None, description="Optional article image URL.")


class Entity(BaseModel):
    text: str
    label: str = Field(..., description="Entity type, e.g. PERSON, ORG, or KEYWORD.")


class SummarizeResponse(BaseModel):
    id: int | None = None
    input_text: str = Field("", description="Original text that was summarized.")
    summary: str
    entities: list[Entity]
    context: list[WikiContext]
    model: str = Field(..., description="Summarization model used.")
    created_at: datetime | None = None


class JobStatus(BaseModel):
    """State of a background summarization job (see services/jobs.py)."""

    id: str
    status: str = Field(..., description="queued | running | done | error")
    progress_done: int = 0
    progress_total: int = 0
    result: SummarizeResponse | None = None
    error: str | None = None


class HistoryItem(BaseModel):
    id: int
    input_preview: str = Field(..., description="First ~200 chars of the input.")
    summary: str
    created_at: datetime

    class Config:
        from_attributes = True
