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


class VerifyRequest(BaseModel):
    """Ask whether a summary is supported by the text it came from."""

    source: str = Field(..., description="Full original text.")
    summary: str = Field(..., description="The generated summary to check.")
    context: str | None = Field(
        None, description="Optional Wikipedia context used alongside the source."
    )
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Entailment cut-off for 'supported'; defaults to the server's.",
    )
    top_k: int | None = Field(
        None, ge=1, le=4, description="Evidence sentences per summary sentence."
    )


class EvidenceRef(BaseModel):
    """Points at one indexed sentence the frontend can highlight."""

    origin: str = Field(..., description='"source" or "context".')
    index: int = Field(..., description="Index into that origin's sentence list.")


class VerifiedSentence(BaseModel):
    sentence: str
    entailment: float = Field(..., description="P(evidence entails this sentence).")
    supported: bool = Field(
        ..., description="False means 'not clearly supported', not 'false'."
    )
    evidence: list[EvidenceRef]


class VerifyStats(BaseModel):
    total: int
    unsupported: int
    mean_entailment: float | None = None
    threshold: float


class VerifyResponse(BaseModel):
    source_sentences: list[str] = Field(
        ..., description="Indexed source sentences; evidence indices refer to these."
    )
    context_sentences: list[str] = Field(
        ..., description="Indexed context sentences, same contract."
    )
    summary: list[VerifiedSentence]
    stats: VerifyStats
    truncated: bool = Field(
        False, description="True if the source exceeded the sentence cap."
    )


class HistoryItem(BaseModel):
    id: int
    input_preview: str = Field(..., description="First ~200 chars of the input.")
    summary: str
    created_at: datetime

    class Config:
        from_attributes = True
