"""API routes: the endpoint that ties summarization + retrieval together."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import Submission, get_db
from ..schemas import (
    HistoryItem,
    SummarizeRequest,
    SummarizeResponse,
)
from ..services import entities as entities_service
from ..services import summarizer as summarizer_service
from ..services import wikipedia as wikipedia_service
from ..services.extract_text import extract_text

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


def _run_pipeline(text: str, db: Session) -> SummarizeResponse:
    """The core pipeline: summarize -> extract entities -> retrieve -> persist.

    This is the single place where the two independent pipelines (summarization
    and contextual retrieval) are merged into one response, exactly as the
    architecture diagram in the brief describes.
    """
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # 1. Abstractive summary (BART).
    summary = summarizer_service.summarize(text)

    # 2. Identify key entities/topics, then 3. retrieve context for them.
    #    Both run on the original text so retrieval isn't limited by what the
    #    summary happened to keep.
    entities = entities_service.extract_entities(text)
    context = wikipedia_service.fetch_context(entities)

    # 4. Persist for the history feature.
    submission = Submission(
        input_text=text,
        summary=summary,
        model=settings.summarizer_model,
        entities_json=json.dumps(entities),
        context_json=json.dumps(context),
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    return SummarizeResponse(
        id=submission.id,
        input_text=text,
        summary=summary,
        entities=entities,
        context=context,
        model=settings.summarizer_model,
        created_at=submission.created_at,
    )


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_text(payload: SummarizeRequest, db: Session = Depends(get_db)):
    """Summarize pasted text and enrich it with Wikipedia context."""
    return _run_pipeline(payload.text, db)


@router.post("/summarize/upload", response_model=SummarizeResponse)
async def summarize_upload(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """Summarize an uploaded document (.txt, .md, .pdf, .docx)."""
    data = await file.read()
    try:
        text = extract_text(file.filename or "", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not text.strip():
        raise HTTPException(
            status_code=400, detail="Could not extract any text from the file."
        )
    return _run_pipeline(text, db)


@router.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20, db: Session = Depends(get_db)):
    """Return the most recent submissions (newest first)."""
    rows = db.scalars(
        select(Submission).order_by(Submission.created_at.desc()).limit(limit)
    ).all()
    return [
        HistoryItem(
            id=r.id,
            input_preview=r.input_preview,
            summary=r.summary,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/history/{submission_id}", response_model=SummarizeResponse)
def history_detail(submission_id: int, db: Session = Depends(get_db)):
    """Return a full past result (summary + entities + context)."""
    r = db.get(Submission, submission_id)
    if not r:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return SummarizeResponse(
        id=r.id,
        input_text=r.input_text,
        summary=r.summary,
        entities=r.entities,
        context=r.context,
        model=r.model,
        created_at=r.created_at,
    )
