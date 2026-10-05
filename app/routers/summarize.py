"""API routes: the endpoint that ties summarization + retrieval together."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import JobRecord, SessionLocal, Submission, get_db
from ..schemas import (
    HistoryItem,
    JobStatus,
    SummarizeRequest,
    SummarizeResponse,
)
from ..services import entities as entities_service
from ..services import jobs as jobs_service
from ..services import summarizer as summarizer_service
from ..services import wikipedia as wikipedia_service
from ..services.extract_text import extract_text

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


def _run_pipeline(text: str, db: Session, on_progress=None) -> SummarizeResponse:
    """The core pipeline: summarize -> extract entities -> retrieve -> persist.

    This is the single place where the two independent pipelines (summarization
    and contextual retrieval) are merged into one response, exactly as the
    architecture diagram in the brief describes.
    """
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # 1. Abstractive summary (BART).
    summary = summarizer_service.summarize(text, on_progress=on_progress)

    # 2. Identify key entities/topics, then 3. retrieve context for them.
    #    Both run on the original text so retrieval isn't limited by what the
    #    summary happened to keep.
    entities = entities_service.extract_entities(text)
    context = wikipedia_service.fetch_context(entities)

    # Report what actually ran: the configured model, or the extractive
    # fallback if the model could not be loaded.
    model_name = summarizer_service.active_model_name()

    # 4. Persist for the history feature.
    submission = Submission(
        input_text=text,
        summary=summary,
        model=model_name,
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
        model=model_name,
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
    text = _extract_upload(file, await _read_capped(file))
    return _run_pipeline(text, db)


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload in chunks, refusing anything over the size limit.

    Extraction holds the whole file plus its extracted text in memory, so an
    oversized PDF can exhaust a small server before summarization even starts.
    Reading chunk by chunk means we reject it without buffering it all first.
    """
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(256 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if limit > 0 and total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large (limit {limit // (1024 * 1024)}MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_upload(file: UploadFile, data: bytes) -> str:
    try:
        text = extract_text(file.filename or "", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not text.strip():
        raise HTTPException(
            status_code=400, detail="Could not extract any text from the file."
        )
    return text


def _start_job(text: str) -> JobStatus:
    """Run the pipeline in the background with its own DB session."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    def work(on_progress):
        db = SessionLocal()
        try:
            return _run_pipeline(text, db, on_progress=on_progress)
        finally:
            db.close()

    return _job_status(jobs_service.submit(work))


def _job_status(job: jobs_service.Job) -> JobStatus:
    return JobStatus(
        id=job.id,
        status=job.status,
        progress_done=job.progress_done,
        progress_total=job.progress_total,
        result=job.result,
        error=job.error,
    )


@router.post("/jobs", response_model=JobStatus, status_code=202)
def start_text_job(payload: SummarizeRequest):
    """Start summarizing pasted text in the background; poll GET /jobs/{id}."""
    return _start_job(payload.text)


@router.post("/jobs/upload", response_model=JobStatus, status_code=202)
async def start_upload_job(file: UploadFile = File(...)):
    """Start summarizing an uploaded document in the background."""
    text = _extract_upload(file, await _read_capped(file))
    return _start_job(text)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str, db: Session = Depends(get_db)):
    """Poll a background job. `result` is set once `status` is "done"."""
    job = jobs_service.get(job_id)
    if job:
        return _job_status(job)

    # Not in memory: either this process restarted (an out-of-memory kill mid
    # summary is the usual cause) or the in-memory record aged out. The durable
    # record says which, and still carries the result if the work finished.
    row = db.get(JobRecord, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found or expired.")

    result = None
    if row.submission_id:
        submission = db.get(Submission, row.submission_id)
        if submission:
            result = _submission_response(submission)
    return JobStatus(
        id=row.id,
        status=row.status,
        progress_done=row.progress_done,
        progress_total=row.progress_total,
        result=result,
        error=row.error,
    )


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


def _submission_response(r: Submission) -> SummarizeResponse:
    """Shape a stored submission as the pipeline's response."""
    return SummarizeResponse(
        id=r.id,
        input_text=r.input_text,
        summary=r.summary,
        entities=r.entities,
        context=r.context,
        model=r.model,
        created_at=r.created_at,
    )


@router.get("/history/{submission_id}", response_model=SummarizeResponse)
def history_detail(submission_id: int, db: Session = Depends(get_db)):
    """Return a full past result (summary + entities + context)."""
    r = db.get(Submission, submission_id)
    if not r:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return _submission_response(r)
