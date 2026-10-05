"""In-process background jobs for long-running summaries.

Summarizing a long document on a CPU-only server can take minutes, which is far
longer than browsers and reverse proxies (nginx: 60s, Cloudflare: 100s) will keep
an HTTP request open. So the API starts a job, returns its id immediately, and
the frontend polls for the result.

Jobs run on a single worker thread: the model is CPU-bound, so running two
summaries at once would only make both slower. The store is in memory, which is
fine for a single uvicorn process; finished results are also persisted to the
database by the pipeline, so nothing is lost if a job record expires.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Finished jobs are forgotten (in memory) after this long.
JOB_TTL_SECONDS = 60 * 60
# How long the durable job rows are kept before being cleaned up at startup.
RECORD_RETENTION_DAYS = 7

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summarize-job")
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    progress_done: int = 0
    progress_total: int = 0
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


def _prune() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id in [j.id for j in _jobs.values() if j.finished_at and j.finished_at < cutoff]:
        del _jobs[job_id]


def _persist(job: "Job", submission_id: int | None = None) -> None:
    """Mirror a job's status to the database so it survives a restart.

    Progress counters are deliberately *not* written on every chunk — they
    change often, matter only while the process is alive, and each write would
    be a disk hit on a small server.
    """
    from ..database import JobRecord, SessionLocal

    db = SessionLocal()
    try:
        row = db.get(JobRecord, job.id)
        if row is None:
            row = JobRecord(id=job.id)
            db.add(row)
        row.status = job.status
        row.progress_done = job.progress_done
        row.progress_total = job.progress_total
        row.error = job.error
        if submission_id is not None:
            row.submission_id = submission_id
        row.finished_at = (
            datetime.fromtimestamp(job.finished_at, tz=timezone.utc).replace(tzinfo=None)
            if job.finished_at
            else None
        )
        db.commit()
    except Exception as exc:  # a job must still run if its bookkeeping fails
        logger.warning("Could not persist job %s: %s", job.id, exc)
        db.rollback()
    finally:
        db.close()


def submit(work: Callable[[Callable[[int, int], None]], Any]) -> Job:
    """Queue `work(on_progress)` to run in the background and return its job."""
    job = Job(id=uuid.uuid4().hex)
    with _lock:
        _prune()
        _jobs[job.id] = job
    _persist(job)

    def on_progress(done: int, total: int) -> None:
        job.progress_done, job.progress_total = done, total

    def run() -> None:
        job.status = "running"
        _persist(job)
        try:
            job.result = work(on_progress)
            job.status = "done"
        except HTTPException as exc:
            job.error = str(exc.detail)
            job.status = "error"
        except Exception:  # noqa: BLE001 — report any failure to the client
            logger.exception("Job %s failed", job.id)
            job.error = "Summarization failed. Please try again."
            job.status = "error"
        finally:
            job.finished_at = time.time()
            # The pipeline stores its output as a Submission; record which one,
            # so a poll after a restart can still return the result.
            _persist(job, submission_id=getattr(job.result, "id", None))

    _executor.submit(run)
    return job


def recover_interrupted() -> int:
    """Close out jobs left mid-flight by a restart. Returns how many.

    Called at startup: nothing is still running at that point, so any job the
    database thinks is queued or running died with the previous process. Saying
    so beats letting the frontend poll an id that no longer exists.
    """
    from sqlalchemy import select

    from ..database import JobRecord, SessionLocal

    db = SessionLocal()
    try:
        rows = db.scalars(
            select(JobRecord).where(JobRecord.status.in_(("queued", "running")))
        ).all()
        for row in rows:
            row.status = "error"
            row.error = (
                "The server restarted while this summary was running — most "
                "likely it ran out of memory. Try a shorter document."
            )
            row.finished_at = datetime.utcnow()
        # Old finished rows are only useful for a while; keep the table small.
        cutoff = datetime.utcnow() - timedelta(days=RECORD_RETENTION_DAYS)
        stale = db.scalars(
            select(JobRecord).where(JobRecord.finished_at < cutoff)
        ).all()
        for row in stale:
            db.delete(row)
        db.commit()
        if rows:
            logger.warning(
                "Marked %d job(s) as interrupted by a restart: %s",
                len(rows), ", ".join(r.id for r in rows),
            )
        return len(rows)
    except Exception as exc:  # never block startup on bookkeeping
        logger.warning("Could not recover interrupted jobs: %s", exc)
        db.rollback()
        return 0
    finally:
        db.close()


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)
