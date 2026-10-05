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
# How long the durable job rows are kept before being cleaned up.
RECORD_RETENTION_DAYS = 7
# Old rows are also pruned while running, at most this often.
RECORD_PRUNE_INTERVAL_SECONDS = 60 * 60
_last_record_prune = 0.0
# How often the background sweep runs, independent of request traffic. Each
# finished Job.result holds the full input text, summary and context, so on a
# quiet server — with no new submissions to piggyback a prune on — that memory
# would otherwise sit there for up to JOB_TTL_SECONDS.
PRUNE_INTERVAL_SECONDS = 5 * 60
_pruner: threading.Thread | None = None

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


def _prune_loop() -> None:
    """Reclaim finished jobs on a timer, whatever the traffic looks like."""
    while True:
        time.sleep(PRUNE_INTERVAL_SECONDS)
        try:
            with _lock:
                _prune()
            # Must be outside the lock above: this takes `_lock` itself.
            _maybe_prune_records()
        except Exception as exc:  # a sweep failing must not kill the thread
            logger.warning("Job prune sweep failed: %s", exc)


def start_pruner() -> None:
    """Start the background sweep once, from application startup.

    Started here rather than at import time so that importing this module has
    no side effects (tests, CLI tooling, `--reload` double imports).
    """
    global _pruner
    with _lock:
        if _pruner is not None and _pruner.is_alive():
            return
        _pruner = threading.Thread(
            target=_prune_loop, daemon=True, name="job-pruner"
        )
        _pruner.start()
    logger.info("Job pruner started (every %ds)", PRUNE_INTERVAL_SECONDS)


def _prune_records(db) -> int:
    """Delete finished job rows past the retention window. Caller commits."""
    from sqlalchemy import select

    from ..database import JobRecord

    cutoff = datetime.utcnow() - timedelta(days=RECORD_RETENTION_DAYS)
    stale = db.scalars(select(JobRecord).where(JobRecord.finished_at < cutoff)).all()
    for row in stale:
        db.delete(row)
    if stale:
        logger.info("Pruned %d old job record(s)", len(stale))
    return len(stale)


def _maybe_prune_records() -> None:
    """Prune old rows at most once per interval.

    Startup alone isn't enough: this service can run for weeks between
    restarts, and the table would grow the whole time.
    """
    global _last_record_prune
    now = time.time()
    with _lock:
        if now - _last_record_prune < RECORD_PRUNE_INTERVAL_SECONDS:
            return
        _last_record_prune = now

    from ..database import SessionLocal

    db = SessionLocal()
    try:
        _prune_records(db)
        db.commit()
    except Exception as exc:  # bookkeeping must never fail a submission
        logger.warning("Could not prune job records: %s", exc)
        db.rollback()
    finally:
        db.close()


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
    _maybe_prune_records()

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
        _prune_records(db)
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
