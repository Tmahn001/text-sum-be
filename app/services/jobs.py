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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Finished jobs are forgotten after this long.
JOB_TTL_SECONDS = 60 * 60

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


def submit(work: Callable[[Callable[[int, int], None]], Any]) -> Job:
    """Queue `work(on_progress)` to run in the background and return its job."""
    job = Job(id=uuid.uuid4().hex)
    with _lock:
        _prune()
        _jobs[job.id] = job

    def on_progress(done: int, total: int) -> None:
        job.progress_done, job.progress_total = done, total

    def run() -> None:
        job.status = "running"
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

    _executor.submit(run)
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)
