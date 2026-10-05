"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload
Interactive API docs are then served at http://localhost:8000/docs
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db
from .routers import summarize, verify

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

settings = get_settings()

app = FastAPI(
    title="Contextual Text Summarizer",
    description=(
        "Summarizes text with BART and enriches it with background information "
        "retrieved from Wikipedia based on the key entities in the text."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summarize.router, prefix="/api", tags=["summarize"])
app.include_router(verify.router, prefix="/api", tags=["verify"])


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    # Nothing is running yet, so any job the DB still calls queued/running died
    # with the previous process. Close them out with a reason the UI can show.
    from .services.jobs import recover_interrupted, start_pruner

    recover_interrupted()
    # Finished jobs hold their full input and output in memory, so they are
    # swept on a timer rather than only when the next submission arrives.
    start_pruner()
    # Set WARMUP_MODEL=1 to load BART at startup (slower boot, instant first
    # request). Left off by default so the server starts quickly.
    if os.getenv("WARMUP_MODEL") == "1":
        from .services.summarizer import warmup

        warmup()
    # The verification models add ~800MB on top of BART, so they load on first
    # use by default. Set VERIFY_WARMUP=1 only where there is RAM to spare.
    if os.getenv("VERIFY_WARMUP") == "1":
        from .services.verification import warmup as verify_warmup

        verify_warmup()


@app.get("/api/health", tags=["health"])
def health() -> dict:
    """Liveness probe + which model is configured."""
    return {"status": "ok", "model": settings.summarizer_model}
