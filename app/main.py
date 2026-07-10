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
from .routers import summarize

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


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    # Set WARMUP_MODEL=1 to load BART at startup (slower boot, instant first
    # request). Left off by default so the server starts quickly.
    if os.getenv("WARMUP_MODEL") == "1":
        from .services.summarizer import warmup

        warmup()


@app.get("/api/health", tags=["health"])
def health() -> dict:
    """Liveness probe + which model is configured."""
    return {"status": "ok", "model": settings.summarizer_model}
