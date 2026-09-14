"""Central configuration.

All values can be overridden with environment variables (or a .env file) so the
same code runs on a laptop for the defense demo and on a small server without
edits. See .env.example for the full list.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Summarization model ---
    # facebook/bart-large-cnn is the canonical abstractive summarizer.
    # sshleifer/distilbart-cnn-12-6 is ~2x smaller/faster with a small quality
    # drop — a good choice for a laptop demo. Swap via SUMMARIZER_MODEL env var.
    summarizer_model: str = "facebook/bart-large-cnn"
    # BART's positional embeddings cap the encoder at 1024 tokens. Longer inputs
    # are split into chunks (see summarizer.py).
    max_input_tokens: int = 1024
    # Upper bound on chunks summarized per document (evenly sampled). Each chunk
    # takes ~10-15s on a small CPU server; 0 disables the cap.
    max_chunks: int = 8
    # Target length of the generated summary, in tokens. These are intentionally
    # generous so the summary is a few informative sentences, not a one-liner.
    summary_min_length: int = 90
    summary_max_length: int = 320
    # Upper bound on how long the summary may be relative to the source, so short
    # inputs still get a fuller summary without exceeding the original text.
    summary_ratio: float = 0.75

    # --- Entity / keyword extraction ---
    spacy_model: str = "en_core_web_sm"
    # How many distinct entities to look up on Wikipedia per submission.
    max_entities: int = 6

    # --- Wikipedia retrieval ---
    wikipedia_lang: str = "en"
    wikipedia_timeout: int = 8  # seconds per HTTP call
    # Wikipedia asks API clients to send a descriptive User-Agent.
    wikipedia_user_agent: str = (
        "ContextualSummarizer/1.0 (academic project; contact: student@example.edu)"
    )

    # --- Database ---
    database_url: str = "sqlite:///./submissions.db"

    # --- CORS (frontend origins allowed to call the API) ---
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
