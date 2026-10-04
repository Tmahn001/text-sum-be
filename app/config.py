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
    # Length cap for each chunk's intermediate summary when a document is split.
    # max_chunks x this must fit in one model window (8 x 120 < 1008) so the final
    # map-reduce pass can run; shorter generations also make each chunk faster.
    chunk_summary_max_length: int = 120

    # --- Verification (sentence linking + faithfulness) ---
    # Finds which source sentences a summary sentence came from.
    verify_embed_model: str = "all-MiniLM-L6-v2"
    # Judges whether those sentences entail it. Must be an NLI checkpoint (the
    # entailment label is read from its config). A smaller alternative that
    # halves the memory cost: typeform/distilbert-base-uncased-mnli.
    verify_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    # Evidence sentences per summary sentence. Beyond 4 the 512-token premise
    # limit starts truncating the evidence.
    verify_top_k: int = 2
    # Entailment probability below which a sentence is marked unsupported.
    # Tune against hand-read output (see README); overridable per request.
    verify_threshold: float = 0.5
    verify_nli_batch_size: int = 16
    # Bound on source sentences compared per request; 0 disables the cap.
    verify_max_source_sentences: int = 2000
    # One JSON line per run, for the Chapter 4 evaluation. "" disables logging.
    verification_log_path: str = "verification_log.jsonl"

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
