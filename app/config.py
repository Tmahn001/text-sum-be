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

    # --- Runtime (CPU/memory budget; see services/runtime.py) ---
    # Store Linear weights as int8: ~3-4x less weight memory and usually faster
    # on CPU, for a small accuracy cost. Set 0 to keep fp32 where RAM allows.
    quantize_models: bool = True
    # Threads per inference. 1 suits a single-core server; 0 leaves torch's default.
    torch_num_threads: int = 1
    # How long a request may wait for the single inference slot before giving up.
    inference_timeout_seconds: int = 600
    # Reject uploads larger than this before they are read into memory.
    max_upload_bytes: int = 10 * 1024 * 1024

    # --- Verification (sentence linking + faithfulness) ---
    # Kill-switch for /api/verify. Its models add ~260MB (int8) on top of BART,
    # so on a memory-capped server it can be turned off without a code change:
    # the endpoint then returns 503 immediately and loads nothing. The frontend
    # falls back to the plain summary.
    verify_enabled: bool = True
    # Finds which source sentences a summary sentence came from.
    verify_embed_model: str = "all-MiniLM-L6-v2"
    # Judges whether those sentences entail it. Must be an NLI checkpoint (the
    # entailment label is read from its config). distilroberta is the default
    # because deberta-v3-base needs ~500MB more than a 2GB server has; where
    # there is RAM, cross-encoder/nli-deberta-v3-base scores better.
    verify_nli_model: str = "cross-encoder/nli-distilroberta-base"
    # Evidence sentences per summary sentence. Beyond 4 the 512-token premise
    # limit starts truncating the evidence.
    verify_top_k: int = 2
    # Entailment probability below which a sentence is marked unsupported.
    # Tune against hand-read output (see README); overridable per request.
    verify_threshold: float = 0.5
    # Batch sizes trade memory for speed; these are sized for a 2GB server.
    verify_nli_batch_size: int = 8
    verify_embed_batch_size: int = 32
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
