"""Abstractive summarization with BART (HuggingFace Transformers).

Design notes (for the defense):
* We use an *abstractive* model (BART) rather than an extractive one. Extractive
  methods (e.g. TextRank) only copy existing sentences; BART is a seq2seq
  transformer that generates new phrasing, which is what the brief asks for.
* facebook/bart-large-cnn is BART fine-tuned on the CNN/DailyMail news
  summarization dataset — the standard off-the-shelf abstractive summarizer.
* BART's encoder is capped at 1024 tokens. Real documents exceed that, so we
  chunk long inputs, summarize each chunk, then (if there are several chunks)
  summarize the concatenation of chunk-summaries — a simple "map-reduce" that
  keeps the whole document in scope without retraining anything.
* The pipeline is loaded lazily and cached, so the ~1.6GB model download/load
  cost is paid once, on the first request, not at import time.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load and cache the summarization pipeline (heavy — happens once)."""
    # Imported here rather than at module top so that importing this module
    # (e.g. in tests) doesn't pull in torch/transformers until it's needed.
    from transformers import pipeline

    logger.info("Loading summarization model: %s", settings.summarizer_model)
    return pipeline(
        "summarization",
        model=settings.summarizer_model,
        tokenizer=settings.summarizer_model,
    )


def _chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into chunks that each fit inside the model's token limit.

    We tokenize once, slice the token ids into windows, and decode each window
    back to text. Using the model's own tokenizer (not a naive word count) is
    what guarantees we never overflow the 1024-token encoder.
    """
    # Leave a little headroom for the special tokens the model adds.
    window = max_tokens - 16
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= window:
        return [text]

    chunks = []
    for start in range(0, len(token_ids), window):
        window_ids = token_ids[start : start + window]
        chunks.append(tokenizer.decode(window_ids, skip_special_tokens=True))
    return chunks


def _summarize_one(pipe, text: str) -> str:
    # max_length/min_length are measured in generated tokens. We scale the target
    # to the input so the summary is fuller for long text but never longer than a
    # short source (summary_ratio caps it at a fraction of the original).
    input_len = len(pipe.tokenizer.encode(text, add_special_tokens=False))
    max_len = min(settings.summary_max_length, max(60, int(input_len * settings.summary_ratio)))
    min_len = min(settings.summary_min_length, max(20, max_len // 2))
    result = pipe(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,  # deterministic — important for a reproducible demo
        truncation=True,
    )
    return result[0]["summary_text"].strip()


def _sample_evenly(chunks: list[str], limit: int) -> list[str]:
    """Keep at most `limit` chunks, spread evenly from start to end of the document.

    On a CPU-only server each chunk costs ~10-15s, so a 40-chunk document would
    take ~10 minutes. Sampling evenly bounds the cost while still covering the
    beginning, middle and end of the text.
    """
    if limit <= 0 or len(chunks) <= limit:
        return chunks
    if limit == 1:
        return chunks[:1]
    step = (len(chunks) - 1) / (limit - 1)
    return [chunks[round(i * step)] for i in range(limit)]


def summarize(text: str, on_progress: Callable[[int, int], None] | None = None) -> str:
    """Summarize arbitrary-length text and return a single summary string.

    `on_progress(done, total)` is called as chunks complete, so a background job
    can report progress to the frontend.
    """
    text = text.strip()
    if not text:
        return ""

    pipe = _get_pipeline()
    all_chunks = _chunk_by_tokens(text, pipe.tokenizer, settings.max_input_tokens)
    chunks = _sample_evenly(all_chunks, settings.max_chunks)

    logger.info("Summarizing in %d chunk(s) (of %d)", len(chunks), len(all_chunks))
    # +1 for the final map-reduce pass when there are several chunks.
    total = len(chunks) + (1 if len(chunks) > 1 else 0)
    chunk_summaries = []
    for chunk in chunks:
        chunk_summaries.append(_summarize_one(pipe, chunk))
        if on_progress:
            on_progress(len(chunk_summaries), total)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Map-reduce step: summarize the combined chunk-summaries into one coherent
    # summary of the whole document.
    combined = " ".join(chunk_summaries)
    combined_chunks = _chunk_by_tokens(combined, pipe.tokenizer, settings.max_input_tokens)
    if len(combined_chunks) == 1:
        return _summarize_one(pipe, combined)
    # Extremely long documents: fall back to the joined chunk summaries.
    return combined


def warmup() -> None:
    """Optionally trigger model loading at startup (see main.py)."""
    _get_pipeline()
