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
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from functools import lru_cache

from ..config import get_settings
from .runtime import configure_torch, inference_slot, log_peak_memory, quantize

logger = logging.getLogger(__name__)
settings = get_settings()


# Set when the model could not be loaded and the extractive backend stood in,
# so the API can report honestly which summarizer produced a result.
_fell_back = False

# Enough to stop function words dominating the frequency counts. A full stopword
# list would mean another dependency for a fallback path.
_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his in into is it
    its of on or she that the their them they this to was were which who will with would
    about after all also any can could do does more most no not one other our out over
    said so some such than then there these those up what when where while you your""".split()
)


def active_model_name() -> str:
    """What actually produced the last summary — the model, or the fallback."""
    if settings.summarizer_backend == "extractive" or _fell_back:
        return "extractive (frequency-based)"
    return settings.summarizer_model


def _extractive_summary(text: str, max_sentences: int) -> str:
    """Pick the most informative sentences, kept in document order.

    Scores each sentence by the summed frequency of its content words, divided
    by the square root of its length so long sentences don't simply win. No
    transformer, no torch: this is the path that always survives a 512MB box.
    """
    from .entities import split_sentences

    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words_per_sentence = [
        [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in _STOPWORDS and len(w) > 2]
        for s in sentences
    ]
    frequencies = Counter(w for words in words_per_sentence for w in words)

    scored = []
    for i, words in enumerate(words_per_sentence):
        if not words:
            continue
        score = sum(frequencies[w] for w in words) / math.sqrt(len(words))
        scored.append((score, i))

    keep = sorted(i for _, i in sorted(scored, reverse=True)[:max_sentences])
    return " ".join(sentences[i] for i in keep)


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load and cache the summarization pipeline (heavy — happens once)."""
    # Imported here rather than at module top so that importing this module
    # (e.g. in tests) doesn't pull in torch/transformers until it's needed.
    from transformers import pipeline

    configure_torch()
    logger.info("Loading summarization model: %s", settings.summarizer_model)
    pipe = pipeline(
        "summarization",
        model=settings.summarizer_model,
        tokenizer=settings.summarizer_model,
    )
    # int8 weights: BART is ~1.2GB in fp32, which alone fills most of a 2GB box.
    pipe.model = quantize(pipe.model)
    log_peak_memory("summarizer load")
    return pipe


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


def _summarize_one(pipe, text: str, max_length: int | None = None) -> str:
    # max_length/min_length are measured in generated tokens. We scale the target
    # to the input so the summary is fuller for long text but never longer than a
    # short source (summary_ratio caps it at a fraction of the original).
    input_len = len(pipe.tokenizer.encode(text, add_special_tokens=False))
    cap = max_length or settings.summary_max_length
    max_len = min(cap, max(60, int(input_len * settings.summary_ratio)))
    min_len = min(settings.summary_min_length, max(20, max_len // 2))
    # One generation at a time process-wide: concurrent generations are what
    # spike memory enough to get the container killed.
    with inference_slot("summary"):
        result = pipe(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,  # deterministic — important for a reproducible demo
            # Greedy by default: beam search costs roughly its width in time,
            # which dominates on a CPU-only host.
            num_beams=settings.summarizer_num_beams,
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

    global _fell_back
    if settings.summarizer_backend == "extractive":
        logger.info("Summarizing extractively (abstractive backend disabled)")
        summary = _extractive_summary(text, settings.extractive_sentences)
        if on_progress:
            on_progress(1, 1)
        return summary

    try:
        pipe = _get_pipeline()
    except Exception as exc:  # noqa: BLE001 — a demo should degrade, not fail
        # Typically the weights are missing or too large for the instance. An
        # actual OOM kill cannot be caught here: the process just dies.
        logger.warning(
            "Could not load %s (%s); falling back to extractive summarization",
            settings.summarizer_model, exc,
        )
        _fell_back = True
        summary = _extractive_summary(text, settings.extractive_sentences)
        if on_progress:
            on_progress(1, 1)
        return summary
    _fell_back = False

    all_chunks = _chunk_by_tokens(text, pipe.tokenizer, settings.max_input_tokens)
    chunks = _sample_evenly(all_chunks, settings.max_chunks)

    logger.info("Summarizing in %d chunk(s) (of %d)", len(chunks), len(all_chunks))
    # +1 for the final map-reduce pass when there are several chunks.
    total = len(chunks) + (1 if len(chunks) > 1 else 0)
    # Report the total up front so clients see "0 of N" instead of "0 of 0"
    # while the first (slowest-feeling) chunk is still generating.
    if on_progress:
        on_progress(0, total)
    # Intermediate summaries are kept short so they all fit in the final pass.
    chunk_cap = settings.chunk_summary_max_length if len(chunks) > 1 else None
    chunk_summaries = []
    for chunk in chunks:
        started = time.perf_counter()
        chunk_summaries.append(_summarize_one(pipe, chunk, max_length=chunk_cap))
        logger.info(
            "Chunk %d/%d summarized in %.1fs",
            len(chunk_summaries), len(chunks), time.perf_counter() - started,
        )
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
