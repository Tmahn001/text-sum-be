"""Sentence-level source linking and faithfulness checking.

Design notes (for the defense):
* This answers the panel's "why not just use ChatGPT?" question: every summary
  sentence is traced back to the passages it came from, and checked against them.
* Two models, two different jobs:
    1. a sentence-embedding model (MiniLM) finds *where* a summary sentence came
       from, by cosine similarity over the source sentences;
    2. a natural language inference (NLI) model judges whether those passages
       actually *entail* the sentence. Topical similarity alone is not support.
* A low entailment score means "the evidence we found doesn't clearly back this
  up", not "this is false" — BART paraphrases heavily and NLI scores valid
  paraphrases as neutral surprisingly often. The API says `supported: false`;
  the UI must say "unverified", never "wrong".
* Sentence indices are assigned once, over the whole source, and are what the
  frontend highlights. They are independent of how the summarizer chunked the
  document.
* Models are loaded lazily and cached (never per request). Set VERIFY_WARMUP=1
  to pay that cost at startup instead of on the first verification — but see
  README: all three models together do not fit a 2GB server.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from .entities import get_nlp, split_sentences
from .runtime import configure_torch, inference_slot, log_peak_memory, quantize

logger = logging.getLogger(__name__)
settings = get_settings()

# Model loading is slow and not thread-safe; serialize it across requests.
_load_lock = threading.Lock()
_log_lock = threading.Lock()

@lru_cache(maxsize=1)
def _get_embedder():
    """Load and cache the sentence-embedding model."""
    from sentence_transformers import SentenceTransformer

    configure_torch()
    logger.info("Loading embedding model: %s", settings.verify_embed_model)
    model = SentenceTransformer(settings.verify_embed_model)
    # Quantize the underlying transformer, leaving the pooling layers alone.
    first = model[0]
    if hasattr(first, "auto_model"):
        first.auto_model = quantize(first.auto_model)
    return model


@lru_cache(maxsize=1)
def _get_nli():
    """Load and cache the NLI model, its tokenizer, and its entailment index."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    configure_torch()
    logger.info("Loading NLI model: %s", settings.verify_nli_model)
    tokenizer = AutoTokenizer.from_pretrained(settings.verify_nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.verify_nli_model
    ).eval()
    # Read the label order *before* quantizing: the wrapper keeps .config, but
    # read it from the source model to be safe across torch versions.
    id2label = dict(model.config.id2label)
    model = quantize(model)
    model.config.id2label = id2label
    log_peak_memory("verification load")
    # Read the entailment class index from the model config rather than assuming
    # a label order — it differs between NLI checkpoints.
    entail_index = next(
        (i for i, label in model.config.id2label.items() if "entail" in label.lower()),
        None,
    )
    if entail_index is None:
        raise RuntimeError(
            f"Model '{settings.verify_nli_model}' has no entailment label; "
            f"labels are {list(model.config.id2label.values())}. "
            "Set VERIFY_NLI_MODEL to an NLI checkpoint."
        )
    return tokenizer, model, entail_index


def _entailment_scores(premises: list[str], hypotheses: list[str]) -> list[float]:
    """P(premise entails hypothesis) for each pair, in batches."""
    import torch

    tokenizer, model, entail_index = _get_nli()
    batch_size = settings.verify_nli_batch_size
    scores: list[float] = []
    for start in range(0, len(premises), batch_size):
        batch = tokenizer(
            premises[start : start + batch_size],
            hypotheses[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            # The NLI model caps at 512 tokens, which is why only the top
            # evidence sentences are used as the premise, never the document.
            max_length=512,
        )
        with torch.no_grad(), inference_slot("verification"):
            probs = torch.softmax(model(**batch).logits, dim=-1)
        scores.extend(probs[:, entail_index].tolist())
    return scores


def _empty_result(src: list[str], ctx: list[str], threshold: float) -> dict:
    # Logged like any other run so the evaluation log has one line per request.
    _log_run(
        source_sentences=len(src),
        summary_sentences=0,
        unsupported=0,
        mean_entailment=None,
        threshold=threshold,
        duration_ms=0,
    )
    return {
        "source_sentences": src,
        "context_sentences": ctx,
        "summary": [],
        "stats": {
            "total": 0,
            "unsupported": 0,
            "mean_entailment": None,
            "threshold": threshold,
        },
        "truncated": False,
    }


def verify_summary(
    source: str,
    summary: str,
    context: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> dict:
    """Link each summary sentence to its evidence and score how well it holds up.

    Returns the full payload the frontend needs: the indexed source (and context)
    sentences, and per summary sentence its evidence references and entailment
    score. Never raises for empty or unusable input — it returns an empty result.
    """
    top_k = top_k or settings.verify_top_k
    threshold = settings.verify_threshold if threshold is None else threshold
    started = time.perf_counter()

    src_sents = split_sentences(source)
    ctx_sents = split_sentences(context) if context else []
    summ_sents = split_sentences(summary)

    # Keep the document bounded so one huge upload can't monopolize the server.
    limit = settings.verify_max_source_sentences
    truncated = limit > 0 and len(src_sents) > limit
    if truncated:
        logger.info("Source truncated for verification: %d -> %d sentences", len(src_sents), limit)
        src_sents = src_sents[:limit]

    # Evidence pool: source sentences first, then any Wikipedia context. Each
    # entry keeps its origin so the UI can highlight in the right panel — a
    # sentence drawn from context would look unsupported against the source alone.
    pool = [("source", i, s) for i, s in enumerate(src_sents)]
    pool += [("context", i, s) for i, s in enumerate(ctx_sents)]

    if not pool or not summ_sents:
        return _empty_result(src_sents, ctx_sents, threshold)

    with _load_lock:
        embedder = _get_embedder()
        _get_nli()

    import torch
    from sentence_transformers import util

    batch_size = settings.verify_embed_batch_size
    with inference_slot("verification"):
        pool_emb = embedder.encode(
            [p[2] for p in pool], convert_to_tensor=True, batch_size=batch_size
        )
        summ_emb = embedder.encode(
            summ_sents, convert_to_tensor=True, batch_size=batch_size
        )
    sims = util.cos_sim(summ_emb, pool_emb)

    premises: list[str] = []
    evidence_refs: list[list[dict]] = []
    for i in range(len(summ_sents)):
        k = min(top_k, len(pool))
        # Sorted so the premise reads in document order, not similarity order.
        idxs = sorted(torch.topk(sims[i], k=k).indices.tolist())
        premises.append(" ".join(pool[j][2] for j in idxs))
        evidence_refs.append(
            [{"origin": pool[j][0], "index": pool[j][1]} for j in idxs]
        )

    scores = _entailment_scores(premises, summ_sents)

    results = [
        {
            "sentence": sent,
            "entailment": round(score, 3),
            "supported": score >= threshold,
            "evidence": ev,
        }
        for sent, score, ev in zip(summ_sents, scores, evidence_refs)
    ]

    unsupported = sum(1 for r in results if not r["supported"])
    mean_entailment = round(sum(scores) / len(scores), 3) if scores else None
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Verified %d summary sentence(s) against %d passage(s) in %dms: %d unsupported",
        len(results), len(pool), elapsed_ms, unsupported,
    )

    _log_run(
        source_sentences=len(src_sents),
        summary_sentences=len(results),
        unsupported=unsupported,
        mean_entailment=mean_entailment,
        threshold=threshold,
        duration_ms=elapsed_ms,
    )

    return {
        "source_sentences": src_sents,
        "context_sentences": ctx_sents,
        "summary": results,
        "stats": {
            "total": len(results),
            "unsupported": unsupported,
            "mean_entailment": mean_entailment,
            "threshold": threshold,
        },
        "truncated": truncated,
    }


def _log_run(**fields) -> None:
    """Append one JSON line per run — evaluation data for Chapter 4.

    Counts and scores only: never the document text.
    """
    path = settings.verification_log_path
    if not path:
        return
    record = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **fields}
    try:
        with _log_lock:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # a read-only volume must not fail the request
        logger.warning("Could not write verification log: %s", exc)


def warmup() -> None:
    """Load both verification models up front (see main.py)."""
    with _load_lock:
        _get_embedder()
        _get_nli()
