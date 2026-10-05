"""Entity and keyword extraction — decides *what to look up* on Wikipedia.

Design notes (for the defense):
* Primary signal is spaCy Named Entity Recognition (NER). We keep entity types
  that make good encyclopedia lookups — people, organizations, places, works,
  events, products, and nationalities/groups — and drop noisy ones like DATE,
  CARDINAL, PERCENT, MONEY.
* As a fallback (and to enrich short texts with no named entities), we add the
  most frequent salient noun-chunks/proper nouns as KEYWORD entities. This means
  a text about "photosynthesis" still retrieves context even though that word
  isn't a named entity.
* Results are de-duplicated case-insensitively and ranked by frequency, so the
  most central concepts are looked up first (bounded by settings.max_entities).
"""
from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Entity labels worth sending to an encyclopedia.
_USEFUL_LABELS = {
    "PERSON",
    "ORG",
    "GPE",  # countries, cities, states
    "LOC",  # non-GPE locations (mountains, rivers)
    "NORP",  # nationalities, religious/political groups
    "FAC",  # buildings, airports, highways
    "EVENT",
    "WORK_OF_ART",
    "PRODUCT",
    "LAW",
    "LANGUAGE",
}


@lru_cache(maxsize=1)
def _get_nlp():
    """Load and cache the spaCy pipeline."""
    import spacy

    try:
        # The lemmatizer's lookup tables are several MB and nothing here uses
        # lemmas; NER, the tagger and the parser all stay.
        return spacy.load(settings.spacy_model, exclude=["lemmatizer"])
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            f"spaCy model '{settings.spacy_model}' is not installed. Run:\n"
            f"    python -m spacy download {settings.spacy_model}"
        ) from exc


def get_nlp():
    """Shared spaCy pipeline, loaded once and reused by other services."""
    return _get_nlp()


# spaCy refuses very long strings, so sentence splitting runs block by block.
_MAX_BLOCK_CHARS = 90_000


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with spaCy.

    spaCy rather than NLTK: the model is already loaded for entity extraction
    (no extra dependency, no punkt download), and its parser handles
    abbreviations like "Prof." and "Fig. 3" better than punctuation rules.
    """
    text = (text or "").strip()
    if not text:
        return []

    nlp = get_nlp()
    # Only the tokenizer and parser are needed for sentence boundaries; skipping
    # the rest (NER, tagger, lemmatizer) makes this several times faster.
    keep = {"tok2vec", "transformer", "parser", "senter"}
    disable = [name for name in nlp.pipe_names if name not in keep]

    sentences: list[str] = []
    with nlp.select_pipes(disable=disable):
        for doc in nlp.pipe(_blocks(text)):
            sentences.extend(s.text.strip() for s in doc.sents if s.text.strip())
    return sentences


def _blocks(text: str) -> list[str]:
    """Break very long text into parse-sized blocks, preferring line breaks."""
    if len(text) <= _MAX_BLOCK_CHARS:
        return [text]

    blocks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        # A single line longer than the limit is hard-split; nothing else to do.
        while len(line) > _MAX_BLOCK_CHARS:
            if current:
                blocks.append(current)
                current = ""
            blocks.append(line[:_MAX_BLOCK_CHARS])
            line = line[_MAX_BLOCK_CHARS:]
        if len(current) + len(line) > _MAX_BLOCK_CHARS:
            blocks.append(current)
            current = line
        else:
            current += line
    if current.strip():
        blocks.append(current)
    return blocks


def _clean(term: str) -> str:
    return " ".join(term.split()).strip(" \t\n.,;:\"'()[]")


def extract_entities(text: str) -> list[dict]:
    """Return a ranked, de-duplicated list of {text, label} dicts."""
    nlp = _get_nlp()
    doc = nlp(text)

    # Count named entities of useful types.
    named: Counter[tuple[str, str]] = Counter()
    for ent in doc.ents:
        if ent.label_ not in _USEFUL_LABELS:
            continue
        term = _clean(ent.text)
        if len(term) < 2:
            continue
        named[(term, ent.label_)] += 1

    # Keyword fallback: frequent proper nouns / salient noun chunks.
    keywords: Counter[str] = Counter()
    for chunk in doc.noun_chunks:
        # Strip leading determiners ("the theory" -> "theory").
        tokens = [t for t in chunk if t.pos_ in {"NOUN", "PROPN"} and not t.is_stop]
        if not tokens:
            continue
        term = _clean(" ".join(t.text for t in tokens))
        if len(term) < 3:
            continue
        keywords[term.lower()] += 1

    # Build a ranked, de-duplicated result. Named entities first (higher value),
    # then keywords that aren't already covered by a named entity.
    seen: set[str] = set()
    ranked: list[dict] = []

    for (term, label), _count in named.most_common():
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append({"text": term, "label": label})

    for term_lower, count in keywords.most_common():
        if count < 2:  # only keywords that recur, to avoid noise
            continue
        if term_lower in seen or any(term_lower in s or s in term_lower for s in seen):
            continue
        seen.add(term_lower)
        ranked.append({"text": term_lower, "label": "KEYWORD"})

    result = ranked[: settings.max_entities]
    logger.info("Extracted %d entities/keywords", len(result))
    return result
