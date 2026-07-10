# Backend — design & defense notes

This document explains **why** the backend is built the way it is, so the
implementation can be defended academically. It maps directly onto the
methodology (Chapter 3) of the report.

## Request lifecycle

`POST /api/summarize` runs one function, `_run_pipeline`
([app/routers/summarize.py](app/routers/summarize.py)):

1. **Summarize** the input with BART (`services/summarizer.py`).
2. **Extract entities/keywords** from the *original* text (`services/entities.py`).
3. **Retrieve context** for those entities from Wikipedia (`services/wikipedia.py`).
4. **Persist** the result to the database for the history feature.
5. **Merge** all three into a single JSON response.

Steps 1 and 2–3 are independent pipelines that are combined at step 5 — exactly
the architecture in the brief.

## Why these choices

### Abstractive summarization with BART
- The brief requires *abstractive* summarization. Extractive methods (TextRank,
  LexRank) only select and copy existing sentences; they cannot rephrase.
- **BART** is a denoising sequence-to-sequence transformer. `facebook/bart-large-cnn`
  is BART fine-tuned on the CNN/DailyMail summarization corpus — the standard,
  well-cited off-the-shelf abstractive summarizer, so results are reproducible and
  easy to justify in a literature review.
- **Long-input handling.** BART's encoder is limited to 1024 tokens. We tokenize
  with the model's own tokenizer, split overflowing input into token-bounded
  chunks, summarize each, then summarize the concatenated chunk-summaries — a
  simple *map-reduce* that keeps the whole document in scope. This is a deliberate,
  defensible engineering choice rather than silently truncating the text.
- **Deterministic decoding** (`do_sample=False`) so the same input always yields the
  same summary during a live demo.

### Entity extraction with spaCy
- Retrieval quality depends entirely on *what* we look up. We use spaCy **Named
  Entity Recognition** and keep only encyclopedia-worthy types (PERSON, ORG, GPE,
  NORP, EVENT, WORK_OF_ART, …), discarding noise like DATE/PERCENT/MONEY.
- A **keyword fallback** (frequent salient noun phrases) means texts *without* named
  entities — e.g. a passage about "photosynthesis" — still retrieve context.
- Entities are de-duplicated and frequency-ranked, so the most central concepts are
  looked up first (bounded by `MAX_ENTITIES`).

### Contextual retrieval from Wikipedia
- We call the **official MediaWiki + REST APIs** directly (no scraping library):
  fewer dependencies, predictable behaviour, and a clear citation of the data source.
- Two steps per entity: a **search** call resolves free text to the best article
  title (handling redirects, casing, disambiguation), then the **REST summary**
  endpoint returns a clean extract, canonical URL, and thumbnail.
- **Graceful degradation:** each lookup is wrapped so a timeout or 404 on one entity
  never fails the whole request. Disambiguation pages are skipped.
- A descriptive `User-Agent` is sent, per Wikipedia's API etiquette.

### Database: SQLite via SQLAlchemy
- SQLite needs **zero setup** — the DB is a single file created on first run, ideal
  for a portable demo. Going through SQLAlchemy means switching to Postgres is a
  one-line `DATABASE_URL` change with no code edits, so the "Postgres or similar"
  option from the brief is preserved without extra work now.

### FastAPI
- Async-capable, and its **automatic OpenAPI docs** at `/docs` double as a live,
  interactive demo of every endpoint — handy during a defense.
- **Pydantic** schemas (`schemas.py`) enforce the request/response contract shared
  with the frontend.

## File map

```
backend/app/
├── main.py                  # FastAPI app, CORS, startup, /health
├── config.py                # env-driven settings (model, limits, DB, CORS)
├── schemas.py               # Pydantic request/response models
├── database.py              # SQLAlchemy engine + Submission model
├── routers/summarize.py     # endpoints + the merge pipeline
└── services/
    ├── summarizer.py        # BART load + chunked map-reduce summarization
    ├── entities.py          # spaCy NER + keyword fallback + ranking
    ├── wikipedia.py         # search -> summary retrieval, graceful failure
    └── extract_text.py      # .txt/.md/.pdf/.docx -> plain text
```

## Trade-offs / limits (be ready to discuss these)

- **CPU inference is slow** for `bart-large-cnn`; the distilled model or a GPU host
  is the answer. The first request also pays a one-time model download/load.
- **Entity extraction is not perfect** — NER can miss domain-specific terms; the
  keyword fallback mitigates but does not eliminate this.
- **Wikipedia coverage** varies; niche entities may return no context (handled
  gracefully with an empty-state in the UI).
- The map-reduce summary of a *very* long document is a summary-of-summaries, which
  can lose fine detail — an accepted trade-off for staying within the token limit.
