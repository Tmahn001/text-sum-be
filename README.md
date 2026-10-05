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

### Verifiable summaries: source linking + faithfulness (`services/verification.py`)

This is the answer to "why not just use ChatGPT?" — every summary sentence can be
traced to the passages it came from and checked against them. `POST /api/verify`
is a **separate** endpoint so the summary renders immediately and the (slower)
checking arrives behind it; if it fails, the UI shows the plain summary.

- **Two models, two different questions.** A sentence-embedding model
  (`all-MiniLM-L6-v2`) answers *where did this come from*, by cosine similarity
  over the source sentences. An **NLI** model (`cross-encoder/nli-deberta-v3-base`)
  answers *does that passage actually back this up*. Both are needed: a sentence
  can be topically close to a passage and still say something it does not support.
- **Evidence, not the whole document.** Only the top `VERIFY_TOP_K` (default 2)
  sentences become the NLI premise — the model caps at 512 tokens. Past 4
  evidence sentences the premise starts getting truncated.
- **"Unverified", never "wrong".** BART paraphrases heavily, and NLI scores many
  valid paraphrases as neutral. A score below the threshold means *the evidence
  we found doesn't clearly back this up*. The UI must never say "incorrect" or
  "hallucinated" — overclaiming is the easy way to lose that argument at the panel.
- **Threshold is a tunable, not a constant.** `VERIFY_THRESHOLD` starts at 0.5 and
  the endpoint accepts a per-request `threshold`, so it can be tuned without a
  redeploy. Tune it by reading the flagged *and* unflagged sentences from 10–15
  varied documents and trying values between 0.3 and 0.7.
- **Sentence indices are the contract.** The source is split and indexed once, so
  the frontend's highlighting is unaffected by how the summarizer chunked the
  document. spaCy does the splitting (not NLTK): the model is already loaded for
  entity extraction, so there is no extra dependency and no `punkt` download, and
  its parser handles "Prof." and "Fig. 3" better than punctuation rules.
- **Memory.** MiniLM + DeBERTa add roughly 800MB on top of BART, which does **not**
  fit the 2GB droplet. They therefore load on first use, cached for the process
  (never per request); `VERIFY_WARMUP=1` loads them at startup where there is RAM
  to spare. On a small server either raise the RAM or set
  `VERIFY_NLI_MODEL=typeform/distilbert-base-uncased-mnli`, which is about half
  the size for some accuracy.
- **Evaluation data for Chapter 4.** Every request appends one JSON line to
  `VERIFICATION_LOG_PATH` (sentence counts, unsupported count, mean entailment,
  threshold, duration) — counts and scores only, never document text.

### Background jobs, and surviving a restart

Long summaries run as background jobs (`POST /api/jobs`, then poll
`GET /api/jobs/{id}`) because a CPU-only summary takes minutes — longer than
browsers and proxies keep a request open.

The queue is in memory, so a restart used to leave the frontend polling an id
nothing remembered, reported as the unhelpful *"Job not found or expired"*. Each
status change is now mirrored to a `jobs` table:

- A poll after a restart returns `status: "error"` explaining that the server
  restarted (usually an out-of-memory kill) instead of a bare 404.
- If the work had already finished, the result is read back from its stored
  `Submission`, so the summary is not lost.
- At startup, any job still marked queued/running belongs to the dead process
  and is closed out; rows older than a week are deleted.
- A genuinely unknown id still 404s.

Progress counters stay in memory only: they change constantly, matter only while
the process lives, and each write would be a disk hit on a small server.

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
├── routers/verify.py        # POST /api/verify
└── services/
    ├── summarizer.py        # BART load + chunked map-reduce summarization
    ├── entities.py          # spaCy NER + keyword fallback + ranking
    ├── verification.py      # sentence linking + NLI faithfulness scoring
    ├── runtime.py           # int8 quantization, 1-at-a-time inference, threads
    ├── wikipedia.py         # search -> summary retrieval, graceful failure
    └── extract_text.py      # .txt/.md/.pdf/.docx -> plain text
```

## Memory budget (1 vCPU / 2GB droplet)

Three transformer models on a 2GB box does not work in fp32 — the container was
being OOM-killed mid-summary, which looked like a job hanging at "0 of 0".
Everything below is in `services/runtime.py` and switchable by env var.

| | fp32 | int8 (default) |
|---|---|---|
| distilbart-cnn-12-6 | ~1220 MB | ~470 MB |
| all-MiniLM-L6-v2 | ~90 MB | ~60 MB |
| NLI (nli-distilroberta-base) | ~330 MB | ~200 MB |
| spaCy + torch runtime + uvicorn | ~300 MB | ~300 MB |
| **total weights** | **~1.9 GB** | **~1.0 GB** |

These are calculated from parameter counts, not measured — verify them against
the `Peak memory after …` lines the service now logs on each model load.

- **Dynamic int8 quantization** (`QUANTIZE_MODELS=1`). Linear layers are stored
  as int8; embedding tables stay fp32, which is why the saving is ~2.6x rather
  than 4x. CPU inference usually gets *faster*, since int8 matmuls move less
  data. Set `QUANTIZE_MODELS=0` on a host with RAM to spare.
- **One inference at a time.** Weights are shared between requests but
  activations are not, so two concurrent generations are what actually blow the
  limit. A process-wide slot serializes them; waiting longer than
  `INFERENCE_TIMEOUT_SECONDS` returns a 503 rather than queueing forever.
- **Smaller NLI model by default** (`nli-distilroberta-base` instead of
  `nli-deberta-v3-base`): ~500 MB less, because DeBERTa-v3's 128k-token vocab
  alone is ~390 MB of fp32 embeddings. Switch back where there is RAM.
- **Single-threaded torch** plus `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1` and
  `TOKENIZERS_PARALLELISM=false` in the Dockerfile. On one core extra threads
  add per-thread malloc arenas (64MB each, rarely returned) and buy nothing.
- **Smaller batches** for NLI (8) and embeddings (32), which bounds activation
  size on the 512-token premises.
- **Upload cap** (`MAX_UPLOAD_BYTES`, 10MB). Extraction holds the file *and* its
  text in memory, so oversized uploads are now refused with a 413 before being
  read, instead of being buffered first.
- **Verification models load on first use**, not at startup, so a deployment
  that never calls `/api/verify` never pays for them.

## Trade-offs / limits (be ready to discuss these)

- **CPU inference is slow** for `bart-large-cnn`; the distilled model or a GPU host
  is the answer. The first request also pays a one-time model download/load.
- **Entity extraction is not perfect** — NER can miss domain-specific terms; the
  keyword fallback mitigates but does not eliminate this.
- **Wikipedia coverage** varies; niche entities may return no context (handled
  gracefully with an empty-state in the UI).
- The map-reduce summary of a *very* long document is a summary-of-summaries, which
  can lose fine detail — an accepted trade-off for staying within the token limit.
