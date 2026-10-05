# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace caches models here. We bake the model into this path at build
    # time (below) so the container starts with no network dependency. Do NOT
    # mount a volume over /models or you'll hide the baked-in weights.
    HF_HOME=/models \
    # Load the model at startup so the first real request is fast.
    WARMUP_MODEL=1 \
    # --- Memory budget for a 1 vCPU / 2GB server ---
    # One core: extra BLAS/OMP threads cost memory arenas and buy no throughput.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # glibc keeps a per-thread malloc arena (64MB each) and never returns much
    # of it; capping the count keeps RSS close to what is actually live.
    MALLOC_ARENA_MAX=2 \
    # The fast tokenizers fork a thread pool per process — wasted here.
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only build of torch first — a droplet has no GPU, and the
# default PyPI wheel pulls a multi-GB CUDA build we don't need. Then the rest.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Bake the summarization model into the image for fast, offline container starts.
# Override at build time for the larger model:
#   docker build --build-arg SUMMARIZER_MODEL=facebook/bart-large-cnn .
ARG SUMMARIZER_MODEL=sshleifer/distilbart-cnn-12-6
ENV SUMMARIZER_MODEL=${SUMMARIZER_MODEL}
RUN python -c "from transformers import pipeline; pipeline('summarization', model='${SUMMARIZER_MODEL}')"

COPY . .

EXPOSE 8000

# Shell form on purpose: Render (and most PaaS hosts) inject the port to listen
# on as $PORT and will fail the deploy if nothing binds it. Falls back to 8000,
# so docker-compose and `docker run` keep working unchanged.
# One worker, deliberately: the models are the memory budget, and the job queue
# lives in this process (see services/jobs.py).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
