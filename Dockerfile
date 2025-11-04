# ===== Base image
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS deps (curl not needed anymore since no healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== Dependencies layer (cached)
FROM base AS deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -U -r /app/requirements.txt

# Install spaCy PT model (latest available for your base)
RUN python -m spacy download pt_core_news_lg

# ===== Runtime image
FROM base AS runtime
# bring installed deps
COPY --from=deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=deps /usr/local/bin /usr/local/bin

# app code
COPY . /app

# app port
EXPOSE 8000

# env (no upload limit here—set at run if you want)
ENV NLP_MAX_LENGTH=10000000

# start app
CMD ["uvicorn", "FastApi:app", "--host", "0.0.0.0", "--port", "8000"]
