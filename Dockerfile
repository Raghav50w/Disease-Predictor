# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder
# Installs dependencies and trains the models from the CSVs committed to this
# repo. Artifacts are never checked into git, so the image is reproducible
# from source rather than from a binary blob someone uploaded once.
FROM python:3.13-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# README.md is required: pyproject.toml declares it as the project readme.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --prefix=/install .

COPY datasets/ ./datasets/

ENV PYTHONPATH=/install/lib/python3.13/site-packages \
    PATH=/install/bin:$PATH \
    HEALTHSCAN_DATASETS_DIR=/build/datasets \
    HEALTHSCAN_MODELS_DIR=/build/models

RUN python -m healthscan.ml.preprocess && python -m healthscan.ml.train

# ----------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

# Non-root: the app only ever reads its artifacts.
RUN useradd --create-home --shell /usr/sbin/nologin healthscan

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/models /app/models
COPY static/ /app/static/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HEALTHSCAN_MODELS_DIR=/app/models \
    HEALTHSCAN_STATIC_DIR=/app/static \
    PORT=5000

USER healthscan
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/healthz').read()"

# Render (and most PaaS hosts) inject $PORT, so bind to it rather than a
# hardcoded value. Two workers fits comfortably in a 512 MB free instance.
CMD ["sh", "-c", "gunicorn healthscan.wsgi:app --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 60 --access-logfile -"]
