FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install --yes --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/src ./src
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic
RUN uv sync --frozen --no-dev

RUN mkdir -p /var/lib/redacted /artifacts/corpus && chown -R app:app /app /var/lib/redacted /artifacts

USER app

EXPOSE 8000

CMD ["uvicorn", "redacted_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
