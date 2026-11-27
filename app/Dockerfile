# syntax=docker/dockerfile:1
# --- base ---
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser


# --- test ---
FROM base AS test

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .

USER appuser
CMD ["python", "-m", "pytest", "-v"]


# --- runtime ---
FROM base AS runtime

COPY --exclude=tests --exclude=requirements-dev.* --exclude=pytest.toml . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
