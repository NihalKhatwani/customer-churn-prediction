# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# libgomp1 is required at runtime by XGBoost (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code + pre-trained model artifacts + processed data
# (baked into the image so the container is self-contained and reproducible;
# retrain with `python -m src.models.train_model` and rebuild to refresh)
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY models/ models/
COPY data/processed/ data/processed/
COPY reports/ reports/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: serve the prediction API. Override CMD (see docker-compose.yml)
# to run the Streamlit dashboard instead.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
