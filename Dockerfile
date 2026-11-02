FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim
WORKDIR /app

ENV PROJECT_ROOT=/app

COPY pyproject.toml .
COPY --from=builder /usr/local /usr/local
COPY src/ ./src/
COPY config.yaml .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "src.pipeline.predict"]