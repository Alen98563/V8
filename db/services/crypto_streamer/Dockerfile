# V8 Crypto Streamer — Redis → TimescaleDB writer
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TSDB_DSN="postgresql://quant:quant2024@timescaledb:5432/qts_market"
ENV REDIS_URL="redis://:quant2024@redis:6379/0"
ENV FLUSH_INTERVAL_MS=200
ENV BATCH_SIZE=1000
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "main.py"]
