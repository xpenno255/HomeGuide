FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Data (SQLite DB, original PDFs, embedding model cache) lives on a volume
ENV DATA_DIR=/data \
    FASTEMBED_CACHE_PATH=/data/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
