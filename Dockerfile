FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_BASE_DIR=/app \
    TOR_RAB_WORKSPACE=/data/workspace \
    CHECKLIST_KATALOG_DIR=/data/workspace/cache/katalog_sbm

# Tesseract + bahasa Indonesia/Inggris + OSD (deteksi orientasi halaman scan)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng tesseract-ocr-osd \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY Ekstraktor-TOR-RAB/ Ekstraktor-TOR-RAB/
COPY Checklist-Rule-TOR-RAB/ Checklist-Rule-TOR-RAB/

# Katalog SBM bawaan repo disalin ke volume saat start (lihat docker-entrypoint.sh) agar bisa dibangun ulang
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && useradd -m -u 1000 appuser && mkdir -p /data && chown -R appuser /data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/health || exit 1
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
