
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_THREAD_LIMIT=1
ENV OPENBLAS_NUM_THREADS=1
ENV TMPDIR=/tmp

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/uploads

EXPOSE 10000

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--graceful-timeout", "30", "--bind", "0.0.0.0:10000", "--chdir", "backend", "app:app"]
