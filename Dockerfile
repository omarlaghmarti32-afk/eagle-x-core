FROM python:3.12-slim

WORKDIR /app

RUN useradd -m -u 1000 eagle && mkdir -p /app/data /app/logs && chown -R eagle:eagle /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY static/ static/

USER eagle

ENV EAGLE_DATA_DIR=/app/data \
    EAGLE_LOG_DIR=/app/logs \
    EAGLE_LIVE_MONITOR=1 \
    EAGLE_API_TOKEN=eagle-dev-token-change-me

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/ready')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
