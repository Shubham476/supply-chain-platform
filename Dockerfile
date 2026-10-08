# Single-stage image: the service is a Python API plus a handful of
# offline batch scripts (dq_check, build_pipeline, train_model) that run
# as one-off jobs via docker compose before the API starts.
FROM python:3.12-slim

WORKDIR /srv/app

# System deps kept minimal on purpose -- duckdb/pandas/scikit-learn ship
# their own wheels, no compiler toolchain required.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else sys.exit(1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
