# Lean serving image for the EduAssist LLMOps API.
# Builds after the model is pulled and the venv is created:
#   ./initial_deploy.sh   (or)   docker build -t eduassist-llmops-api:v1 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

WORKDIR /app

# Runtime dependencies only (no MLflow/pandas/benchmark tooling)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ /app/src/
COPY llmops_pipeline.py /app/

# Non-root user; prometheus_client writes nothing to disk but keep /app writable
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
