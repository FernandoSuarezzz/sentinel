# Serving image: ONNX Runtime + FastAPI only. No PyTorch, no training code paths.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTINEL_MODEL_DIR=/model

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Never run as root: if the process were compromised, it could not modify the image.
RUN useradd --create-home --uid 10001 sentinel
USER sentinel

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# The model is mounted at runtime (-v ./artifacts/onnx:/model:ro), so the same image
# serves any model version and the image stays small.
CMD ["uvicorn", "sentinel.api:app", "--host", "0.0.0.0", "--port", "8000"]
