FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentinel ./sentinel

# Usage:  docker run --rm -v "$PWD/models:/app/models" sentinel train
ENTRYPOINT ["python", "-m", "sentinel"]
