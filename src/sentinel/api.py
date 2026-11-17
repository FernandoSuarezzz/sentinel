"""HTTP service: put Sentinel in front of any LLM.

Run:  SENTINEL_MODEL_DIR=artifacts/onnx uvicorn sentinel.api:app --port 8000

Endpoints
    POST /v1/scan        one text  -> verdict
    POST /v1/scan/batch  up to 32 texts (e.g. all new reviews of a business) -> verdicts
    GET  /healthz        liveness: the process is up
    GET  /readyz         readiness: the model is loaded and answering
    GET  /metrics        Prometheus metrics (request counts by verdict, latency histogram)

Privacy by design: the service never logs the texts it scans.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from sentinel import __version__
from sentinel.detector import Detector, InputTooLongError, ScanResult

MAX_CHARS = 20_000  # rejects oversized payloads before any tokenisation work
MAX_BATCH = 32

Text = Annotated[str, Field(min_length=1, max_length=MAX_CHARS)]


class ScanRequest(BaseModel):
    text: Text


class BatchScanRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH)


class ScanResponse(BaseModel):
    verdict: Literal["injection", "safe"]
    score: float = Field(description="Probability of injection of the most suspicious window")
    threshold: float
    chunks_scanned: int
    flagged_chunk: int | None
    latency_ms: float
    model_version: str


class BatchScanResponse(BaseModel):
    results: list[ScanResponse]


def create_app(detector: Detector | None = None) -> FastAPI:
    """Build the app. Tests inject a detector; in production it is loaded at startup."""
    registry = CollectorRegistry()
    scans = Counter("sentinel_scans", "Texts scanned, by verdict", ["verdict"], registry=registry)
    latency = Histogram("sentinel_scan_latency_seconds", "Model latency per request",
                        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
                        registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = detector or Detector.from_dir(os.environ["SENTINEL_MODEL_DIR"])
        yield

    app = FastAPI(title="Sentinel", version=__version__, lifespan=lifespan,
                  description="Machine-learning firewall against prompt injection.")

    def _respond(d: Detector, r: ScanResult) -> ScanResponse:
        verdict = "injection" if r.flagged else "safe"
        scans.labels(verdict=verdict).inc()
        return ScanResponse(verdict=verdict, score=r.score, threshold=d.config.threshold,
                            chunks_scanned=r.chunks_scanned, flagged_chunk=r.flagged_chunk,
                            latency_ms=r.latency_ms, model_version=d.config.version)

    def _scan(request: Request, texts: list[str]) -> list[ScanResponse]:
        d: Detector = request.app.state.detector
        try:
            with latency.time():
                results = d.scan_batch(texts)
        except InputTooLongError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        return [_respond(d, r) for r in results]

    # Plain `def` (not async): FastAPI runs these in a thread pool, and ONNX Runtime
    # releases the GIL, so concurrent requests do not block the event loop.
    @app.post("/v1/scan", response_model=ScanResponse)
    def scan(body: ScanRequest, request: Request) -> ScanResponse:
        return _scan(request, [body.text])[0]

    @app.post("/v1/scan/batch", response_model=BatchScanResponse)
    def scan_batch(body: BatchScanRequest, request: Request) -> BatchScanResponse:
        return BatchScanResponse(results=_scan(request, body.texts))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> dict:
        d: Detector | None = getattr(request.app.state, "detector", None)
        if d is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        d.scan("readiness probe")
        return {"status": "ready", "model_version": d.config.version}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
