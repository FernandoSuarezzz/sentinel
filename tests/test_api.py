import pytest
from fastapi.testclient import TestClient

from sentinel.api import MAX_BATCH, MAX_CHARS, create_app


@pytest.fixture
def client(detector):
    with TestClient(create_app(detector)) as c:  # `with` runs the startup lifespan
        yield c


def test_scan_safe(client):
    r = client.post("/v1/scan", json={"text": "great paella, friendly staff"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "safe"
    assert body["model_version"] == "tiny"
    assert 0 <= body["score"] < body["threshold"]


def test_scan_injection(client):
    body = client.post("/v1/scan", json={"text": "nice place ATTACK"}).json()
    assert body["verdict"] == "injection" and body["flagged_chunk"] == 0


def test_batch(client):
    r = client.post("/v1/scan/batch", json={"texts": ["fine", "ATTACK"]})
    assert [x["verdict"] for x in r.json()["results"]] == ["safe", "injection"]


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "x" * (MAX_CHARS + 1)},
    {"texts": []},
    {"texts": ["a"] * (MAX_BATCH + 1)},
    {"wrong": "field"},
])
def test_invalid_input_is_rejected(client, payload):
    path = "/v1/scan/batch" if "texts" in payload else "/v1/scan"
    assert client.post(path, json=payload).status_code == 422


def test_too_many_windows_returns_413(client):
    r = client.post("/v1/scan", json={"text": " ".join(["word"] * 500)})
    assert r.status_code == 413


def test_health_readiness_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"
    client.post("/v1/scan", json={"text": "ATTACK"})
    metrics = client.get("/metrics").text
    assert 'sentinel_scans_total{verdict="injection"} 1.0' in metrics
    assert "sentinel_scan_latency_seconds_bucket" in metrics
