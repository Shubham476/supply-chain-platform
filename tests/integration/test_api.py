"""Integration tests: exercise real HTTP endpoints via FastAPI's TestClient
(which runs the actual ASGI app -- routing, middleware, exception handlers,
DuckDB queries, and the loaded model artefact all run for real).

These require the warehouse DB, DQ report, and model artefact to already
exist (see README "Getting started" / Makefile `make build`).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import DB_PATH, DQ_REPORT_PATH, MODEL_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DB_PATH.exists() and DQ_REPORT_PATH.exists() and MODEL_PATH.exists()),
    reason=(
        "Requires the built warehouse, DQ report, and model artefact. "
        "Run `make build` (or the pipeline/dq/train scripts) first."
    ),
)

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "version" in body


def test_list_shipments_with_pagination():
    resp = client.get("/shipments?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) <= 5
    assert body["total"] > 0


def test_get_shipment_by_id_success():
    listing = client.get("/shipments?limit=1").json()
    shipment_id = listing["items"][0]["shipment_id"]

    resp = client.get(f"/shipments/{shipment_id}")
    assert resp.status_code == 200
    assert resp.json()["shipment_id"] == shipment_id


def test_get_shipment_unknown_id_returns_404_with_structured_error():
    resp = client.get("/shipments/SHP-DOES-NOT-EXIST")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "HTTPException"
    assert "not found" in body["detail"].lower()


def test_route_stats_known_route():
    resp = client.get("/routes/NLRTM/PKKAR/stats")
    assert resp.status_code in {200, 404}  # depends on curated data; both are valid structured responses
    if resp.status_code == 200:
        body = resp.json()
        assert body["shipment_count"] > 0
        assert 0.0 <= body["on_time_rate"] <= 1.0


def test_route_stats_unknown_route_returns_404():
    resp = client.get("/routes/ZZZZZ/YYYYY/stats")
    assert resp.status_code == 404


def test_top_delayed_routes_returns_sorted_list():
    resp = client.get("/routes/top-delayed?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) <= 5
    delays = [r["avg_delay_hours"] for r in body]
    assert delays == sorted(delays, reverse=True)


def test_top_delayed_routes_respects_min_shipments():
    resp = client.get("/routes/top-delayed?limit=10&min_shipments=1000000")
    assert resp.status_code == 200
    assert resp.json() == []


def test_data_quality_report_endpoint():
    resp = client.get("/data-quality/report")
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert "issues" in body
    assert body["summary"]["total_checks"] == len(body["issues"])


def test_predict_delay_with_shipment_id():
    listing = client.get("/shipments?limit=1").json()
    shipment_id = listing["items"][0]["shipment_id"]

    resp = client.post("/predict-delay", json={"shipment_id": shipment_id})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["delay_risk_probability"] <= 1.0
    assert body["risk_label"] in {"LOW", "HIGH"}


def test_predict_delay_adhoc_without_shipment_id():
    payload = {
        "origin_port": "NLRTM",
        "destination_port": "PKKAR",
        "cargo_type": "Electronics",
        "container_count": 20,
        "weight_tons": 500,
        "booking_date": "2024-08-01T00:00:00",
        "planned_departure": "2024-08-10T00:00:00",
        "planned_arrival": "2024-08-20T00:00:00",
    }
    resp = client.post("/predict-delay", json=payload)
    assert resp.status_code == 200


def test_predict_delay_missing_everything_returns_422():
    resp = client.post("/predict-delay", json={})
    assert resp.status_code == 422


def test_response_has_request_id_header():
    resp = client.get("/health")
    assert "X-Request-ID" in resp.headers
