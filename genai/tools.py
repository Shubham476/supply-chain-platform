"""Tool implementations the GenAI assistant can call.

Each tool reads from the curated/serving data product via the running API
(so the assistant exercises the same code path a human/Postman user
would), never by querying the database directly. This keeps a single
source of truth for business logic.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_shipments",
            "description": (
                "Look up shipments from the curated data product with optional "
                "filters. Use this to answer questions about specific shipments "
                "or lists of shipments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_port": {"type": "string", "description": "5-letter UN/LOCODE, e.g. CNSHA"},
                    "destination_port": {"type": "string", "description": "5-letter UN/LOCODE, e.g. NLRTM"},
                    "status": {"type": "string", "enum": ["DELIVERED", "DELAYED", "CANCELLED", "UNKNOWN"]},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route_stats",
            "description": (
                "Get aggregated delay and on-time metrics for a specific "
                "origin -> destination route."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "5-letter origin port code"},
                    "destination": {"type": "string", "description": "5-letter destination port code"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_delay",
            "description": "Predict the delay risk for an existing shipment id by calling the ML model endpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment_id": {"type": "string"},
                },
                "required": ["shipment_id"],
            },
        },
    },
]


class Tools:
    def __init__(self, api_base_url: str) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.api_base_url, timeout=10.0)

    def query_shipments(
        self,
        origin_port: Optional[str] = None,
        destination_port: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        params = {"limit": limit}
        if origin_port:
            params["origin_port"] = origin_port
        if destination_port:
            params["destination_port"] = destination_port
        if status:
            params["status"] = status
        resp = self.client.get("/shipments", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_route_stats(self, origin: str, destination: str) -> dict[str, Any]:
        resp = self.client.get(f"/routes/{origin}/{destination}/stats")
        if resp.status_code == 404:
            return {"error": resp.json().get("detail", "route not found")}
        resp.raise_for_status()
        return resp.json()

    def predict_delay(self, shipment_id: str) -> dict[str, Any]:
        resp = self.client.post("/predict-delay", json={"shipment_id": shipment_id})
        if resp.status_code == 404:
            return {"error": resp.json().get("detail", "shipment not found")}
        resp.raise_for_status()
        return resp.json()

    def call(self, name: str, arguments: dict) -> dict[str, Any]:
        fn = getattr(self, name, None)
        if fn is None:
            return {"error": f"Unknown tool '{name}'"}
        try:
            return fn(**arguments)
        except Exception as exc:  # defensive -- tool errors must not crash the assistant
            return {"error": str(exc)}
