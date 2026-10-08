from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.db import get_connection
from app.model_runtime import predict_one
from app.schemas import PredictDelayRequest, PredictDelayResponse
from ml.features import build_features

router = APIRouter(tags=["predict"])


def _row_from_shipment_id(shipment_id: str) -> pd.DataFrame:
    con = get_connection()
    try:
        ship = con.execute(
            """
            SELECT shipment_id, origin_port, destination_port, cargo_type,
                   container_count, weight_tons, booking_date,
                   planned_departure, planned_arrival
            FROM curated_shipments WHERE shipment_id = ?
            """,
            [shipment_id],
        ).fetchdf()
        ports = con.execute("SELECT * FROM curated_ports").fetchdf()
    finally:
        con.close()

    if ship.empty:
        raise HTTPException(status_code=404, detail=f"Shipment '{shipment_id}' not found")

    return build_features(ship, ports)


def _row_from_request(payload: PredictDelayRequest) -> pd.DataFrame:
    required = [
        "origin_port",
        "destination_port",
        "cargo_type",
        "container_count",
        "weight_tons",
        "booking_date",
        "planned_departure",
        "planned_arrival",
    ]
    missing = [f for f in required if getattr(payload, f) is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Either provide an existing `shipment_id`, or all of: "
                f"{', '.join(required)}. Missing: {', '.join(missing)}"
            ),
        )

    con = get_connection()
    try:
        ports = con.execute("SELECT * FROM curated_ports").fetchdf()
    finally:
        con.close()

    row = pd.DataFrame(
        [
            {
                "shipment_id": payload.shipment_id or "AD-HOC",
                "origin_port": payload.origin_port,
                "destination_port": payload.destination_port,
                "cargo_type": payload.cargo_type,
                "container_count": payload.container_count,
                "weight_tons": payload.weight_tons,
                "booking_date": payload.booking_date,
                "planned_departure": payload.planned_departure,
                "planned_arrival": payload.planned_arrival,
            }
        ]
    )
    return build_features(row, ports)


@router.post("/predict-delay", response_model=PredictDelayResponse)
def predict_delay(payload: PredictDelayRequest) -> PredictDelayResponse:
    if payload.shipment_id and not payload.origin_port:
        feature_row = _row_from_shipment_id(payload.shipment_id)
    else:
        feature_row = _row_from_request(payload)

    proba, label, model_name, threshold = predict_one(feature_row)
    return PredictDelayResponse(
        shipment_id=payload.shipment_id,
        delay_risk_probability=round(proba, 4),
        risk_label=label,
        model_name=model_name,
        threshold_hours=threshold,
    )
