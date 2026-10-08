from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db import get_connection
from app.schemas import RouteStats

router = APIRouter(tags=["routes"])


@router.get("/routes/{origin}/{destination}/stats", response_model=RouteStats)
def route_stats(origin: str, destination: str) -> RouteStats:
    con = get_connection()
    try:
        row = con.execute(
            """
            SELECT origin_port, destination_port, route_key, shipment_count,
                   avg_delay_hours, on_time_rate,
                   avg_transit_days_planned, avg_transit_days_actual
            FROM serving_route_stats
            WHERE origin_port = ? AND destination_port = ?
            """,
            [origin.upper(), destination.upper()],
        ).fetchdf()
    finally:
        con.close()

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No shipments found for route {origin.upper()} -> {destination.upper()}",
        )

    return RouteStats(**row.iloc[0].to_dict())
