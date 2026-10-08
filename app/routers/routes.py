from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.db import get_connection
from app.schemas import RouteStats

router = APIRouter(tags=["routes"])


@router.get("/routes/top-delayed", response_model=list[RouteStats])
def top_delayed_routes(
    limit: int = Query(default=5, ge=1, le=100),
    min_shipments: int = Query(
        default=3, ge=1, description="Exclude routes with fewer shipments to avoid noisy rankings."
    ),
    order: Literal["desc", "asc"] = Query(
        default="desc", description="desc = worst (highest avg delay) first, asc = best first."
    ),
) -> list[RouteStats]:
    """Rank routes by average delay hours. Backs the GenAI assistant's
    'which routes have had the highest average delay' question -- there is
    no time-window column in the curated data, so this ranks over all
    available history rather than a true rolling quarter (documented
    assumption, see README)."""
    con = get_connection()
    try:
        df = con.execute(
            f"""
            SELECT origin_port, destination_port, route_key, shipment_count,
                   avg_delay_hours, on_time_rate,
                   avg_transit_days_planned, avg_transit_days_actual
            FROM serving_route_stats
            WHERE avg_delay_hours IS NOT NULL AND shipment_count >= ?
            ORDER BY avg_delay_hours {order.upper()}
            LIMIT ?
            """,
            [min_shipments, limit],
        ).fetchdf()
    finally:
        con.close()

    return [RouteStats(**r) for r in df.to_dict(orient="records")]


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
