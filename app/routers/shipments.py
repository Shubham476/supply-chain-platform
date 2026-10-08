from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.db import get_connection
from app.schemas import Shipment, ShipmentList

router = APIRouter(tags=["shipments"])

SELECT_COLUMNS = """
    shipment_id, customer_id, origin_port, destination_port, vessel_id,
    booking_date, planned_departure, actual_departure, planned_arrival,
    actual_arrival, container_count, cargo_type, weight_tons, status,
    actual_delay_hours, on_time_flag, route_key,
    transit_days_planned, transit_days_actual
"""


@router.get("/shipments/{shipment_id}", response_model=Shipment)
def get_shipment(shipment_id: str) -> Shipment:
    con = get_connection()
    try:
        row = con.execute(
            f"SELECT {SELECT_COLUMNS} FROM serving_shipments WHERE shipment_id = ?",
            [shipment_id],
        ).fetchdf()
    finally:
        con.close()

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Shipment '{shipment_id}' not found")

    return Shipment(**row.iloc[0].to_dict())


@router.get("/shipments", response_model=ShipmentList)
def list_shipments(
    origin_port: Optional[str] = Query(default=None),
    destination_port: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    date_from: Optional[datetime] = Query(default=None, description="planned_departure >= this"),
    date_to: Optional[datetime] = Query(default=None, description="planned_departure <= this"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ShipmentList:
    where = []
    params: list = []
    if origin_port:
        where.append("origin_port = ?")
        params.append(origin_port)
    if destination_port:
        where.append("destination_port = ?")
        params.append(destination_port)
    if status:
        where.append("status = ?")
        params.append(status.upper())
    if date_from:
        where.append("planned_departure >= ?")
        params.append(date_from)
    if date_to:
        where.append("planned_departure <= ?")
        params.append(date_to)

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    con = get_connection()
    try:
        total = con.execute(
            f"SELECT COUNT(*) FROM serving_shipments {where_clause}", params
        ).fetchone()[0]
        df = con.execute(
            f"""
            SELECT {SELECT_COLUMNS} FROM serving_shipments
            {where_clause}
            ORDER BY planned_departure
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchdf()
    finally:
        con.close()

    items = [Shipment(**r) for r in df.to_dict(orient="records")]
    return ShipmentList(items=items, total=total, limit=limit, offset=offset)
