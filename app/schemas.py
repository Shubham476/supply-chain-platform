from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Shipment(BaseModel):
    shipment_id: str
    customer_id: str
    origin_port: str
    destination_port: str
    vessel_id: str
    booking_date: Optional[datetime] = None
    planned_departure: Optional[datetime] = None
    actual_departure: Optional[datetime] = None
    planned_arrival: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    container_count: int
    cargo_type: str
    weight_tons: float
    status: str
    actual_delay_hours: Optional[float] = None
    on_time_flag: Optional[bool] = None
    route_key: str
    transit_days_planned: Optional[float] = None
    transit_days_actual: Optional[float] = None


class ShipmentList(BaseModel):
    items: list[Shipment]
    total: int
    limit: int
    offset: int


class RouteStats(BaseModel):
    origin_port: str
    destination_port: str
    route_key: str
    shipment_count: int
    avg_delay_hours: Optional[float] = None
    on_time_rate: Optional[float] = None
    avg_transit_days_planned: Optional[float] = None
    avg_transit_days_actual: Optional[float] = None


class PredictDelayRequest(BaseModel):
    shipment_id: Optional[str] = Field(
        default=None, description="Existing shipment id to look up booking-time features for."
    )
    origin_port: Optional[str] = None
    destination_port: Optional[str] = None
    cargo_type: Optional[str] = None
    container_count: Optional[int] = None
    weight_tons: Optional[float] = None
    booking_date: Optional[datetime] = None
    planned_departure: Optional[datetime] = None
    planned_arrival: Optional[datetime] = None


class PredictDelayResponse(BaseModel):
    shipment_id: Optional[str] = None
    delay_risk_probability: float
    risk_label: str
    model_name: str
    threshold_hours: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
