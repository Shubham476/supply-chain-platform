"""
Feature engineering for the delay-risk model.

CRITICAL CONSTRAINT: every feature here must be computable at booking
time. We deliberately never touch actual_departure or actual_arrival --
those are only known after the fact and would leak the label.
"""
from __future__ import annotations

import pandas as pd

BOOKING_TIME_NUMERIC_FEATURES = [
    "container_count",
    "weight_tons",
    "transit_days_planned",
    "origin_congestion_score",
    "destination_congestion_score",
    "booking_lead_days",
    "planned_departure_dow",
    "planned_departure_month",
]
BOOKING_TIME_CATEGORICAL_FEATURES = [
    "origin_port",
    "destination_port",
    "cargo_type",
]

ALL_FEATURES = BOOKING_TIME_NUMERIC_FEATURES + BOOKING_TIME_CATEGORICAL_FEATURES


def build_features(shipments: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    """Build the booking-time feature frame.

    `shipments` is expected to be the curated table (already deduplicated /
    DQ-cleaned), but we only read columns that are known at booking time:
    booking_date, planned_departure, planned_arrival, origin_port,
    destination_port, container_count, cargo_type, weight_tons.
    """
    df = shipments.copy()

    for c in ["booking_date", "planned_departure", "planned_arrival"]:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c])

    df["transit_days_planned"] = (
        df["planned_arrival"] - df["planned_departure"]
    ).dt.total_seconds() / 86400.0
    df["booking_lead_days"] = (
        df["planned_departure"] - df["booking_date"]
    ).dt.total_seconds() / 86400.0
    df["planned_departure_dow"] = df["planned_departure"].dt.dayofweek
    df["planned_departure_month"] = df["planned_departure"].dt.month

    congestion = ports.set_index("port_code")["avg_congestion_score"]
    df["origin_congestion_score"] = df["origin_port"].map(congestion)
    df["destination_congestion_score"] = df["destination_port"].map(congestion)

    return df[["shipment_id"] + ALL_FEATURES] if "shipment_id" in df.columns else df[ALL_FEATURES]


def build_label(shipments: pd.DataFrame, on_time_threshold_hours: float = 24.0) -> pd.Series:
    """1 = arrived more than `on_time_threshold_hours` late, 0 = on time.

    Only defined where actual_delay_hours is known (i.e. the shipment has
    completed and has both planned and actual arrival times). Rows without
    a usable label (cancelled, unknown status, missing dates) return NaN
    and must be dropped by the caller before training.
    """
    return shipments["actual_delay_hours"].apply(
        lambda h: 1 if pd.notna(h) and h > on_time_threshold_hours else (0 if pd.notna(h) else None)
    )
