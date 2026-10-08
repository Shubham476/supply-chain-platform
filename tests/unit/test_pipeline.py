"""Unit tests for the data pipeline's cleaning/derivation logic."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from build_pipeline import clean_ports, clean_shipments  # noqa: E402


def _raw_shipments():
    return pd.DataFrame(
        {
            "shipment_id": ["SHP-1", "SHP-1", "SHP-2", "SHP-3", "SHP-4"],
            "customer_id": ["C1", "C1", "C2", "C3", "C4"],
            "origin_port": ["SGSIN", "SGSIN", "SGSIN", "ZZZZZ", "SGSIN"],
            "destination_port": ["NLRTM", "NLRTM", "NLRTM", "NLRTM", "NLRTM"],
            "vessel_id": ["V1", "V1", "V2", "V3", "V4"],
            "booking_date": ["2024-01-01"] * 5,
            "planned_departure": ["2024-01-05"] * 5,
            "actual_departure": ["2024-01-05"] * 5,
            "planned_arrival": ["2024-01-20"] * 5,
            "actual_arrival": ["2024-01-22"] * 5,
            "container_count": [10, 10, 0, 5, 5],
            "cargo_type": ["ELECTRONICS", "ELECTRONICS", "Chemicals", "Furniture", None],
            "weight_tons": [500.0, 500.0, -50.0, 300.0, 400.0],
            "status": ["delivered", "delivered", "DELAYED", "Complete", "COMPLETED"],
        }
    )


def test_exact_duplicates_are_dropped():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    # SHP-1 appears twice identically -> collapses to a single row.
    assert (df["shipment_id"] == "SHP-1").sum() == 1


def test_invalid_port_rows_are_removed():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    assert "SHP-3" not in set(df["shipment_id"])


def test_zero_container_rows_are_removed():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    assert "SHP-2" not in set(df["shipment_id"])


def test_negative_weight_is_corrected_not_dropped():
    raw = _raw_shipments()
    raw.loc[raw["shipment_id"] == "SHP-2", "container_count"] = 5  # keep the row this time
    df = clean_shipments(raw, {"SGSIN", "NLRTM"})
    row = df[df["shipment_id"] == "SHP-2"].iloc[0]
    assert row["weight_tons"] == 50.0
    assert bool(row["weight_corrected"]) is True


def test_cargo_type_casing_normalised():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    assert set(df["cargo_type"].unique()) <= {"Electronics", "Furniture", "Unknown"}


def test_status_synonyms_normalised():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    assert set(df["status"].unique()) <= {"DELIVERED", "DELAYED"}


def test_derived_fields_present():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    for col in [
        "actual_delay_hours",
        "on_time_flag",
        "route_key",
        "transit_days_planned",
        "transit_days_actual",
    ]:
        assert col in df.columns


def test_route_key_format():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    assert all(df["route_key"].str.contains("\u2192"))


def test_on_time_flag_threshold():
    df = clean_shipments(_raw_shipments(), {"SGSIN", "NLRTM"})
    # planned_arrival 2024-01-20, actual_arrival 2024-01-22 -> 48h late -> not on time
    row = df.iloc[0]
    assert row["actual_delay_hours"] == 48.0
    assert bool(row["on_time_flag"]) is False


def test_clean_ports_fills_missing_country():
    ports = pd.DataFrame(
        {
            "port_code": ["SGSIN", "BEANR"],
            "port_name": ["Singapore", "Antwerp"],
            "country": ["Singapore", None],
            "region": ["APAC", "EUR"],
            "timezone": ["UTC+8", "UTC+1"],
            "avg_congestion_score": [0.7, 0.6],
        }
    )
    cleaned = clean_ports(ports)
    assert cleaned.loc[cleaned["port_code"] == "BEANR", "country"].iloc[0] == "Unknown"
