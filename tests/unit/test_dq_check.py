"""Unit tests for the standalone DQ checks. Pure in-memory dataframes --
no filesystem, no network, runs in milliseconds."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dq"))
from dq_check import check_ports, check_shipments  # noqa: E402


def _base_shipments(n=3):
    return pd.DataFrame(
        {
            "shipment_id": [f"SHP-{i:05d}" for i in range(n)],
            "customer_id": [f"CUST-{i}" for i in range(n)],
            "origin_port": ["SGSIN"] * n,
            "destination_port": ["NLRTM"] * n,
            "vessel_id": ["VSL-0001"] * n,
            "booking_date": ["2024-01-01"] * n,
            "planned_departure": ["2024-01-05"] * n,
            "actual_departure": ["2024-01-05"] * n,
            "planned_arrival": ["2024-01-20"] * n,
            "actual_arrival": ["2024-01-20"] * n,
            "container_count": [10] * n,
            "cargo_type": ["Electronics"] * n,
            "weight_tons": [500.0] * n,
            "status": ["DELIVERED"] * n,
        }
    )


def test_duplicate_shipment_id_detected():
    df = _base_shipments(2)
    df.loc[1, "shipment_id"] = df.loc[0, "shipment_id"]
    issues = check_shipments(df, {"SGSIN", "NLRTM"})
    names = {i.check_name for i in issues}
    assert "shipments.duplicate_shipment_id" in names


def test_invalid_port_code_detected():
    df = _base_shipments(1)
    df.loc[0, "origin_port"] = "ZZZZZ"
    issues = check_shipments(df, {"SGSIN", "NLRTM"})
    names = {i.check_name for i in issues}
    assert "shipments.invalid_port_code" in names
    issue = next(i for i in issues if i.check_name == "shipments.invalid_port_code")
    assert issue.rows_affected == 1
    assert issue.severity == "critical"


def test_negative_weight_detected():
    df = _base_shipments(1)
    df.loc[0, "weight_tons"] = -50.0
    issues = check_shipments(df, {"SGSIN", "NLRTM"})
    names = {i.check_name for i in issues}
    assert "shipments.negative_weight_tons" in names


def test_zero_container_count_detected():
    df = _base_shipments(1)
    df.loc[0, "container_count"] = 0
    issues = check_shipments(df, {"SGSIN", "NLRTM"})
    names = {i.check_name for i in issues}
    assert "shipments.zero_container_count" in names


def test_clean_data_produces_no_critical_issues():
    df = _base_shipments(5)
    issues = check_shipments(df, {"SGSIN", "NLRTM"})
    assert not any(i.severity == "critical" for i in issues)


def test_missing_country_detected_in_ports():
    ports = pd.DataFrame(
        {
            "port_code": ["SGSIN", "NLRTM"],
            "port_name": ["Singapore", "Rotterdam"],
            "country": ["Singapore", None],
            "region": ["APAC", "EUR"],
            "timezone": ["UTC+8", "UTC+1"],
            "avg_congestion_score": [0.7, 0.6],
        }
    )
    issues = check_ports(ports)
    names = {i.check_name for i in issues}
    assert "ports.missing_country" in names


def test_arrival_status_conflict_detected():
    df = _base_shipments(1)
    df.loc[0, "actual_arrival"] = None
    df.loc[0, "status"] = "DELIVERED"
    issues = check_shipments(df, {"SGSIN", "NLRTM"})
    names = {i.check_name for i in issues}
    assert "shipments.missing_actual_arrival_unexplained" in names
