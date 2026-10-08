"""Unit tests for ML feature engineering.

The most important test here is `test_no_leakage_columns`: features must
never be derived from actual_departure/actual_arrival.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from features import ALL_FEATURES, build_features, build_label  # noqa: E402


def _shipments():
    return pd.DataFrame(
        {
            "shipment_id": ["SHP-1"],
            "origin_port": ["SGSIN"],
            "destination_port": ["NLRTM"],
            "cargo_type": ["Electronics"],
            "container_count": [10],
            "weight_tons": [500.0],
            "booking_date": pd.to_datetime(["2024-01-01"]),
            "planned_departure": pd.to_datetime(["2024-01-05"]),
            "planned_arrival": pd.to_datetime(["2024-01-20"]),
            "actual_delay_hours": [48.0],
        }
    )


def _ports():
    return pd.DataFrame(
        {
            "port_code": ["SGSIN", "NLRTM"],
            "avg_congestion_score": [0.7, 0.6],
        }
    )


def test_no_leakage_columns():
    forbidden = {"actual_departure", "actual_arrival"}
    assert forbidden.isdisjoint(set(ALL_FEATURES))


def test_build_features_returns_expected_columns():
    feats = build_features(_shipments(), _ports())
    for col in ALL_FEATURES:
        assert col in feats.columns


def test_transit_days_planned_computed_correctly():
    feats = build_features(_shipments(), _ports())
    assert feats["transit_days_planned"].iloc[0] == 15.0


def test_congestion_scores_mapped():
    feats = build_features(_shipments(), _ports())
    assert feats["origin_congestion_score"].iloc[0] == 0.7
    assert feats["destination_congestion_score"].iloc[0] == 0.6


def test_build_label_threshold():
    labels = build_label(_shipments(), on_time_threshold_hours=24.0)
    assert labels.iloc[0] == 1  # 48h > 24h threshold -> late


def test_build_label_none_when_delay_unknown():
    df = _shipments()
    df["actual_delay_hours"] = [None]
    labels = build_label(df)
    assert labels.iloc[0] is None
