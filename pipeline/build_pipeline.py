"""
Raw -> Curated -> Serving pipeline.

Layers
------
raw        : exactly what's in data/raw/*.csv, untouched.
curated    : cleaned, typed, deduplicated, with DQ fixes applied and
             derived fields added (actual_delay_hours, on_time_flag,
             route_key, transit_days_planned/actual).
serving    : aggregated, API-friendly views (route stats) built on top
             of curated tables. Cheap to rebuild from curated at any time.

All tables live in a single DuckDB file (data/warehouse.duckdb). The
pipeline is idempotent: every stage does CREATE OR REPLACE TABLE, so
running it N times produces exactly the same result as running it once.

Usage:
    python build_pipeline.py --raw ../data/raw --db ../data/warehouse.duckdb
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

CARGO_CANONICAL = {
    "furniture": "Furniture",
    "automotive parts": "Automotive Parts",
    "machinery": "Machinery",
    "pharmaceuticals": "Pharmaceuticals",
    "electronics": "Electronics",
    "chemicals": "Chemicals",
    "consumer goods": "Consumer Goods",
    "apparel": "Apparel",
    "food & beverage": "Food & Beverage",
    "raw materials": "Raw Materials",
}
STATUS_CANONICAL = {
    "DELIVERED": "DELIVERED",
    "DELAYED": "DELAYED",
    "CANCELLED": "CANCELLED",
    "COMPLETED": "DELIVERED",
    "COMPLETE": "DELIVERED",
}
CONTAINER_COUNT_OUTLIER_THRESHOLD = 500
ON_TIME_THRESHOLD_HOURS = 24


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ports = pd.read_csv(raw_dir / "ports.csv")
    ship = pd.read_csv(raw_dir / "shipments.csv")
    return ports, ship


def clean_ports(ports: pd.DataFrame) -> pd.DataFrame:
    ports = ports.drop_duplicates(subset=["port_code"], keep="first").copy()
    ports["country"] = ports["country"].fillna("Unknown")
    return ports


def clean_shipments(ship: pd.DataFrame, valid_ports: set[str]) -> pd.DataFrame:
    """Apply the DQ decisions documented in dq/dq_check.py to produce a
    curated, analysis-ready shipments table."""
    df = ship.copy()

    # --- Parse dates up front ---
    date_cols = ["booking_date", "planned_departure", "actual_departure", "planned_arrival", "actual_arrival"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # --- 1. Drop exact full-row duplicates ---
    before = len(df)
    df = df.drop_duplicates(keep="first")
    logger.info("Dropped %d exact duplicate rows", before - len(df))

    # --- 2. Quarantine shipment_id collisions with conflicting data ---
    # (exact dupes already removed above; remaining dupes are true conflicts)
    dup_mask = df["shipment_id"].duplicated(keep=False)
    quarantined_dupe_ids = df.loc[dup_mask, "shipment_id"].unique().tolist()
    df = df.loc[~dup_mask].copy()
    logger.info(
        "Quarantined %d rows (%d shipment_ids) with conflicting duplicate keys",
        int(dup_mask.sum()),
        len(quarantined_dupe_ids),
    )

    # --- 3. Quarantine invalid port codes ---
    bad_port_mask = ~df["origin_port"].isin(valid_ports) | ~df["destination_port"].isin(valid_ports)
    logger.info("Quarantined %d rows with invalid port codes", int(bad_port_mask.sum()))
    df = df.loc[~bad_port_mask].copy()

    # --- 4. Quarantine zero container_count (not a valid booking) ---
    zero_container_mask = df["container_count"] == 0
    logger.info("Quarantined %d rows with zero container_count", int(zero_container_mask.sum()))
    df = df.loc[~zero_container_mask].copy()

    # --- 5. Fix negative weight_tons: take absolute value, flag ---
    neg_weight_mask = df["weight_tons"] < 0
    df["weight_corrected"] = False
    df.loc[neg_weight_mask, "weight_tons"] = df.loc[neg_weight_mask, "weight_tons"].abs()
    df.loc[neg_weight_mask, "weight_corrected"] = True
    logger.info("Corrected sign on %d negative weight_tons values", int(neg_weight_mask.sum()))

    # --- 6. Flag (not drop) container_count outliers ---
    df["container_count_outlier"] = df["container_count"] > CONTAINER_COUNT_OUTLIER_THRESHOLD

    # --- 7. Normalise cargo_type casing; impute missing with 'Unknown' ---
    def norm_cargo(v):
        if pd.isna(v):
            return "Unknown"
        key = str(v).strip().lower()
        return CARGO_CANONICAL.get(key, str(v).strip())

    df["cargo_type"] = df["cargo_type"].map(norm_cargo)

    # --- 8. Impute missing weight_tons with per-cargo-type median ---
    df["weight_imputed"] = df["weight_tons"].isna()
    medians = df.groupby("cargo_type")["weight_tons"].transform("median")
    df["weight_tons"] = df["weight_tons"].fillna(medians)
    overall_median = df["weight_tons"].median()
    df["weight_tons"] = df["weight_tons"].fillna(overall_median)

    # --- 9. Normalise status casing/synonyms; infer missing where possible ---
    def norm_status(v):
        if pd.isna(v):
            return None
        return STATUS_CANONICAL.get(str(v).strip().upper(), str(v).strip().upper())

    df["status"] = df["status"].map(norm_status)
    df["status_inferred"] = False
    can_infer = df["status"].isna() & df["actual_arrival"].notna() & df["planned_arrival"].notna()
    delay_hours_tmp = (df.loc[can_infer, "actual_arrival"] - df.loc[can_infer, "planned_arrival"]).dt.total_seconds() / 3600.0
    inferred = delay_hours_tmp.apply(lambda h: "DELAYED" if h > ON_TIME_THRESHOLD_HOURS else "DELIVERED")
    df.loc[can_infer, "status"] = inferred
    df.loc[can_infer, "status_inferred"] = True
    still_missing = df["status"].isna()
    df.loc[still_missing, "status"] = "UNKNOWN"
    logger.info(
        "Inferred status for %d rows from actual vs planned arrival; %d remain UNKNOWN",
        int(can_infer.sum()),
        int(still_missing.sum()),
    )

    # --- 10. Flag (do not drop) rows where status says delivered/delayed but
    # actual_arrival is missing -- a genuine data-integrity conflict ---
    df["arrival_status_conflict"] = df["actual_arrival"].isna() & df["status"].isin(["DELIVERED", "DELAYED"])

    # --- Derived fields (Section 2.3) ---
    df["actual_delay_hours"] = (df["actual_arrival"] - df["planned_arrival"]).dt.total_seconds() / 3600.0
    df["on_time_flag"] = df["actual_delay_hours"].apply(
        lambda h: (h <= ON_TIME_THRESHOLD_HOURS) if pd.notna(h) else None
    )
    df["route_key"] = df["origin_port"] + " \u2192 " + df["destination_port"]
    df["transit_days_planned"] = (df["planned_arrival"] - df["planned_departure"]).dt.total_seconds() / 86400.0
    df["transit_days_actual"] = (df["actual_arrival"] - df["actual_departure"]).dt.total_seconds() / 86400.0

    df = df.reset_index(drop=True)
    return df


def build(raw_dir: Path, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ports_raw, ship_raw = load_raw(raw_dir)

    ports_curated = clean_ports(ports_raw)
    ship_curated = clean_shipments(ship_raw, set(ports_curated["port_code"]))

    con = duckdb.connect(str(db_path))
    try:
        # RAW layer: load untouched, exactly as supplied.
        con.register("ports_raw_df", ports_raw)
        con.register("ship_raw_df", ship_raw)
        con.execute("CREATE OR REPLACE TABLE raw_ports AS SELECT * FROM ports_raw_df")
        con.execute("CREATE OR REPLACE TABLE raw_shipments AS SELECT * FROM ship_raw_df")

        # CURATED layer: cleaned + derived fields.
        con.register("ports_curated_df", ports_curated)
        con.register("ship_curated_df", ship_curated)
        con.execute("CREATE OR REPLACE TABLE curated_ports AS SELECT * FROM ports_curated_df")
        con.execute("CREATE OR REPLACE TABLE curated_shipments AS SELECT * FROM ship_curated_df")

        # SERVING layer: pre-aggregated route stats for the API/assistant.
        con.execute(
            """
            CREATE OR REPLACE TABLE serving_route_stats AS
            SELECT
                origin_port,
                destination_port,
                route_key,
                COUNT(*) AS shipment_count,
                AVG(actual_delay_hours) FILTER (WHERE actual_delay_hours IS NOT NULL) AS avg_delay_hours,
                AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END)
                    FILTER (WHERE on_time_flag IS NOT NULL) AS on_time_rate,
                AVG(transit_days_planned) AS avg_transit_days_planned,
                AVG(transit_days_actual) FILTER (WHERE transit_days_actual IS NOT NULL) AS avg_transit_days_actual
            FROM curated_shipments
            GROUP BY 1, 2, 3
            """
        )

        con.execute(
            """
            CREATE OR REPLACE VIEW serving_shipments AS
            SELECT
                shipment_id, customer_id, origin_port, destination_port, vessel_id,
                booking_date, planned_departure, actual_departure, planned_arrival,
                actual_arrival, container_count, cargo_type, weight_tons, status,
                actual_delay_hours, on_time_flag, route_key,
                transit_days_planned, transit_days_actual,
                weight_corrected, weight_imputed, status_inferred,
                container_count_outlier, arrival_status_conflict
            FROM curated_shipments
            """
        )

        n_raw = con.execute("SELECT COUNT(*) FROM raw_shipments").fetchone()[0]
        n_curated = con.execute("SELECT COUNT(*) FROM curated_shipments").fetchone()[0]
        n_routes = con.execute("SELECT COUNT(*) FROM serving_route_stats").fetchone()[0]
        logger.info(
            "Pipeline complete: raw_shipments=%d curated_shipments=%d serving_route_stats=%d",
            n_raw,
            n_curated,
            n_routes,
        )
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the raw -> curated -> serving pipeline.")
    parser.add_argument("--raw", default="../data/raw")
    parser.add_argument("--db", default="../data/warehouse.duckdb")
    args = parser.parse_args()

    build(Path(args.raw).resolve(), Path(args.db).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
