"""
Standalone data quality checker.

Runs directly against the raw CSV files -- it does NOT depend on the
ingestion pipeline or the data store. This is intentional: we want to be
able to verify data quality *before* anything is loaded anywhere.

Usage:
    python dq_check.py --input ../data/raw --output ../reports/dq_report.json

Exit code is always 0 (this is a reporting tool, not a gate), unless the
input files themselves cannot be read.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

VALID_STATUSES = {"DELIVERED", "DELAYED", "CANCELLED", "COMPLETED"}
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
# Container counts above this are treated as implausible for a single booking
CONTAINER_COUNT_OUTLIER_THRESHOLD = 500


@dataclass
class Issue:
    check_name: str
    description: str
    severity: str  # critical | warning | info
    rows_affected: int
    total_rows: int
    pct_affected: float
    recommended_action: str
    example_row_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "description": self.description,
            "severity": self.severity,
            "rows_affected": self.rows_affected,
            "total_rows": self.total_rows,
            "pct_affected": round(self.pct_affected, 3),
            "recommended_action": self.recommended_action,
            "example_row_ids": self.example_row_ids[:5],
        }


def _pct(n: int, total: int) -> float:
    return (n / total * 100.0) if total else 0.0


def _examples(df: pd.DataFrame, mask: pd.Series, id_col: str) -> list[str]:
    if id_col not in df.columns:
        return []
    return df.loc[mask, id_col].astype(str).head(5).tolist()


def check_ports(ports: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    total = len(ports)

    missing_country = ports["country"].isna()
    if missing_country.any():
        issues.append(
            Issue(
                check_name="ports.missing_country",
                description="Port reference rows with a null country field.",
                severity="warning",
                rows_affected=int(missing_country.sum()),
                total_rows=total,
                pct_affected=_pct(int(missing_country.sum()), total),
                recommended_action=(
                    "Flag and backfill manually -- reference data is small enough to "
                    "fix by hand. Does not block ingestion since country is not used "
                    "in any derived calculation yet."
                ),
                example_row_ids=_examples(ports, missing_country, "port_code"),
            )
        )

    dup_codes = ports["port_code"].duplicated()
    if dup_codes.any():
        issues.append(
            Issue(
                check_name="ports.duplicate_port_code",
                description="Duplicate port_code values in reference data.",
                severity="critical",
                rows_affected=int(dup_codes.sum()),
                total_rows=total,
                pct_affected=_pct(int(dup_codes.sum()), total),
                recommended_action="Drop duplicates, keep first occurrence.",
                example_row_ids=_examples(ports, dup_codes, "port_code"),
            )
        )
    return issues


def check_shipments(ship: pd.DataFrame, valid_ports: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    total = len(ship)

    # 1. Duplicate shipment_id (same booking appearing more than once)
    dup_id_mask = ship["shipment_id"].duplicated(keep=False)
    if dup_id_mask.any():
        n_dupe_rows = int(dup_id_mask.sum())
        issues.append(
            Issue(
                check_name="shipments.duplicate_shipment_id",
                description=(
                    "shipment_id appears more than once. Some are exact duplicate "
                    "rows (safe to drop); others have the same id with conflicting "
                    "values (e.g. different weight_tons), which indicates a "
                    "re-booking or upstream key collision."
                ),
                severity="critical",
                rows_affected=n_dupe_rows,
                total_rows=total,
                pct_affected=_pct(n_dupe_rows, total),
                recommended_action=(
                    "Drop exact duplicate rows. For id collisions with conflicting "
                    "values, quarantine both rows for manual review rather than "
                    "guessing which is correct."
                ),
                example_row_ids=_examples(ship, dup_id_mask, "shipment_id"),
            )
        )

    # 2. Exact full-row duplicates
    exact_dupe_mask = ship.duplicated(keep="first")
    if exact_dupe_mask.any():
        issues.append(
            Issue(
                check_name="shipments.exact_duplicate_rows",
                description="Fully identical rows (every column matches another row).",
                severity="warning",
                rows_affected=int(exact_dupe_mask.sum()),
                total_rows=total,
                pct_affected=_pct(int(exact_dupe_mask.sum()), total),
                recommended_action="Drop, keeping the first occurrence.",
                example_row_ids=_examples(ship, exact_dupe_mask, "shipment_id"),
            )
        )

    # 3. Invalid port codes (origin/destination not in reference data)
    bad_origin = ~ship["origin_port"].isin(valid_ports)
    bad_dest = ~ship["destination_port"].isin(valid_ports)
    bad_port_mask = bad_origin | bad_dest
    if bad_port_mask.any():
        issues.append(
            Issue(
                check_name="shipments.invalid_port_code",
                description=(
                    "origin_port or destination_port is not a known port code "
                    "(sentinel/placeholder values such as ZZZZZ, XXTST, UNKNW found)."
                ),
                severity="critical",
                rows_affected=int(bad_port_mask.sum()),
                total_rows=total,
                pct_affected=_pct(int(bad_port_mask.sum()), total),
                recommended_action=(
                    "Quarantine -- cannot compute route stats without a valid "
                    "origin/destination. Excluded from the curated layer."
                ),
                example_row_ids=_examples(ship, bad_port_mask, "shipment_id"),
            )
        )

    # 4. Missing actual_departure / actual_arrival that is NOT explained by status
    status_norm = ship["status"].astype(str).str.upper().str.strip()
    in_flight_like = status_norm.isin({"CANCELLED", "COMPLETED", "COMPLETE", "NAN"})
    missing_arrival = ship["actual_arrival"].isna() & ~in_flight_like
    if missing_arrival.any():
        issues.append(
            Issue(
                check_name="shipments.missing_actual_arrival_unexplained",
                description=(
                    "actual_arrival is null for a shipment whose status is "
                    "DELIVERED/DELAYED (i.e. not cancelled), which is contradictory."
                ),
                severity="critical",
                rows_affected=int(missing_arrival.sum()),
                total_rows=total,
                pct_affected=_pct(int(missing_arrival.sum()), total),
                recommended_action=(
                    "Flag as data-integrity conflict; exclude from on-time/delay "
                    "aggregates until resolved, but keep the row (it is still a "
                    "real booking)."
                ),
                example_row_ids=_examples(ship, missing_arrival, "shipment_id"),
            )
        )

    # 5. Missing cargo_type
    missing_cargo = ship["cargo_type"].isna()
    if missing_cargo.any():
        issues.append(
            Issue(
                check_name="shipments.missing_cargo_type",
                description="cargo_type is null.",
                severity="info",
                rows_affected=int(missing_cargo.sum()),
                total_rows=total,
                pct_affected=_pct(int(missing_cargo.sum()), total),
                recommended_action="Fix: impute as 'Unknown'. Not used as a leakage-risk field.",
                example_row_ids=_examples(ship, missing_cargo, "shipment_id"),
            )
        )

    # 6. Inconsistent cargo_type casing / whitespace (same value, different spelling)
    cargo_stripped = ship["cargo_type"].dropna().astype(str).str.strip().str.lower()
    needs_norm = cargo_stripped.map(lambda v: v in CARGO_CANONICAL) & (
        ship.loc[cargo_stripped.index, "cargo_type"].astype(str) != cargo_stripped.map(CARGO_CANONICAL)
    )
    if needs_norm.any():
        issues.append(
            Issue(
                check_name="shipments.inconsistent_cargo_type_casing",
                description=(
                    "cargo_type has casing/whitespace variants of the same value "
                    "(e.g. 'FURNITURE', 'electronics', ' Chemicals') that should "
                    "collapse to one canonical label."
                ),
                severity="warning",
                rows_affected=int(needs_norm.sum()),
                total_rows=total,
                pct_affected=_pct(int(needs_norm.sum()), total),
                recommended_action="Fix: normalise to Title Case canonical set at ingestion time.",
                example_row_ids=_examples(ship, needs_norm.reindex(ship.index, fill_value=False), "shipment_id"),
            )
        )

    # 7. Missing status
    missing_status = ship["status"].isna()
    if missing_status.any():
        issues.append(
            Issue(
                check_name="shipments.missing_status",
                description="status is null.",
                severity="warning",
                rows_affected=int(missing_status.sum()),
                total_rows=total,
                pct_affected=_pct(int(missing_status.sum()), total),
                recommended_action=(
                    "Flag + derive: infer DELIVERED/DELAYED from actual_arrival vs "
                    "planned_arrival when dates are present, else quarantine."
                ),
                example_row_ids=_examples(ship, missing_status, "shipment_id"),
            )
        )

    # 8. Inconsistent status values (casing / synonyms)
    non_canonical_status = ship["status"].notna() & ~ship["status"].isin(
        ["DELIVERED", "DELAYED", "CANCELLED"]
    )
    if non_canonical_status.any():
        issues.append(
            Issue(
                check_name="shipments.inconsistent_status_labels",
                description=(
                    "status uses non-canonical labels/casing: 'COMPLETED', "
                    "'Complete', 'delivered' instead of the canonical "
                    "DELIVERED/DELAYED/CANCELLED."
                ),
                severity="warning",
                rows_affected=int(non_canonical_status.sum()),
                total_rows=total,
                pct_affected=_pct(int(non_canonical_status.sum()), total),
                recommended_action=(
                    "Fix: map COMPLETED/Complete -> DELIVERED, uppercase everything, "
                    "done once in the curated layer."
                ),
                example_row_ids=_examples(ship, non_canonical_status, "shipment_id"),
            )
        )

    # 9. Missing weight_tons
    missing_weight = ship["weight_tons"].isna()
    if missing_weight.any():
        issues.append(
            Issue(
                check_name="shipments.missing_weight_tons",
                description="weight_tons is null.",
                severity="info",
                rows_affected=int(missing_weight.sum()),
                total_rows=total,
                pct_affected=_pct(int(missing_weight.sum()), total),
                recommended_action="Flag; impute with per-cargo-type median in the curated layer.",
                example_row_ids=_examples(ship, missing_weight, "shipment_id"),
            )
        )

    # 10. Negative weight_tons (physically impossible)
    negative_weight = ship["weight_tons"] < 0
    if negative_weight.any():
        issues.append(
            Issue(
                check_name="shipments.negative_weight_tons",
                description="weight_tons is negative, which is physically impossible.",
                severity="critical",
                rows_affected=int(negative_weight.sum()),
                total_rows=total,
                pct_affected=_pct(int(negative_weight.sum()), total),
                recommended_action=(
                    "Fix: take absolute value (most likely a sign-flip bug "
                    "upstream) and flag the row as corrected."
                ),
                example_row_ids=_examples(ship, negative_weight, "shipment_id"),
            )
        )

    # 11. Zero container_count (a shipment cannot have zero containers)
    zero_containers = ship["container_count"] == 0
    if zero_containers.any():
        issues.append(
            Issue(
                check_name="shipments.zero_container_count",
                description="container_count is 0, which is not a valid booking.",
                severity="critical",
                rows_affected=int(zero_containers.sum()),
                total_rows=total,
                pct_affected=_pct(int(zero_containers.sum()), total),
                recommended_action="Quarantine for manual review; excluded from curated layer.",
                example_row_ids=_examples(ship, zero_containers, "shipment_id"),
            )
        )

    # 12. Implausible container_count outliers (likely fat-fingered extra digit)
    outlier_containers = ship["container_count"] > CONTAINER_COUNT_OUTLIER_THRESHOLD
    if outlier_containers.any():
        issues.append(
            Issue(
                check_name="shipments.container_count_outlier",
                description=(
                    f"container_count > {CONTAINER_COUNT_OUTLIER_THRESHOLD}, far "
                    "outside the observed distribution (median ~21, p99 ~70)."
                ),
                severity="warning",
                rows_affected=int(outlier_containers.sum()),
                total_rows=total,
                pct_affected=_pct(int(outlier_containers.sum()), total),
                recommended_action=(
                    "Flag as statistical outlier; keep the row but exclude from "
                    "features that are sensitive to scale until reviewed."
                ),
                example_row_ids=_examples(ship, outlier_containers, "shipment_id"),
            )
        )

    return issues


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run(input_dir: Path) -> dict[str, Any]:
    ports_path = input_dir / "ports.csv"
    ship_path = input_dir / "shipments.csv"

    ports = load_csv(ports_path)
    ship = load_csv(ship_path)

    issues = check_ports(ports) + check_shipments(ship, set(ports["port_code"]))

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: (severity_order.get(i.severity, 9), -i.rows_affected))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "datasets": {
            "ports.csv": {"rows": int(len(ports)), "columns": list(ports.columns)},
            "shipments.csv": {"rows": int(len(ship)), "columns": list(ship.columns)},
        },
        "summary": {
            "total_checks": len(issues),
            "critical": sum(1 for i in issues if i.severity == "critical"),
            "warning": sum(1 for i in issues if i.severity == "warning"),
            "info": sum(1 for i in issues if i.severity == "info"),
        },
        "issues": [i.to_dict() for i in issues],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone raw-data quality checker.")
    parser.add_argument(
        "--input", default="../data/raw", help="Directory containing ports.csv and shipments.csv"
    )
    parser.add_argument(
        "--output", default="../reports/dq_report.json", help="Where to write the JSON report"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = run(input_dir)
    except Exception as exc:  # pragma: no cover - top level guard
        print(f"DQ check failed: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(json.dumps(report, indent=2))
    print(f"DQ report written to {output_path}")
    print(
        f"Checks: {report['summary']['total_checks']} "
        f"(critical={report['summary']['critical']}, "
        f"warning={report['summary']['warning']}, "
        f"info={report['summary']['info']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
