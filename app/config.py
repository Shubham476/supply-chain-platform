"""Central configuration. Reads from environment variables with sane
defaults so the service runs out of the box with `docker compose up`."""
from __future__ import annotations

import os
from pathlib import Path

APP_VERSION = "0.1.0"

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.environ.get("WAREHOUSE_DB_PATH", BASE_DIR / "data" / "warehouse.duckdb"))
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", BASE_DIR / "data" / "raw"))
DQ_REPORT_PATH = Path(os.environ.get("DQ_REPORT_PATH", BASE_DIR / "reports" / "dq_report.json"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "ml" / "model.joblib"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
