"""DuckDB connection helper.

DuckDB is single-writer; for a read-mostly API we open a fresh read-only
connection per request (cheap, avoids locking issues) and keep a single
writer connection only for the pipeline/offline jobs.
"""
from __future__ import annotations

import duckdb

from app.config import DB_PATH


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)
