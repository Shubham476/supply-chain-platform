from __future__ import annotations

from fastapi import APIRouter

from app.config import APP_VERSION, DB_PATH

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    db_available = DB_PATH.exists()
    return {
        "status": "ok" if db_available else "degraded",
        "version": APP_VERSION,
        "database_available": db_available,
    }
