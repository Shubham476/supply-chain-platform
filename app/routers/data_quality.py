from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.config import DQ_REPORT_PATH

router = APIRouter(tags=["data-quality"])


@router.get("/data-quality/report")
def data_quality_report() -> dict:
    if not DQ_REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "DQ report not found. Run `python dq/dq_check.py` to generate "
                f"{DQ_REPORT_PATH}."
            ),
        )
    return json.loads(DQ_REPORT_PATH.read_text())
