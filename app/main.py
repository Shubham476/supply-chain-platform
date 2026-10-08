from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import APP_VERSION, LOG_LEVEL
from app.logging_config import RequestLoggingMiddleware, configure_logging
from app.routers import data_quality, health, predict, routes, shipments

configure_logging(LOG_LEVEL)
logger = logging.getLogger("app")

app = FastAPI(
    title="Supply Chain Intelligence Mini-Platform",
    version=APP_VERSION,
    description="REST API for shipment lookups, route stats, delay-risk prediction, and data quality reporting.",
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(health.router)
app.include_router(shipments.router)
app.include_router(routes.router)
app.include_router(predict.router)
app.include_router(data_quality.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": exc.errors(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )
