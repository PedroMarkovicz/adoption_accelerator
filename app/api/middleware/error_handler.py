"""Global exception handlers for the Adoption Accelerator API.

Catches known exception types and returns structured JSON error responses
so the Streamlit client always receives a predictable error shape:

    {"error": true, "error_type": "...", "message": "...", "session_id": "..."}
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
    session_id: str = "",
    details: list[dict] | None = None,
) -> JSONResponse:
    """Build a structured JSON error response."""
    body: dict = {
        "error": True,
        "error_type": error_type,
        "message": message,
        "session_id": session_id or str(uuid.uuid4()),
    }
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


async def _handle_runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
    """RuntimeError -- typically model bundle validation failures (503)."""
    logger.error("RuntimeError during %s %s: %s", request.method, request.url.path, exc)
    return _error_response(
        status_code=503,
        error_type="service_unavailable",
        message="The prediction service is temporarily unavailable. Please try again later.",
    )


async def _handle_validation_error(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Pydantic ValidationError -- malformed input (422)."""
    field_errors = []
    for err in exc.errors():
        field_errors.append(
            {
                "field": ".".join(str(loc) for loc in err.get("loc", [])),
                "message": err.get("msg", ""),
                "type": err.get("type", ""),
            }
        )
    logger.warning(
        "ValidationError during %s %s: %d field error(s)",
        request.method,
        request.url.path,
        len(field_errors),
    )
    return _error_response(
        status_code=422,
        error_type="validation_error",
        message="One or more input fields are invalid.",
        details=field_errors,
    )


async def _handle_generic_exception(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for unhandled exceptions during graph execution (500)."""
    logger.exception(
        "Unhandled exception during %s %s", request.method, request.url.path
    )
    return _error_response(
        status_code=500,
        error_type="internal_error",
        message="An unexpected error occurred. Please try again or contact support.",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI app."""
    app.add_exception_handler(RuntimeError, _handle_runtime_error)
    app.add_exception_handler(ValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_generic_exception)
