"""Centralized error-response helpers for the API layer.

Maps domain exceptions (from app/services/) to FastAPI HTTPExceptions with
structured JSON bodies. The service layer must never import from this module —
it only raises domain exceptions. The API layer (routes_transcribe.py) calls
raise_for_domain_error() to convert them to the appropriate HTTP response.

Adding a new error:
    1. Define a domain exception in app/services/transcribe_service.py.
    2. Add a branch to raise_for_domain_error() below.
    3. Add the new status code to the `responses` dict in the route decorator.
"""

from fastapi import HTTPException

from app.services.transcribe_service import UnsupportedFormatError, FileTooLargeError
from app.adapters.language_detection_base import LanguageDetectionFatalError


def raise_for_domain_error(exc: Exception) -> None:
    """Convert a domain exception to an HTTPException and raise it.

    Call this inside an ``except Exception`` block in a route handler.
    If ``exc`` is a known domain exception it is mapped to the appropriate
    HTTP status and structured JSON body. Unknown exceptions are re-raised
    as-is so they bubble up to FastAPI''s default 500 handler.

    Args:
        exc: The caught exception.

    Raises:
        HTTPException: always (for known domain exceptions).
        The original exception: for unknown types (re-raised).
    """
    if isinstance(exc, UnsupportedFormatError):
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_format", "detail": str(exc)},
        )

    if isinstance(exc, FileTooLargeError):
        raise HTTPException(
            status_code=400,
            detail={"error": "file_too_large", "detail": str(exc)},
        )

    if isinstance(exc, LanguageDetectionFatalError):
        # Both Groq and the local fallback failed — extremely rare.
        raise HTTPException(
            status_code=500,
            detail={
                "error": "language_detection_failed",
                "detail": (
                    "Language detection failed using both the remote (Groq) "
                    "and local fallback detectors. Transcription aborted."
                ),
            },
        )

    # Unknown exception — let FastAPI''s default 500 handler deal with it.
    raise exc
