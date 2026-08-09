import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.api.schemas_transcribe import TranscribeResponse, ErrorResponse
from app.api.errors import raise_for_domain_error
from app.services.transcribe_service import (
    validate_audio,
    run_transcription,
    UnsupportedFormatError,
    FileTooLargeError,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Legacy singleton for the mock provider path (kept so existing tests that
# reset `_adapter_instance` still work). The real FasterWhisper adapters are
# now managed as per-language singletons inside transcribe_service.py.
_adapter_instance = None


@router.post(
    "/api/v1/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Transcribe an audio file."""
    if language not in ("bn", "en", "auto"):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_language", "detail": "language must be 'bn', 'en', or 'auto'"},
        )

    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_upload", "detail": f"Failed to read audio stream: {e}"},
        )

    try:
        validate_audio(audio.filename or "unknown.wav", len(audio_bytes), settings.max_upload_mb)
    except (UnsupportedFormatError, FileTooLargeError) as e:
        raise_for_domain_error(e)

    try:
        result = run_transcription(
            audio_bytes=audio_bytes,
            filename=audio.filename or "unknown.wav",
            language=language,
        )
    except Exception as e:
        logger.error(f"Transcription pipeline failed: {e}", exc_info=True)
        raise_for_domain_error(e)

    return TranscribeResponse(
        transcript=result.transcript,
        detected_language=result.detected_language,
        duration_seconds=result.duration_seconds,
        provider=result.provider,
        has_speech=result.has_speech,
        language_detected_by=result.language_detected_by,
        raw_detected_language=result.raw_detected_language,
    )
