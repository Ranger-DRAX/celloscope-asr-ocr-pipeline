import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.api.schemas_transcribe import TranscribeResponse, ErrorResponse
from app.services.transcribe_service import (
    validate_audio,
    run_transcription,
    UnsupportedFormatError,
    FileTooLargeError,
)
from app.config import settings
from app.adapters.mock_transcribe_adapter import MockTranscribeAdapter

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton adapter instance
_adapter_instance = None


def get_adapter():
    """Return the configured transcription adapter."""
    global _adapter_instance
    if _adapter_instance is not None:
        return _adapter_instance

    if settings.transcribe_provider == "faster_whisper":
        from app.adapters.faster_whisper_adapter import FasterWhisperAdapter
        _adapter_instance = FasterWhisperAdapter()
    else:
        _adapter_instance = MockTranscribeAdapter(settings.mock_responses_dir)
    return _adapter_instance


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
    except UnsupportedFormatError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_format", "detail": str(e)},
        )
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "file_too_large", "detail": str(e)},
        )

    try:
        adapter = get_adapter()
        result = run_transcription(adapter, audio_bytes, audio.filename or "unknown.wav", language)
    except Exception as e:
        logger.error(f"Transcription execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "transcription_failed", "detail": f"Transcription engine error: {e}"},
        )

    return TranscribeResponse(
        transcript=result.transcript,
        detected_language=result.detected_language,
        duration_seconds=result.duration_seconds,
        provider=result.provider,
        has_speech=result.has_speech,
    )
