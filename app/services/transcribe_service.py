"""Two-stage transcription orchestration.

Pipeline:
    audio + language
        |
        v
    validate_audio()
        |
        v
    language == "auto"?
        YES -> GroqLanguageDetectorAdapter (with hard timeout)
                 -> on LanguageDetectionError: fall back to local Whisper auto-detect
        NO  -> skip detection; routing_language = language
        |
        v
    resolve_routing_language(raw_detected_language)
        -> "en"  : primary whisper-small model
        -> "bn"  : faster-whisper-bangla-small-int8 model
        |
        v
    get_transcription_adapter_for_language(routing_language)
        -> lazy singleton; only the needed model is loaded (4GB-VRAM budget)
        |
        v
    adapter.transcribe()
        |
        v
    PipelineResult (TranscriptionResult + language_detected_by + raw_detected_language)

Layer constraints:
    - Zero FastAPI imports in this module (no HTTPException, UploadFile, etc.)
    - Raise domain exceptions; let app/api/errors.py map them to HTTP codes.
    - LanguageDetectionError from Groq -> warning log + graceful fallback.
    - Never 500 merely because Groq is slow or down.
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from app.adapters.base import TranscriptionAdapter, TranscriptionResult
from app.adapters.language_detection_base import (
    LanguageDetectionError,
    LanguageDetectionFatalError,
)
from app.services.language_routing import resolve_routing_language

logger = logging.getLogger(__name__)


# ── Domain exceptions ─────────────────────────────────────────────────────────

class UnsupportedFormatError(Exception):
    """Audio file has an unsupported extension."""


class FileTooLargeError(Exception):
    """Audio file exceeds the configured maximum upload size."""


# ── Allowed extensions (kept here so validate_audio is self-contained) ────────
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Extended result that wraps TranscriptionResult with detection metadata.

    Fields:
        transcript:            The transcribed text (empty string if no speech).
        detected_language:     The routing language actually used ("en" or "bn").
                               This is what model produced the transcript.
        duration_seconds:      Audio duration in seconds.
        provider:              Adapter label (e.g. "faster-whisper-small (cuda)").
        has_speech:            False if audio is silence/ambient noise.
        language_detected_by:  Who decided the language:
                                 "user_specified" — caller passed "bn" or "en" explicitly.
                                 "groq"           — Groq API detected the language.
                                 "local_fallback" — Groq failed; local Whisper auto-detect was used.
        raw_detected_language: The raw code the detector reported BEFORE routing,
                               e.g. "hi", "es". None when language_detected_by is
                               "user_specified" (no detection happened).
    """
    transcript: str
    detected_language: str | None
    duration_seconds: float
    provider: str
    has_speech: bool
    language_detected_by: Literal["user_specified", "groq", "local_fallback"]
    raw_detected_language: str | None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_audio(filename: str, size_bytes: int, max_mb: int) -> None:
    """Validate filename extension and file size.

    Raises:
        UnsupportedFormatError: if the extension is not in ALLOWED_EXTENSIONS.
        FileTooLargeError:      if size_bytes > max_mb * 1024 * 1024.

    No FastAPI types here — this is pure business logic.
    """
    ext = filename.lower().rsplit(".", 1)
    ext = "." + ext[-1] if len(ext) == 2 else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported audio format: ''{ext or 'unknown'}''")
    if size_bytes > max_mb * 1024 * 1024:
        raise FileTooLargeError(f"File exceeds {max_mb}MB limit")


# ── Adapter singletons (lazy, one per language) ────────────────────────────────

_adapter_en: Optional[TranscriptionAdapter] = None
_adapter_bn: Optional[TranscriptionAdapter] = None


def get_transcription_adapter_for_language(lang: Literal["en", "bn"]) -> TranscriptionAdapter:
    """Return the transcription adapter for the given routing language.

    Adapters are loaded lazily on first use per language. Only the adapter
    that is actually needed for this request is loaded, which avoids
    simultaneously holding two large models in GPU memory (4GB-VRAM budget).

    Note: This factory always returns a FasterWhisperAdapter. The "mock"
    provider path is handled by routes_transcribe.py, which short-circuits
    before reaching this function.

    Args:
        lang: "en" -> primary whisper-small adapter.
              "bn" -> Bengali fine-tuned model adapter.

    Returns:
        A TranscriptionAdapter ready to call .transcribe().
    """
    from app.config import settings
    from app.adapters.faster_whisper_adapter import FasterWhisperAdapter

    global _adapter_en, _adapter_bn

    if lang == "en":
        if _adapter_en is None:
            logger.info("Lazy-loading English transcription adapter (whisper-small)")
            _adapter_en = FasterWhisperAdapter(model_override=None)  # uses settings.whisper_model
        return _adapter_en

    # lang == "bn"
    if _adapter_bn is None:
        logger.info(
            f"Lazy-loading Bengali transcription adapter ({settings.whisper_model_bn})"
        )
        _adapter_bn = FasterWhisperAdapter(model_override=settings.whisper_model_bn)
    return _adapter_bn


def _reset_adapter_singletons() -> None:
    """Reset cached adapter singletons. Used in tests to swap providers."""
    global _adapter_en, _adapter_bn
    _adapter_en = None
    _adapter_bn = None


# ── Main orchestration ────────────────────────────────────────────────────────

def run_transcription(
    audio_bytes: bytes,
    filename: str,
    language: str,          # "auto", "en", or "bn" — already validated by the API layer
) -> PipelineResult:
    """Run the full two-stage detect-then-transcribe pipeline.

    Stage 1 (language == "auto" only):
        Try Groq language detection. On LanguageDetectionError, fall back to
        local Whisper auto-detect. Log a warning but never raise; Groq being
        down must not fail the request.

    Stage 2:
        Apply resolve_routing_language() to get a binary routing decision,
        select the correct local adapter, and run transcription.

    Args:
        audio_bytes:  Raw audio bytes, already size-validated.
        filename:     Original filename (used for extension + temp file suffix).
        language:     "auto", "en", or "bn".

    Returns:
        PipelineResult with transcription output and detection provenance.

    Raises:
        LanguageDetectionFatalError: only when BOTH Groq and the local fallback
            have failed (extremely unlikely in practice).
        UnsupportedFormatError: propagated from validate_audio().
        FileTooLargeError: propagated from validate_audio().
    """
    from app.config import settings

    # ── Stage 1: language detection ──────────────────────────────────────────
    if language != "auto":
        # Caller specified the language explicitly — skip detection entirely.
        # Never call Groq; saves cost and latency.
        routing_language = language                     # already "en" or "bn"
        language_detected_by: Literal["user_specified", "groq", "local_fallback"] = "user_specified"
        raw_detected_language: Optional[str] = None

    else:
        # "auto" path: try Groq, fall back to local Whisper auto-detect.
        raw_detected_language = None
        language_detected_by = "local_fallback"  # will be overwritten on Groq success

        # Try Groq if it is enabled and a key is configured.
        groq_attempted = (
            settings.enable_remote_language_detection
            and bool(settings.groq_api_key)
        )

        if groq_attempted:
            try:
                from app.adapters.groq_language_detector import GroqLanguageDetectorAdapter
                groq = GroqLanguageDetectorAdapter(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                    timeout=settings.language_detection_timeout_seconds,
                    sample_seconds=settings.language_detection_sample_seconds,
                )
                det = groq.detect(audio_bytes, filename)
                raw_detected_language = det.detected_language
                language_detected_by = "groq"
                logger.info(
                    f"Groq detected language=''{raw_detected_language}'' "
                    f"for ''{filename}''"
                )
            except LanguageDetectionError as exc:
                logger.warning(
                    f"Groq language detection failed ({exc}); "
                    "falling back to local Whisper auto-detect"
                )
                # groq_attempted remains True but language_detected_by stays "local_fallback"
        else:
            reason = (
                "GROQ_API_KEY not set"
                if not settings.groq_api_key
                else "ENABLE_REMOTE_LANGUAGE_DETECTION=false"
            )
            logger.debug(f"Skipping Groq language detection ({reason})")

        # Local fallback: detect language with the best available means.
        # When transcribe_provider=="mock" (test / dev path), use the mock
        # adapter with language=None — it returns a sensible detected_language
        # from the fixture JSON.  In production use the real FasterWhisperAdapter
        # with auto-detect (language=None).
        if language_detected_by == "local_fallback":
            try:
                from app.config import settings as cfg
                if cfg.transcribe_provider == "mock":
                    from app.adapters.mock_transcribe_adapter import MockTranscribeAdapter
                    local_adapter = MockTranscribeAdapter(cfg.mock_responses_dir)
                    local_result = local_adapter.transcribe(audio_bytes, filename, language=None)
                    raw_detected_language = local_result.detected_language  # from fixture JSON
                    language_detected_by = "local_fallback"
                    logger.debug(
                        f"Mock adapter auto-detected language=''{raw_detected_language}'' "
                        f"for ''{filename}''"
                    )
                else:
                    from app.adapters.faster_whisper_adapter import FasterWhisperAdapter
                    local_detector = FasterWhisperAdapter(model_override=None)
                    local_result = local_detector.transcribe(audio_bytes, filename, language=None)
                    raw_detected_language = local_result.detected_language  # e.g. "en", "bn"
                    language_detected_by = "local_fallback"
                    logger.info(
                        f"Local Whisper auto-detected language=''{raw_detected_language}'' "
                        f"for ''{filename}''"
                    )
            except Exception as exc:
                if groq_attempted:
                    raise LanguageDetectionFatalError(
                        f"Both Groq and local fallback failed for ''{filename}''. "
                        f"Local error: {exc}"
                    ) from exc
                raise

        # Apply the routing rule to the raw detected code.
        routing_language = resolve_routing_language(raw_detected_language or "")

    # ── Stage 2: transcription ───────────────────────────────────────────────
    from app.config import settings as cfg

    if cfg.transcribe_provider == "mock":
        from app.adapters.mock_transcribe_adapter import MockTranscribeAdapter
        adapter: TranscriptionAdapter = MockTranscribeAdapter(cfg.mock_responses_dir)
        # Always pass routing_language so the mock adapter can key fixtures
        # correctly (e.g. en_clean_01 → detected_language="en").
        result: TranscriptionResult = adapter.transcribe(
            audio_bytes, filename, language=routing_language
        )
    else:
        adapter = get_transcription_adapter_for_language(routing_language)
        result = adapter.transcribe(
            audio_bytes,
            filename,
            language=routing_language,   # always explicit at this point
        )

    return PipelineResult(
        transcript=result.transcript,
        detected_language=routing_language if result.has_speech else None,
        duration_seconds=result.duration_seconds,
        provider=result.provider,
        has_speech=result.has_speech,
        language_detected_by=language_detected_by,
        raw_detected_language=raw_detected_language,
    )
