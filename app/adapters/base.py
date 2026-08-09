from typing import Protocol, Literal
from dataclasses import dataclass

Lang = Literal["bn", "en"]


@dataclass
class TranscriptionResult:
    transcript: str
    detected_language: Lang | None   # None only in a genuine no-speech case
    duration_seconds: float
    provider: str
    has_speech: bool


class TranscriptionAdapter(Protocol):
    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        """language is 'bn', 'en', or None (meaning 'auto')."""
        ...


# ── Language detection types ──────────────────────────────────────────────────

@dataclass
class LanguageDetectionResult:
    """Raw output from a language detection call.

    detected_language is the raw code returned by the provider (e.g. "en",
    "bn", "hi", "es"). It is NOT yet the routing language — call
    resolve_routing_language() in app/services/language_routing.py to get
    the binary "en" | "bn" routing decision.
    """
    detected_language: str           # raw provider code, e.g. "en", "hi", "es"
    confidence: float | None         # provider-reported confidence, if available
    raw_provider_response: dict | None
    provider: str                    # e.g. "groq", "local_fallback"


class LanguageDetectionAdapter(Protocol):
    def detect(self, audio_bytes: bytes, filename: str) -> LanguageDetectionResult:
        """Detect the spoken language in the audio.

        Raises LanguageDetectionError (from app.adapters.language_detection_base)
        on genuine failures: network error, timeout, malformed response, or
        missing/empty language field. Does NOT raise just because the detected
        language is unexpected.
        """
        ...
