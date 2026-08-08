"""Domain exceptions for language detection adapters.

Kept in a separate module so both the Groq adapter and the local fallback
adapter can import it without creating circular dependencies.
"""


class LanguageDetectionError(Exception):
    """Raised when a language detection adapter encounters a genuine failure.

    Genuine failures include:
    - Network errors or connection timeouts
    - HTTP errors from the remote API (4xx / 5xx)
    - Malformed or unparseable response body
    - Missing or empty ``language`` field in the response

    This is NOT raised merely because the detected language is unexpected,
    unsupported, or non-English. The routing layer (language_routing.py)
    handles that case independently.

    The service layer (transcribe_service.py) catches this exception, logs a
    warning, and falls back to the local Whisper auto-detect path, so a
    LanguageDetectionError must never cause an HTTP 500 on its own.
    """

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


class LanguageDetectionFatalError(Exception):
    """Raised when BOTH Groq detection AND the local fallback have failed.

    This is the only scenario where language detection causes an HTTP 500.
    The API layer (errors.py) maps this to a 500 with a structured JSON body.

    Under normal operating conditions this should never be raised — the local
    fallback uses the already-loaded FasterWhisper model and has no external
    dependencies.
    """
