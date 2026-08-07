from app.adapters.base import TranscriptionAdapter, TranscriptionResult


class UnsupportedFormatError(Exception):
    ...


class FileTooLargeError(Exception):
    ...


ALLOWED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def validate_audio(filename: str, size_bytes: int, max_mb: int) -> None:
    """Validate filename extension and file size.

    Raises UnsupportedFormatError or FileTooLargeError on invalid input.
    No FastAPI types here — this is pure business logic.
    """
    ext = filename.lower().rsplit(".", 1)
    ext = "." + ext[-1] if len(ext) == 2 else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported audio format: '{ext or 'unknown'}'")
    if size_bytes > max_mb * 1024 * 1024:
        raise FileTooLargeError(f"File exceeds {max_mb}MB limit")


def run_transcription(
    adapter: TranscriptionAdapter,
    audio_bytes: bytes,
    filename: str,
    language: str | None,
) -> TranscriptionResult:
    """Orchestrate a single transcription call.

    Converts language="auto" to None (whisper's auto-detect) before
    forwarding to the adapter.
    """
    lang = None if language == "auto" else language
    return adapter.transcribe(audio_bytes, filename, lang)
