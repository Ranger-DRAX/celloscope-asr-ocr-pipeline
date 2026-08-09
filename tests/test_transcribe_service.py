"""Unit tests for the transcription service — mock adapter only."""

from app.adapters.base import TranscriptionResult
from app.services.transcribe_service import run_transcription


class FakeAdapter:
    """Captures the language argument so tests can inspect it."""

    def __init__(self):
        self.last_language = "NOT_CALLED"

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        self.last_language = language
        return TranscriptionResult(
            transcript="fake",
            detected_language="en",
            duration_seconds=1.0,
            provider="fake",
            has_speech=True,
        )


class TestRunTranscription:
    def test_auto_passes_none_to_adapter(self):
        """language='auto' must be converted to None for the adapter."""
        adapter = FakeAdapter()
        run_transcription(adapter, b"data", "test.wav", language="auto")
        assert adapter.last_language is None

    def test_bn_passes_through(self):
        """language='bn' passes through unchanged."""
        adapter = FakeAdapter()
        run_transcription(adapter, b"data", "test.wav", language="bn")
        assert adapter.last_language == "bn"

    def test_en_passes_through(self):
        """language='en' passes through unchanged."""
        adapter = FakeAdapter()
        run_transcription(adapter, b"data", "test.wav", language="en")
        assert adapter.last_language == "en"
