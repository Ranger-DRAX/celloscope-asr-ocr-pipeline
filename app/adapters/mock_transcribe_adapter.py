import json
import os

from app.adapters.base import TranscriptionResult


class MockTranscribeAdapter:
    """Loads pre-recorded fixture JSON files instead of running a real model.

    Falls back to a default mock response if no specific fixture matches
    the input filename stem.
    """

    def __init__(self, fixtures_dir: str):
        self.fixtures_dir = fixtures_dir

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        key = self._key_for(audio_bytes, filename)
        path = os.path.join(self.fixtures_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return TranscriptionResult(**data)
        
        # Graceful fallback mock response if no exact fixture exists
        detected_lang = language if language in ("bn", "en") else "en"
        return TranscriptionResult(
            transcript=f"Mock transcription output for '{filename}'",
            detected_language=detected_lang,
            duration_seconds=3.5,
            provider="mock",
            has_speech=True,
        )

    @staticmethod
    def _key_for(audio_bytes: bytes, filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]
