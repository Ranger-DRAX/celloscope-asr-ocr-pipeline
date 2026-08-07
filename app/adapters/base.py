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
