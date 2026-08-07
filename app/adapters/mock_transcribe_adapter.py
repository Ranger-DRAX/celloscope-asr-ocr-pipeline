import json
import os

from app.adapters.base import TranscriptionResult


class MockTranscribeAdapter:
    """Loads pre-recorded fixture JSON files instead of running a real model.

    Used as the default provider so the entire pipeline works end-to-end
    without GPU, model download, or any heavy dependency.
    """

    def __init__(self, fixtures_dir: str):
        self.fixtures_dir = fixtures_dir

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        key = self._key_for(audio_bytes, filename)
        path = os.path.join(self.fixtures_dir, f"{key}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No mock fixture for '{filename}' (key={key})")
        with open(path) as f:
            data = json.load(f)
        return TranscriptionResult(**data)

    @staticmethod
    def _key_for(audio_bytes: bytes, filename: str) -> str:
        # Match by filename stem so real recorded testdata files map 1:1 to fixtures.
        return os.path.splitext(os.path.basename(filename))[0]
