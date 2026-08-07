"""Integration tests for POST /api/v1/transcribe — mock provider via TestClient."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import routes_transcribe


@pytest.fixture(autouse=True)
def _reset_adapter():
    """Reset the singleton adapter before each test to ensure test isolation."""
    routes_transcribe._adapter_instance = None
    yield
    routes_transcribe._adapter_instance = None


client = TestClient(app)


class TestTranscribeAPI:
    def test_valid_en_returns_200(self):
        """POST a valid .wav file matching the en_clean_01 fixture → 200."""
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("en_clean_01.wav", b"fake-audio-bytes", "audio/wav")},
            data={"language": "en"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["detected_language"] == "en"
        assert body["has_speech"] is True
        assert body["transcript"] != ""
        assert "provider" in body
        assert "duration_seconds" in body

    def test_valid_bn_returns_200(self):
        """POST a valid .mp3 file matching the bn_clean_01 fixture → 200."""
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("bn_clean_01.mp3", b"fake-audio-bytes", "audio/mpeg")},
            data={"language": "bn"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["detected_language"] == "bn"
        assert body["has_speech"] is True

    def test_silence_returns_200_with_no_speech(self):
        """POST silence_01.wav → 200, has_speech=false, empty transcript."""
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("silence_01.wav", b"fake-silence", "audio/wav")},
            data={"language": "auto"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["has_speech"] is False
        assert body["transcript"] == ""
        assert body["detected_language"] is None

    def test_unsupported_format_returns_400(self):
        """POST a .txt file → 400 structured error, not a 500 stack trace."""
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("notes.txt", b"not audio", "text/plain")},
            data={"language": "auto"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["error"] == "unsupported_format"

    def test_oversized_file_returns_400(self):
        """POST a file exceeding 25MB → 400 structured error."""
        oversized = b"x" * (26 * 1024 * 1024)
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("big.wav", oversized, "audio/wav")},
            data={"language": "auto"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["error"] == "file_too_large"

    def test_invalid_language_returns_400(self):
        """POST with language='fr' → 400 with invalid_language error."""
        response = client.post(
            "/api/v1/transcribe",
            files={"audio": ("en_clean_01.wav", b"fake", "audio/wav")},
            data={"language": "fr"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["error"] == "invalid_language"
