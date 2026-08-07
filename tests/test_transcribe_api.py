"""Integration tests for POST /api/v1/transcribe — mock provider via TestClient."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import routes_transcribe
from app.config import settings


ROOT = Path(__file__).resolve().parents[1]
BN_FIXTURE_DIR = ROOT / "testdata" / "audio_BN"

@pytest.fixture(autouse=True)
def _setup_mock_provider():
    """Ensure mock provider is used for API integration tests."""
    original_provider = settings.transcribe_provider
    settings.transcribe_provider = "mock"
    routes_transcribe._adapter_instance = None
    yield
    settings.transcribe_provider = original_provider
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

    def test_bengali_fixture_transcribes_expected_text(self):
        """A real Bengali fixture should transcribe to its paired .txt content."""
        if not os.environ.get("WHISPER_MODEL_BN"):
            pytest.skip("WHISPER_MODEL_BN is required for the Bengali fixture regression test")

        audio_path = BN_FIXTURE_DIR / "sample_2047.mp3"
        expected_text = (BN_FIXTURE_DIR / "sample_2047.txt").read_text(encoding="utf-8").strip()

        routes_transcribe._adapter_instance = None
        original_provider = settings.transcribe_provider
        original_bn_model = settings.whisper_model_bn
        settings.transcribe_provider = "faster_whisper"
        settings.whisper_model_bn = os.environ.get("WHISPER_MODEL_BN")

        try:
            adapter = routes_transcribe.get_adapter()
        finally:
            settings.transcribe_provider = original_provider
            settings.whisper_model_bn = original_bn_model

        audio_bytes = audio_path.read_bytes()
        result = adapter.transcribe(audio_bytes, audio_path.name, language="bn")

        assert result.has_speech is True
        assert result.detected_language == "bn"
        assert result.transcript.strip() == expected_text


# ---------------------------------------------------------------------------
# LID Router Unit Tests
# All tests mock both LidClassifier.classify and WhisperModel.transcribe so
# they run instantly without any model downloads or GPU hardware.
# ---------------------------------------------------------------------------

class TestLidRouter:
    """Unit tests for the LID-first routing logic inside FasterWhisperAdapter.

    Strategy: patch LidClassifier.classify (SpeechBrain) and WhisperModel
    (faster-whisper) so routing decisions can be asserted without real models.
    """

    # ------------------------------------------------------------------
    # Shared mock helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mock_segment(text="hello", no_speech_prob=0.05, avg_logprob=-0.3):
        """Return a mock Whisper segment object."""
        seg = type("Seg", (), {
            "text": text,
            "no_speech_prob": no_speech_prob,
            "avg_logprob": avg_logprob,
        })()
        return seg

    @staticmethod
    def _make_mock_info(language="en", duration=2.0):
        """Return a mock Whisper TranscriptionInfo object."""
        return type("Info", (), {"language": language, "duration": duration})()

    @staticmethod
    def _make_adapter_with_mocks(
        lid_result: tuple[str, float],
        whisper_language: str = "en",
        has_bn_model: bool = True,
    ):
        """Construct a FasterWhisperAdapter with all heavy deps mocked."""
        from unittest.mock import MagicMock, patch

        mock_segment = TestLidRouter._make_mock_segment(
            text="test text",
            no_speech_prob=0.05,
            avg_logprob=-0.3,
        )
        mock_info = TestLidRouter._make_mock_info(language=whisper_language)

        # Build a mock WhisperModel that returns one segment.
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([mock_segment], mock_info)

        with (
            patch("app.adapters.faster_whisper_adapter.WhisperModel", return_value=mock_whisper),
            patch("app.adapters.faster_whisper_adapter.LidClassifier") as mock_lid_cls,
        ):
            mock_lid_instance = MagicMock()
            mock_lid_instance.classify.return_value = lid_result
            mock_lid_cls.return_value = mock_lid_instance

            adapter = __import__(
                "app.adapters.faster_whisper_adapter",
                fromlist=["FasterWhisperAdapter"],
            ).FasterWhisperAdapter()

            # Replace _bn_model reference based on test scenario.
            if has_bn_model:
                adapter._bn_model = MagicMock()
                adapter._bn_model.transcribe.return_value = ([mock_segment], mock_info)
            else:
                adapter._bn_model = None

            # Expose the primary model mock for assertion.
            adapter._primary_mock = mock_whisper

        return adapter

    # ------------------------------------------------------------------
    # Routing tests
    # ------------------------------------------------------------------

    def test_auto_bn_high_confidence_routes_to_bn_model(self):
        """LID returns Bengali at 0.95 (>= 0.7 threshold) -> bn model is used."""
        from unittest.mock import patch
        import tempfile, os

        adapter = self._make_adapter_with_mocks(lid_result=("bn", 0.95), whisper_language="bn")

        # Dummy audio bytes (won't be decoded — Whisper is mocked)
        audio = b"FAKE_AUDIO"
        result = adapter.transcribe(audio, "test.wav", language=None)

        # The Bengali model should have been called.
        adapter._bn_model.transcribe.assert_called_once()
        call_kwargs = adapter._bn_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "bn"
        assert result.has_speech is True

    def test_auto_en_routes_to_primary_model(self):
        """LID returns English -> primary model is used, not bn model."""
        adapter = self._make_adapter_with_mocks(lid_result=("en", 0.98), whisper_language="en")

        result = adapter.transcribe(b"FAKE_AUDIO", "test.wav", language=None)

        adapter._primary_mock.transcribe.assert_called_once()
        adapter._bn_model.transcribe.assert_not_called()
        assert result.has_speech is True

    def test_auto_bn_low_confidence_falls_back_to_primary(self):
        """LID detects Bengali at 0.4 (below 0.7 threshold) -> primary model used."""
        adapter = self._make_adapter_with_mocks(lid_result=("bn", 0.4), whisper_language="en")

        result = adapter.transcribe(b"FAKE_AUDIO", "test.wav", language=None)

        # Low confidence: primary model used, Bengali model never called.
        adapter._primary_mock.transcribe.assert_called_once()
        adapter._bn_model.transcribe.assert_not_called()

    def test_explicit_bn_bypasses_lid(self):
        """Explicit language='bn' routes to bn model without calling LID."""
        adapter = self._make_adapter_with_mocks(lid_result=("en", 0.99), whisper_language="bn")

        adapter.transcribe(b"FAKE_AUDIO", "test.wav", language="bn")

        # LID should NOT have been called.
        adapter._lid.classify.assert_not_called()
        # Bengali model should have been called.
        adapter._bn_model.transcribe.assert_called_once()
        call_kwargs = adapter._bn_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "bn"

    def test_explicit_en_bypasses_lid(self):
        """Explicit language='en' routes to primary model without calling LID."""
        adapter = self._make_adapter_with_mocks(lid_result=("bn", 0.99), whisper_language="en")

        adapter.transcribe(b"FAKE_AUDIO", "test.wav", language="en")

        adapter._lid.classify.assert_not_called()
        adapter._primary_mock.transcribe.assert_called_once()
        adapter._bn_model.transcribe.assert_not_called()

    def test_bn_route_fallback_no_bn_model(self):
        """Bengali LID result but no bn model configured -> primary model + prompt."""
        adapter = self._make_adapter_with_mocks(
            lid_result=("bn", 0.95),
            whisper_language="bn",
            has_bn_model=False,
        )
        # Ensure _bn_model is None (already set by _make_adapter_with_mocks)
        assert adapter._bn_model is None

        result = adapter.transcribe(b"FAKE_AUDIO", "test.wav", language=None)

        # Primary model must have been used with language="bn" and initial_prompt.
        adapter._primary_mock.transcribe.assert_called_once()
        call_kwargs = adapter._primary_mock.transcribe.call_args[1]
        assert call_kwargs.get("language") == "bn"
        assert call_kwargs.get("initial_prompt") is not None
        assert result.has_speech is True
