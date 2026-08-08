"""Consolidated test suite for POST /api/v1/transcribe.

Covers (per agent.md spec, line 156: "validate the test in one file which
gives the output for one endpoint"):
  - validate_audio: good/bad extensions, size limits
  - resolve_routing_language: full truth-table
  - Two-stage pipeline: Groq mocked via unittest.mock patching only the
    Groq adapter module (patch target: app.adapters.language_detector.httpx.Client)
  - Fallback behaviour on Groq timeout / error
  - End-to-end API via FastAPI TestClient with mock transcription adapter
  - New response fields: language_detected_by, raw_detected_language
  - Silence / no-speech (has_speech=false, HTTP 200)
  - Invalid format / oversized file / invalid language (400 errors)
  - Explicit language bypass (Groq is never called)

Run:
    pytest tests/test_transcribe_api.py -v
"""

import json
import unittest.mock
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.transcribe_service import (
    validate_audio,
    UnsupportedFormatError,
    FileTooLargeError,
)
from app.services.language_routing import resolve_routing_language
from app.adapters.language_detection_base import LanguageDetectionError
from app.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "testdata" / "mock_responses"

# NOTE: The `use_mock_provider` fixture in conftest.py automatically forces
# settings.transcribe_provider = "mock" for every test in this file.
client = TestClient(app)

# Patch target: httpx.Client *inside* the Groq adapter module only.
# This avoids accidentally patching the TestClient`s own httpx calls.
GROQ_HTTPX_CLIENT = "app.adapters.language_detector.httpx.Client"


def _post(filename: str, content: bytes = b"fake-audio", language: str = "auto", ctype: str = "audio/wav"):
    """Helper: POST to /api/v1/transcribe."""
    return client.post(
        "/api/v1/transcribe",
        files={"audio": (filename, content, ctype)},
        data={"language": language},
    )


def _make_groq_response(language: str, status_code: int = 200) -> MagicMock:
    """Build a fake httpx.Response for Groq transcription calls."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "task": "transcribe",
        "language": language,
        "duration": 3.5,
        "text": "discarded transcript text",
        "segments": [],
    }
    mock_resp.text = json.dumps(mock_resp.json.return_value)
    return mock_resp


def _groq_client_ctx(language: str = "en", status_code: int = 200):
    """Return a patch context manager that mocks the Groq httpx.Client.

    Usage:
        with _groq_client_ctx("bn") as mock_client:
            resp = _post("clip.wav", language="auto")
    """
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ctx.post = MagicMock(return_value=_make_groq_response(language, status_code))
    return patch(GROQ_HTTPX_CLIENT, return_value=mock_ctx)


# ===========================================================================
# 1. validate_audio - unit tests (no HTTP, no adapters)
# ===========================================================================

class TestValidateAudio:
    def test_valid_wav(self):
        validate_audio("clip.wav", 100, 25)  # should not raise

    def test_valid_mp3(self):
        validate_audio("clip.mp3", 100, 25)

    def test_valid_m4a(self):
        validate_audio("clip.m4a", 100, 25)

    def test_valid_flac(self):
        validate_audio("clip.flac", 100, 25)

    def test_valid_ogg(self):
        validate_audio("clip.ogg", 100, 25)

    def test_unsupported_txt(self):
        with pytest.raises(UnsupportedFormatError):
            validate_audio("notes.txt", 100, 25)

    def test_unsupported_mp4(self):
        with pytest.raises(UnsupportedFormatError):
            validate_audio("video.mp4", 100, 25)

    def test_no_extension(self):
        with pytest.raises(UnsupportedFormatError):
            validate_audio("noext", 100, 25)

    def test_size_exactly_at_limit(self):
        validate_audio("clip.wav", 25 * 1024 * 1024, 25)  # should not raise

    def test_size_over_limit(self):
        with pytest.raises(FileTooLargeError):
            validate_audio("clip.wav", 25 * 1024 * 1024 + 1, 25)


# ===========================================================================
# 2. resolve_routing_language - full truth-table
# ===========================================================================

class TestResolveRoutingLanguage:
    def test_en_routes_to_en(self):
        assert resolve_routing_language("en") == "en"

    def test_EN_uppercase_routes_to_en(self):
        """Case-insensitive normalisation."""
        assert resolve_routing_language("EN") == "en"

    def test_bn_routes_to_bn(self):
        assert resolve_routing_language("bn") == "bn"

    def test_hi_routes_to_bn(self):
        """Hindi -> Bangla model (default-to-Bangla policy)."""
        assert resolve_routing_language("hi") == "bn"

    def test_es_routes_to_bn(self):
        """Spanish -> Bangla model."""
        assert resolve_routing_language("es") == "bn"

    def test_unknown_routes_to_bn(self):
        assert resolve_routing_language("unknown") == "bn"

    def test_empty_string_routes_to_bn(self):
        assert resolve_routing_language("") == "bn"

    def test_whitespace_routes_to_bn(self):
        assert resolve_routing_language("  ") == "bn"

    def test_random_code_routes_to_bn(self):
        assert resolve_routing_language("xyz") == "bn"


# ===========================================================================
# 3. End-to-end API - success paths (mock transcription adapter)
# ===========================================================================

class TestTranscribeAPISuccess:
    def test_en_explicit_returns_200(self):
        """Explicit English: language_detected_by must be user_specified."""
        resp = _post("en_clean_01.wav", language="en")
        assert resp.status_code == 200
        body = resp.json()
        assert body["detected_language"] == "en"
        assert body["has_speech"] is True
        assert body["transcript"] != ""
        assert body["language_detected_by"] == "user_specified"
        assert body["raw_detected_language"] is None

    def test_bn_explicit_returns_200(self):
        """Explicit Bengali: language_detected_by must be user_specified."""
        resp = _post("bn_clean_01.mp3", language="bn", ctype="audio/mpeg")
        assert resp.status_code == 200
        body = resp.json()
        assert body["detected_language"] == "bn"
        assert body["language_detected_by"] == "user_specified"
        assert body["raw_detected_language"] is None

    def test_auto_without_groq_uses_local_fallback(self):
        """When GROQ_API_KEY is unset, auto-detection falls back locally.
        language_detected_by must be local_fallback."""
        original_key = settings.groq_api_key
        settings.groq_api_key = None
        try:
            resp = _post("en_clean_01.wav", language="auto")
        finally:
            settings.groq_api_key = original_key

        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "local_fallback"

    def test_response_schema_has_all_required_fields(self):
        """All fields present on a standard success response."""
        resp = _post("en_clean_01.wav", language="en")
        body = resp.json()
        required = {
            "transcript", "detected_language", "duration_seconds",
            "provider", "has_speech", "language_detected_by", "raw_detected_language",
        }
        assert required.issubset(body.keys()), f"Missing fields: {required - body.keys()}"


# ===========================================================================
# 4. Silence / no-speech handling
# ===========================================================================

class TestSilenceHandling:
    def test_silence_returns_200_no_speech(self):
        """Silence fixture: HTTP 200, has_speech=false, empty transcript.
        Groq is disabled (no API key) so the mock local fallback is used."""
        original_key = settings.groq_api_key
        settings.groq_api_key = None
        try:
            resp = _post("silence_01.wav", language="auto")
        finally:
            settings.groq_api_key = original_key
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_speech"] is False
        assert body["transcript"] == ""
        assert body["detected_language"] is None


# ===========================================================================
# 5. Validation error paths - 400 responses
# ===========================================================================

class TestValidationErrors:
    def test_unsupported_format_400(self):
        resp = _post("notes.txt", ctype="text/plain")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unsupported_format"

    def test_oversized_file_400(self):
        big = b"x" * (26 * 1024 * 1024)
        resp = _post("big.wav", content=big)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "file_too_large"

    def test_invalid_language_400(self):
        resp = _post("en_clean_01.wav", language="fr")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_language"

    def test_invalid_language_zh_400(self):
        resp = _post("en_clean_01.wav", language="zh")
        assert resp.status_code == 400


# ===========================================================================
# 6. Groq integration - mocked via httpx.Client inside the Groq adapter module
# ===========================================================================

class TestGroqDetectionPipeline:
    def setup_method(self, _):
        settings.groq_api_key = "mock-key-for-tests"
        settings.enable_remote_language_detection = True

    def teardown_method(self, _):
        settings.groq_api_key = None

    def test_groq_detects_en_routes_to_en_model(self):
        """Groq returns en -> routing_language=en, language_detected_by=groq."""
        with _groq_client_ctx("en"):
            resp = _post("en_clean_01.wav", language="auto")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "groq"
        assert body["raw_detected_language"] == "en"
        assert body["detected_language"] == "en"

    def test_groq_detects_bn_routes_to_bn_model(self):
        """Groq returns bn -> routing_language=bn, language_detected_by=groq."""
        with _groq_client_ctx("bn"):
            resp = _post("bn_clean_01.mp3", language="auto", ctype="audio/mpeg")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "groq"
        assert body["raw_detected_language"] == "bn"
        assert body["detected_language"] == "bn"

    def test_groq_detects_hi_routes_to_bn_model(self):
        """Groq returns hi (Hindi) -> routing_language=bn (default-to-Bangla policy)."""
        with _groq_client_ctx("hi"):
            resp = _post("en_clean_01.wav", language="auto")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "groq"
        assert body["raw_detected_language"] == "hi"   # raw code preserved
        assert body["detected_language"] == "bn"       # routing collapsed to bn

    def test_groq_detects_es_routes_to_bn_model(self):
        """Spanish -> Bangla model."""
        with _groq_client_ctx("es"):
            resp = _post("en_clean_01.wav", language="auto")
        body = resp.json()
        assert body["raw_detected_language"] == "es"
        assert body["detected_language"] == "bn"

    def test_explicit_language_bypasses_groq(self):
        """When language=en is explicit, Groq httpx.Client must not be called."""
        with patch(GROQ_HTTPX_CLIENT) as mock_client_cls:
            resp = _post("en_clean_01.wav", language="en")
        # httpx.Client should not have been instantiated at all
        mock_client_cls.assert_not_called()
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "user_specified"


# ===========================================================================
# 7. Groq fallback behaviour - timeout and HTTP error
# ===========================================================================

class TestGroqFallback:
    def setup_method(self, _):
        settings.groq_api_key = "mock-key-for-tests"
        settings.enable_remote_language_detection = True

    def teardown_method(self, _):
        settings.groq_api_key = None

    def _make_error_client_ctx(self, side_effect):
        """Patch httpx.Client inside the Groq adapter to raise side_effect on post()."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.post = MagicMock(side_effect=side_effect)
        return patch(GROQ_HTTPX_CLIENT, return_value=mock_ctx)

    def test_groq_timeout_falls_back_gracefully(self):
        """httpx.TimeoutException -> fall back to local, no 500."""
        import httpx
        with self._make_error_client_ctx(httpx.TimeoutException("timed out")):
            resp = _post("en_clean_01.wav", language="auto")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "local_fallback"

    def test_groq_network_error_falls_back_gracefully(self):
        """httpx.RequestError -> fall back to local, no 500."""
        import httpx
        with self._make_error_client_ctx(httpx.RequestError("conn refused")):
            resp = _post("en_clean_01.wav", language="auto")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "local_fallback"

    def test_groq_http_500_falls_back_gracefully(self):
        """Groq returns HTTP 500 -> LanguageDetectionError -> fall back to local."""
        with _groq_client_ctx("en", status_code=500):
            resp = _post("en_clean_01.wav", language="auto")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "local_fallback"

    def test_groq_missing_language_field_falls_back(self):
        """Groq response missing language field -> LanguageDetectionError -> fallback."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.json.return_value = {"text": "hello", "segments": []}  # no language key
        bad_resp.text = "{}"
        mock_ctx.post = MagicMock(return_value=bad_resp)
        with patch(GROQ_HTTPX_CLIENT, return_value=mock_ctx):
            resp = _post("en_clean_01.wav", language="auto")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language_detected_by"] == "local_fallback"