"""Shared pytest fixtures for the Celloscope AI Service test suite.

All fixtures in this file are automatically available to every test module
in the tests/ directory without any explicit import.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services import transcribe_service as svc


@pytest.fixture(autouse=True)
def use_mock_provider():
    """Force the mock transcription provider for every test.

    Resets both the legacy ``_adapter_instance`` in routes_transcribe and the
    per-language singletons in transcribe_service so that tests never
    accidentally load a real GPU model.
    """
    from app.api import routes_transcribe

    original_provider = settings.transcribe_provider
    settings.transcribe_provider = "mock"
    routes_transcribe._adapter_instance = None
    svc._reset_adapter_singletons()

    yield

    settings.transcribe_provider = original_provider
    routes_transcribe._adapter_instance = None
    svc._reset_adapter_singletons()


@pytest.fixture()
def client() -> TestClient:
    """FastAPI TestClient bound to the main app."""
    return TestClient(app)
