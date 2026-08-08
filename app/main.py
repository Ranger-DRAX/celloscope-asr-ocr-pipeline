import logging
import os

from fastapi import FastAPI

from app.api.routes_transcribe import router as transcribe_router

# Configure root logger.
# Level is read from the LOG_LEVEL environment variable (default: INFO).
# Set LOG_LEVEL=DEBUG for verbose adapter-level logs during development.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Celloscope AI Service",
    description=(
        "Two-stage ASR pipeline: "
        "language detection → local faster-whisper transcription."
    ),
    version="1.0.0",
)
app.include_router(transcribe_router)

logger.info("Celloscope AI Service started — POST /api/v1/transcribe is ready")
