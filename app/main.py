import logging
import os

from fastapi import FastAPI

from app.api.routes_transcribe import router as transcribe_router
from app.api.routes_documents import router as documents_router

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
        "Endpoints:\n"
        "- /api/v1/transcribe: Two-stage ASR pipeline\n"
        "- /api/v1/documents/extract: Lab report extraction"
    ),
    version="2.0.0",
)
app.include_router(transcribe_router)
app.include_router(documents_router)

logger.info("Celloscope AI Service started — endpoints are ready")
