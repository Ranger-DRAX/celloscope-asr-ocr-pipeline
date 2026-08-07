from pydantic import BaseModel
from typing import Literal


class TranscribeResponse(BaseModel):
    transcript: str
    detected_language: Literal["bn", "en"] | None
    duration_seconds: float
    provider: str
    has_speech: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
