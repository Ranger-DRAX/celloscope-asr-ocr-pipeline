from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    transcript: str
    detected_language: str | None
    duration_seconds: float
    provider: str
    has_speech: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
