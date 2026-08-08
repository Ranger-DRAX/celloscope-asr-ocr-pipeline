from typing import Literal
from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    transcript: str
    # Routing language that was actually used for transcription: "en" or "bn".
    # Repurposed (still backward-compatible): tells the caller which model
    # produced this transcript. null when has_speech is false.
    detected_language: str | None
    duration_seconds: float
    provider: str
    has_speech: bool
    # Who decided the language for this request.
    #   "user_specified"  — caller passed language="bn" or "en" explicitly.
    #   "groq"            — Groq API detected the language from a short audio sample.
    #   "local_fallback"  — Groq was unavailable; local Whisper auto-detect was used.
    language_detected_by: Literal["user_specified", "groq", "local_fallback"]
    # The raw language code the detector reported BEFORE the routing rule was
    # applied. e.g. "hi" or "es" — languages that the routing rule maps to "bn".
    # null when language_detected_by == "user_specified" (no detection happened).
    raw_detected_language: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
