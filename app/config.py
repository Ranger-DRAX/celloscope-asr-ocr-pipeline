from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Transcription provider ────────────────────────────────────────────────
    transcribe_provider: Literal["mock", "faster_whisper"] = "faster_whisper"

    # ── Local Whisper models ──────────────────────────────────────────────────
    whisper_model: str = "small"
    # Path or HuggingFace model ID for the Bengali-fine-tuned model.
    # Set to a local CT2-converted checkpoint, e.g.:
    #   pretrained_models/faster-whisper-bangla-small-int8
    whisper_model_bn: str | None = None
    whisper_device: Literal["cuda", "cpu"] = "cuda"   # Default to CUDA (GPU)
    whisper_compute_type: str = "default"              # 'default' selects float32/int8 per GPU

    # ── Audio validation ──────────────────────────────────────────────────────
    max_upload_mb: int = 25
    allowed_audio_formats: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    # ── No-speech detection thresholds ────────────────────────────────────────
    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0

    # ── Mock provider ─────────────────────────────────────────────────────────
    mock_responses_dir: str = "testdata/mock_responses"
    document_fixtures_dir: str = "testdata/fixtures/documents"

    # ── Document OCR provider ────────────────────────────────────────────────
    document_extraction_provider: Literal["mock", "mistral_ocr"] = "mock"
    
    # ── Mistral API Key ──────────────────────────────────────────────────────
    mistral_api_key: str | None = None

    # ── Groq language detection (Stage 1 of the two-stage pipeline) ───────────
    # API key for Groq Cloud. Leave unset or empty to disable remote detection
    # and fall back to local Whisper auto-detect for every request.
    # NEVER commit a real key -- keep this in .env (already gitignored).
    groq_api_key: str | None = None

    # Groq Whisper model used for language detection only (not transcription).
    # whisper-large-v3-turbo gives accurate language ID at low latency.
    groq_model: str = "whisper-large-v3-turbo"

    # Set to False to skip Groq entirely and always use local auto-detect.
    # Useful for air-gapped environments or when you want to save API quota.
    enable_remote_language_detection: bool = True

    # Hard timeout (seconds) for the Groq HTTP call. A slow/down Groq endpoint
    # must never block the transcription pipeline -- the local fallback kicks in
    # automatically once this deadline is exceeded.
    language_detection_timeout_seconds: float = 8.0

    # Duration (seconds) of audio sent to Groq for detection. Only this short
    # sample is transmitted -- the full audio stays local and is never sent to
    # any third party. Shorter = faster + cheaper; 12 s is sufficient for
    # confident language identification in practice.
    language_detection_sample_seconds: float = 12.0


settings = Settings()
