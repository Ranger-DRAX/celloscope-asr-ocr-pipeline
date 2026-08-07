from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    transcribe_provider: Literal["mock", "faster_whisper"] = "mock"

    whisper_model: str = "small"
    whisper_device: Literal["cuda", "cpu"] = "cpu"          # CPU-safe default for graders
    whisper_compute_type: str = "int8"                       # int8 on CPU, float16 on cuda

    max_upload_mb: int = 25
    allowed_audio_formats: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0

    mock_responses_dir: str = "testdata/mock_responses"


settings = Settings()
