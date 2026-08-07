from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    transcribe_provider: Literal["mock", "faster_whisper"] = "faster_whisper"

    whisper_model: str = "small"
    whisper_device: Literal["cuda", "cpu"] = "cuda"          # Default to CUDA (GPU)
    whisper_compute_type: str = "default"                    # 'default' safely selects float32/int8 per GPU

    max_upload_mb: int = 25
    allowed_audio_formats: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0

    mock_responses_dir: str = "testdata/mock_responses"


settings = Settings()
