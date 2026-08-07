from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    transcribe_provider: Literal["mock", "faster_whisper"] = "faster_whisper"

    whisper_model: str = "small"
    whisper_model_bn: str | None = None
    whisper_device: Literal["cuda", "cpu"] = "cuda"          # Default to CUDA (GPU)
    whisper_compute_type: str = "default"                    # 'default' safely selects float32/int8 per GPU

    max_upload_mb: int = 25
    allowed_audio_formats: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0

    # Language-ID (LID) router settings.
    # lid_model: HuggingFace repo or local path for the SpeechBrain LID model.
    # lid_bn_threshold: minimum LID confidence to route to the Bengali model.
    #   Below this score the general Whisper model is used instead.
    # lid_savedir: directory where SpeechBrain caches the downloaded model.
    lid_model: str = "speechbrain/lang-id-voxlingua107-ecapa"
    lid_bn_threshold: float = 0.7
    lid_savedir: str = "pretrained_models"

    mock_responses_dir: str = "testdata/mock_responses"


settings = Settings()
