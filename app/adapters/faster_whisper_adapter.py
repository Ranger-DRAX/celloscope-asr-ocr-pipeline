import os
import sys
import tempfile
import logging

logger = logging.getLogger(__name__)

# Ensure CUDA DLLs from nvidia site-packages are registered on Windows
if sys.platform == "win32":
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    nvidia_bins = [
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(site_packages, "nvidia", "cuda_nvrtc", "bin"),
    ]
    for b in nvidia_bins:
        if os.path.exists(b):
            os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(b)
            except Exception:
                pass

from faster_whisper import WhisperModel
from app.adapters.base import TranscriptionAdapter, TranscriptionResult
from app.config import settings
from app.services.transcribe_service import UnsupportedFormatError


class FasterWhisperAdapter:
    """Production adapter — uses faster-whisper on CUDA (GPU) or CPU.

    The model is loaded once in __init__ and reused for all requests.
    """

    def __init__(self):
        device = settings.whisper_device
        compute_type = settings.whisper_compute_type
        logger.info(f"Initializing FasterWhisperAdapter (device={device}, compute_type={compute_type}, model={settings.whisper_model})")

        try:
            self._model = WhisperModel(
                settings.whisper_model,
                device=device,
                compute_type=compute_type,
            )
            self._device = device
        except Exception as e:
            if device == "cuda":
                logger.warning(f"Failed to initialize WhisperModel on CUDA ({e}). Falling back to CPU.")
                self._model = WhisperModel(
                    settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
                self._device = "cpu"
            else:
                raise e

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name

        try:
            segments, info = self._model.transcribe(
                tmp_path,
                language=language,          # None => auto-detect
                task="transcribe",          # NEVER "translate" — keep spoken language as-is
                vad_filter=True,            # trims leading/trailing silence
            )
            segments = list(segments)
        except Exception as e:
            err_msg = str(e)
            if "Invalid data found" in err_msg or "InvalidDataError" in err_msg or "error decoding" in err_msg.lower():
                raise UnsupportedFormatError(f"Corrupt or unreadable audio file: {filename}")
            raise e
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        duration = info.duration
        detected = info.language  # 'bn' / 'en' / etc.

        has_speech, text = self._resolve_speech(segments)

        return TranscriptionResult(
            transcript=text,
            detected_language=detected if has_speech else None,
            duration_seconds=round(duration, 2),
            provider=f"faster-whisper-{settings.whisper_model} ({self._device})",
            has_speech=has_speech,
        )

    @staticmethod
    def _resolve_speech(segments) -> tuple[bool, str]:
        """Combine no_speech_prob and avg_logprob to decide if audio contains real speech."""
        if not segments:
            return False, ""
        avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        if avg_no_speech >= settings.no_speech_prob_threshold and avg_logprob <= settings.avg_logprob_threshold:
            return False, ""
        return True, " ".join(s.text.strip() for s in segments).strip()
