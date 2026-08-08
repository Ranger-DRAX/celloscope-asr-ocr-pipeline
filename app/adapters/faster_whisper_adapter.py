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

    The primary model is loaded once in __init__ and reused for all requests.

    Bengali note: generic multilingual Whisper checkpoints are meaningfully
    weaker on Bengali than English (low-resource language in training data).
    Two mitigations are applied:
      1. Decoding params tuned for weaker languages (see transcribe() below) —
         condition_on_previous_text=False prevents one bad segment from
         poisoning the rest of the transcript; beam_size=5 gives the decoder
         more chance to recover from an early wrong token.
      2. Optional second model for Bengali specifically, activated by setting
         WHISPER_MODEL_BN to a local CT2-converted, Bengali-finetuned
         checkpoint path (e.g. a converted version of
         bangla-speech-processing/whisper_small_bn). If unset, all languages
         use the primary model.
    """

    def __init__(self, model_override: str | None = None):
        """Load a single Whisper model.

        Args:
            model_override: If provided, load this specific model path/name
                instead of settings.whisper_model. Used by the two-stage
                pipeline factory (get_transcription_adapter_for_language) to
                instantiate one adapter per language without loading both
                models simultaneously (4GB-VRAM budget).
                Pass None to use the default settings.whisper_model.
        """
        device = settings.whisper_device
        compute_type = settings.whisper_compute_type
        model_name = model_override if model_override is not None else settings.whisper_model
        logger.info(
            f"Initializing FasterWhisperAdapter (device={device}, "
            f"compute_type={compute_type}, model={model_name})"
        )

        self._model, self._device = self._load_model(model_name, device, compute_type)
        self._model_name = model_name

        # _bn_model is kept for the legacy single-adapter path (model_override=None
        # and whisper_model_bn is set). In the two-stage pipeline, the service
        # layer creates two separate adapter instances instead, so this is only
        # loaded when both conditions are true.
        self._bn_model = None
        if model_override is None:
            bn_model_path = getattr(settings, "whisper_model_bn", None)
            if bn_model_path:
                logger.info(f"Loading dedicated Bengali model from {bn_model_path}")
                self._bn_model, _ = self._load_model(bn_model_path, device, compute_type)
        self._bn_prompt = (
            "This audio is spoken Bengali. Transcribe it in Bengali script only, "
            "preserving the spoken words as closely as possible."
        )

    def _load_model(self, model_name: str, device: str, compute_type: str):
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            return model, device
        except Exception as e:
            if device == "cuda":
                logger.warning(f"Failed to initialize '{model_name}' on CUDA ({e}). Falling back to CPU.")
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
                return model, "cpu"
            raise e

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name

        # Route to the Bengali-specific model only when the caller explicitly
        # asked for 'bn' and a dedicated model is configured. In 'auto' mode
        # we always use the primary model first, since we don't know the
        # language yet — see note below on re-running detection for 'auto'.
        active_model = self._bn_model if (language == "bn" and self._bn_model is not None) else self._model
        transcript_model = active_model

        try:
            segments, info = active_model.transcribe(
                tmp_path,
                language=language,                  # None => auto-detect
                task="transcribe",                   # NEVER "translate" — keep spoken language as-is
                vad_filter=True,                      # trims leading/trailing silence
                condition_on_previous_text=False,    # stop one bad segment poisoning the rest — key fix for Bengali
                beam_size=5,                          # beam search recovers better than greedy on weaker languages
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

        # If 'auto' detected Bengali but we ran on the primary (non-Bengali-
        # tuned) model, and a dedicated bn model is configured, re-run once
        # on that model for better quality. Small latency cost, only on the
        # auto+Bengali path.
        if language is None and detected == "bn" and self._bn_model is not None and active_model is not self._bn_model:
            transcript_model = self._bn_model
            segments, info = transcript_model.transcribe(
                tmp_path,
                language="bn",
                task="transcribe",
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
                initial_prompt=self._bn_prompt,
            )
            segments = list(segments)
            duration = info.duration
            detected = info.language

        has_speech, text = self._resolve_speech(segments)

        if transcript_model is self._bn_model:
            model_label = settings.whisper_model_bn
        else:
            model_label = self._model_name
        detected_language = detected if detected in ("bn", "en") else None
        return TranscriptionResult(
            transcript=text,
            detected_language=detected_language if has_speech else None,
            duration_seconds=round(duration, 2),
            provider=f"faster-whisper-{model_label} ({self._device})",
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