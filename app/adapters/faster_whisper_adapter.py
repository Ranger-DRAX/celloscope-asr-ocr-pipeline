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
from app.adapters.lid_classifier import LidClassifier
from app.config import settings
from app.services.transcribe_service import UnsupportedFormatError


class FasterWhisperAdapter:
    """Production adapter — uses faster-whisper on CUDA (GPU) or CPU.

    Language-ID-first routing
    -------------------------
    When the caller passes ``language='auto'`` (i.e. ``language=None``), a
    lightweight SpeechBrain ECAPA-TDNN model classifies the language *before*
    any Whisper decoding.  The result determines which Whisper model to use:

    * LID detects Bengali with confidence >= ``LID_BN_THRESHOLD`` (default 0.7)
      -> Bengali-tuned model (``WHISPER_MODEL_BN`` env var, e.g.
         ``anuragshas/whisper-small-bn``).
    * Anything else -> primary model (``WHISPER_MODEL``).

    When the caller passes an explicit ``language='bn'`` or ``language='en'``,
    LID is **skipped entirely** — routing is deterministic from the param.

    Fallback safety
    ---------------
    If ``WHISPER_MODEL_BN`` is not configured (or the model fails to load),
    Bengali requests silently fall back to the primary model with a
    Bengali-specific initial prompt.  No operator action required.

    Temp-file lifecycle
    -------------------
    The temp file is created once, used for both the LID pass and the Whisper
    decode pass, then deleted in a single ``finally`` block that wraps the
    entire transcribe body.  This prevents the file-deleted-before-second-pass
    bug that existed in the previous double-decode design.
    """

    def __init__(self):
        device = settings.whisper_device
        compute_type = settings.whisper_compute_type
        logger.info(
            f"Initializing FasterWhisperAdapter (device={device}, "
            f"compute_type={compute_type}, model={settings.whisper_model})"
        )

        self._model, self._device = self._load_model(settings.whisper_model, device, compute_type)

        # Optional Bengali-specific model. Only loaded if configured — keeps
        # the mock/default path untouched and avoids doubling VRAM usage
        # unless the user has explicitly opted in.
        self._bn_model = None
        bn_model_path = getattr(settings, "whisper_model_bn", None)
        if bn_model_path:
            # Resolve to absolute path so faster-whisper's model loader never
            # mistakes a relative directory path for a HuggingFace repo ID.
            # (WhisperModel falls through to HF Hub lookup if os.path.isdir()
            # returns False, which it will for a relative path when the CWD
            # doesn't match the project root.)
            if not os.path.isabs(bn_model_path) and os.path.isdir(bn_model_path):
                bn_model_path = os.path.abspath(bn_model_path)
            logger.info(f"Loading dedicated Bengali model from {bn_model_path}")
            self._bn_model, _ = self._load_model(bn_model_path, device, compute_type)

        self._bn_prompt = (
            "This audio is spoken Bengali. Transcribe it in Bengali script only, "
            "preserving the spoken words as closely as possible."
        )

        # LID classifier — loaded once, reused for every 'auto' request.
        # LidClassifier lazy-imports speechbrain on first instantiation.
        logger.info(
            f"Loading LID classifier ({settings.lid_model}) "
            f"into {device}, savedir={settings.lid_savedir}"
        )
        self._lid = LidClassifier(
            model_source=settings.lid_model,
            savedir=settings.lid_savedir,
            device=device,
        )
        self._lid_threshold = settings.lid_bn_threshold

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

        # Outer finally guarantees temp-file cleanup regardless of which branch
        # executes — including LID errors and Whisper decode errors.
        try:
            active_model, whisper_language, model_label = self._route(tmp_path, language)

            try:
                segments, info = active_model.transcribe(
                    tmp_path,
                    language=whisper_language,         # None => whisper auto-detect (non-bn fallback only)
                    task="transcribe",                  # NEVER "translate" — keep spoken language as-is
                    vad_filter=True,                    # trims leading/trailing silence
                    condition_on_previous_text=False,   # stop one bad segment poisoning the rest
                    beam_size=5,                        # beam search recovers better than greedy on weaker languages
                    initial_prompt=self._bn_prompt if whisper_language == "bn" else None,
                )
                segments = list(segments)
            except Exception as e:
                err_msg = str(e)
                if "Invalid data found" in err_msg or "InvalidDataError" in err_msg or "error decoding" in err_msg.lower():
                    raise UnsupportedFormatError(f"Corrupt or unreadable audio file: {filename}")
                raise e

        finally:
            # Single cleanup point — temp file is guaranteed to still exist here
            # because nothing deleted it earlier (unlike the old design).
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        duration = info.duration
        detected = info.language  # 'bn' / 'en' / etc.

        has_speech, text = self._resolve_speech(segments)

        detected_language = detected if detected in ("bn", "en") else None
        return TranscriptionResult(
            transcript=text,
            detected_language=detected_language if has_speech else None,
            duration_seconds=round(duration, 2),
            provider=f"faster-whisper-{model_label} ({self._device})",
            has_speech=has_speech,
        )

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _route(
        self, audio_path: str, language: str | None
    ) -> tuple[object, str | None, str]:
        """Decide which Whisper model to use and what language param to pass.

        Parameters
        ----------
        audio_path:
            Path to the temp audio file (used for LID classification on auto).
        language:
            The caller-supplied language hint: ``None`` (auto), ``"bn"``, or ``"en"``.

        Returns
        -------
        (active_model, whisper_language, model_label)
            ``active_model`` -- the WhisperModel instance to decode with.
            ``whisper_language`` -- the ``language=`` value to pass to ``.transcribe()``.
            ``model_label`` -- string for the ``provider`` response field.
        """
        if language is None:
            # Auto mode -- run LID first to decide routing.
            return self._route_auto(audio_path)

        if language == "bn":
            # Explicit Bengali -- bypass LID, route directly.
            return self._route_explicit_bn()

        # Explicit non-Bengali language -- primary model.
        return self._model, language, settings.whisper_model

    def _route_auto(self, audio_path: str) -> tuple[object, str | None, str]:
        """LID-first routing for ``language='auto'`` requests."""
        try:
            lang_code, confidence = self._lid.classify(audio_path)
        except Exception as exc:
            logger.warning(
                "LID classification failed (%s) -- falling back to primary model.", exc
            )
            return self._model, None, settings.whisper_model

        logger.info(
            "LID: detected lang=%r confidence=%.3f (threshold=%.2f)",
            lang_code,
            confidence,
            self._lid_threshold,
        )

        if lang_code == "bn" and confidence >= self._lid_threshold:
            return self._route_explicit_bn()

        # Not Bengali (or below threshold) -- let Whisper auto-detect.
        return self._model, None, settings.whisper_model

    def _route_explicit_bn(self) -> tuple[object, str, str]:
        """Route to the Bengali-tuned model, falling back gracefully if unavailable."""
        if self._bn_model is not None:
            label = getattr(settings, "whisper_model_bn", "whisper-bn") or "whisper-bn"
            return self._bn_model, "bn", label

        # No Bengali model configured -- silent fallback to primary with prompt.
        logger.debug(
            "Bengali route requested but WHISPER_MODEL_BN is not configured; "
            "falling back to primary model with Bengali initial prompt."
        )
        return self._model, "bn", settings.whisper_model

    # ------------------------------------------------------------------
    # Speech detection
    # ------------------------------------------------------------------

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