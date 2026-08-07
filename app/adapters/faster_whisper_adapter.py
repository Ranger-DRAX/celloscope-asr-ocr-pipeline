from faster_whisper import WhisperModel

from app.adapters.base import TranscriptionResult
from app.config import settings

import tempfile
import os


class FasterWhisperAdapter:
    """Production adapter — the only file in the repo allowed to import faster_whisper.

    The model is loaded once in __init__ (not per-request) and reused for
    the lifetime of the process.
    """

    def __init__(self):
        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None) -> TranscriptionResult:
        suffix = os.path.splitext(filename)[1]
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
        finally:
            os.unlink(tmp_path)

        duration = info.duration
        detected = info.language  # 'bn' / 'en' / whatever whisper detected

        has_speech, text = self._resolve_speech(segments)

        return TranscriptionResult(
            transcript=text,
            detected_language=detected if has_speech else None,
            duration_seconds=round(duration, 2),
            provider="faster-whisper-small",
            has_speech=has_speech,
        )

    @staticmethod
    def _resolve_speech(segments) -> tuple[bool, str]:
        """Combine no_speech_prob and avg_logprob to decide if audio contains real speech.

        Silence/ambient noise typically shows high no_speech_prob AND very low avg_logprob.
        Using both thresholds together is more reliable than either alone.
        """
        if not segments:
            return False, ""
        avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        if avg_no_speech >= settings.no_speech_prob_threshold and avg_logprob <= settings.avg_logprob_threshold:
            return False, ""
        return True, " ".join(s.text.strip() for s in segments).strip()
