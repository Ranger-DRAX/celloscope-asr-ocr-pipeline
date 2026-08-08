"""Groq-based language detection adapter.

Implements Stage 1 of the two-stage ASR pipeline: sends a short trimmed
audio sample to Groq''s hosted Whisper endpoint, reads back the ``language``
field, and returns a LanguageDetectionResult.

Key design constraints (from agent.md):
- Only the ``language`` field is used — the transcript is discarded.
- Only a short trimmed sample is sent, not the full audio file.
- A hard httpx timeout is enforced; Groq slowness must never block transcription.
- LanguageDetectionError is raised only on genuine failures (network, timeout,
  malformed response, missing language). It is NOT raised because the detected
  language is non-English — that is handled by resolve_routing_language().
- The adapter does NOT decide routing. It returns the raw provider code.
"""

import io
import logging
import subprocess
import tempfile
import os
from typing import Optional

import httpx

from app.adapters.base import LanguageDetectionResult
from app.adapters.language_detection_base import LanguageDetectionError

logger = logging.getLogger(__name__)

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# ── Language name → ISO 639-1 normalization ───────────────────────────────────
# Groq's verbose_json format returns the full English name of the detected
# language (e.g. "english", "bengali", "hindi") rather than an ISO 639-1
# two-letter code ("en", "bn", "hi"). Without normalization, "english" would
# fail the routing rule's exact "en" check and incorrectly route to the
# Bangla model.
#
# This map covers the languages most likely to appear in Groq responses.
# Unknown names are passed through as-is — resolve_routing_language() will
# default-to-Bangla for any unrecognized code, which is the intended policy.
_LANG_NAME_TO_ISO: dict[str, str] = {
    # English
    "english": "en",
    # Bengali / Bangla
    "bengali": "bn",
    "bangla": "bn",
    # Common South/Southeast Asian languages (routed to Bangla model)
    "hindi": "hi",
    "urdu": "ur",
    "arabic": "ar",
    "persian": "fa",
    "farsi": "fa",
    "punjabi": "pa",
    "gujarati": "gu",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "sinhala": "si",
    "burmese": "my",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    # European / other
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
    "polish": "pl",
    "turkish": "tr",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "swahili": "sw",
}


def _normalize_language_code(raw: str) -> str:
    """Normalize a Groq language string to an ISO 639-1 code.

    Groq returns full language names in verbose_json mode (e.g. "english",
    "bengali"). This function maps those names to their ISO 639-1 equivalents.
    If the input is already an ISO code (2–3 chars) or is unrecognized, it is
    returned as-is — the routing layer handles unknown codes gracefully.

    Args:
        raw: Lowercased, stripped string from the Groq ``language`` field.

    Returns:
        ISO 639-1 code string (e.g. "en", "bn", "hi") or the original string.
    """
    return _LANG_NAME_TO_ISO.get(raw, raw)


def _trim_audio_bytes(
    audio_bytes: bytes,
    filename: str,
    sample_seconds: float,
) -> tuple[bytes, str]:
    """Trim audio to at most ``sample_seconds`` using ffmpeg subprocess.

    Returns (trimmed_bytes, output_filename).

    Falls back to raw bytes if ffmpeg is not available on PATH — Groq will
    still attempt detection from whatever audio data it receives, so this is
    a safe degradation.
    """
    suffix = os.path.splitext(filename)[1] or ".wav"
    out_suffix = ".wav"  # ffmpeg encodes cleanly to wav for Groq

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + out_suffix

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-t", str(sample_seconds),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=15,
        )

        if result.returncode == 0 and os.path.exists(tmp_out_path):
            with open(tmp_out_path, "rb") as f:
                trimmed = f.read()
            logger.debug(
                f"Audio trimmed to {sample_seconds}s via ffmpeg "
                f"({len(audio_bytes)} -> {len(trimmed)} bytes)"
            )
            return trimmed, "sample" + out_suffix
        else:
            logger.warning(
                f"ffmpeg trim failed (rc={result.returncode}); sending raw bytes to Groq"
            )
            return audio_bytes, filename

    except FileNotFoundError:
        # ffmpeg not installed — degrade gracefully
        logger.warning("ffmpeg not found on PATH; sending raw audio bytes to Groq (no trim)")
        return audio_bytes, filename
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg trim timed out; sending raw audio bytes to Groq (no trim)")
        return audio_bytes, filename
    except Exception as exc:
        logger.warning(f"Unexpected ffmpeg error ({exc}); sending raw bytes")
        return audio_bytes, filename
    finally:
        for path in [tmp_in_path, tmp_out_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


class GroqLanguageDetectorAdapter:
    """Calls Groq''s hosted Whisper endpoint for language detection only.

    Usage:
        adapter = GroqLanguageDetectorAdapter(
            api_key="gsk_...",
            model="whisper-large-v3-turbo",
            timeout=8.0,
            sample_seconds=12.0,
        )
        result = adapter.detect(audio_bytes, "recording.mp3")
        # result.detected_language -> "en", "bn", "hi", etc. (raw Groq code)
        # Use resolve_routing_language(result.detected_language) for routing.

    Raises:
        LanguageDetectionError: on network failure, timeout, HTTP error,
            malformed JSON, or missing/empty ``language`` field.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        timeout: float = 8.0,
        sample_seconds: float = 12.0,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._sample_seconds = sample_seconds

    def detect(self, audio_bytes: bytes, filename: str) -> LanguageDetectionResult:
        """Send a trimmed audio sample to Groq and return the detected language.

        Only the ``language`` field of the Groq response is used. The full
        transcript text returned by Groq is discarded — this call is for
        language identification only, not transcription.
        """
        trimmed_bytes, trimmed_name = _trim_audio_bytes(
            audio_bytes, filename, self._sample_seconds
        )

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    GROQ_TRANSCRIPTIONS_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": (trimmed_name, io.BytesIO(trimmed_bytes), "audio/wav")},
                    data={
                        "model": self._model,
                        "response_format": "verbose_json",
                    },
                )
        except httpx.TimeoutException as exc:
            raise LanguageDetectionError(
                f"Groq request timed out after {self._timeout}s: {exc}",
                provider="groq",
            ) from exc
        except httpx.RequestError as exc:
            raise LanguageDetectionError(
                f"Groq network error: {exc}",
                provider="groq",
            ) from exc

        # HTTP-level error
        if response.status_code != 200:
            raise LanguageDetectionError(
                f"Groq returned HTTP {response.status_code}: {response.text[:200]}",
                provider="groq",
            )

        # Parse response
        try:
            body = response.json()
        except Exception as exc:
            raise LanguageDetectionError(
                f"Groq response is not valid JSON: {exc}",
                provider="groq",
            ) from exc

        raw_language: Optional[str] = body.get("language")
        if not raw_language or not raw_language.strip():
            raise LanguageDetectionError(
                f"Groq response missing or empty 'language' field. Body keys: {list(body.keys())}",
                provider="groq",
            )

        # Normalize: Groq returns full names ("english") in verbose_json mode.
        # Convert to ISO 639-1 codes ("en") so the routing rule works correctly.
        raw_stripped = raw_language.strip().lower()
        detected = _normalize_language_code(raw_stripped)

        if detected != raw_stripped:
            logger.info(
                f"Groq language normalized: '{raw_stripped}' -> '{detected}' "
                f"(model={self._model})"
            )
        else:
            logger.info(f"Groq detected language: '{detected}' (model={self._model})")

        return LanguageDetectionResult(
            detected_language=detected,
            confidence=None,       # verbose_json does not expose per-language confidence
            raw_provider_response=body,
            provider="groq",
        )
