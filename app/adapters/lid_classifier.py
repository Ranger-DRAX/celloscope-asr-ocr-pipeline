"""Language Identification (LID) classifier backed by SpeechBrain.

Wraps ``speechbrain/lang-id-voxlingua107-ecapa`` (ECAPA-TDNN trained on
107 languages, ~94 MB download).  Exported interface is intentionally
minimal so the adapter layer never touches speechbrain types directly.

Lazy import rule (DECISIONS.md §3): ``speechbrain`` is only imported
inside ``LidClassifier.__init__`` so that the mock/test startup path never
triggers a model download.

Note on SpeechBrain 1.0+: ``speechbrain.pretrained`` was removed; the
correct path is ``speechbrain.inference``. The old path triggers a
lazy-module redirect that is incompatible with Python 3.14's
``inspect.getmodule()`` and causes a spurious ``k2`` ImportError
via ``speechbrain.integrations.k2_fsa``.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# SpeechBrain label format: "bn - Bengali" or "en - English".
# We parse only the two-letter ISO 639-1 prefix.
_LABEL_RE = re.compile(r"^([a-z]{2,3})\s*-")


def _parse_lang(label: str) -> str:
    """Extract the ISO 639-1 code from a SpeechBrain label string.

    Examples::

        "bn - Bengali"  -> "bn"
        "en - English"  -> "en"
        "zh - Chinese"  -> "zh"

    Falls back to the raw label (lower-cased) if the pattern doesn't match.
    """
    label = label.strip()
    m = _LABEL_RE.match(label)
    if m:
        return m.group(1).lower()
    logger.warning("LidClassifier: unexpected label format %r — using raw value", label)
    return label.lower()


class LidClassifier:
    """Lightweight wrapper around the SpeechBrain ECAPA-TDNN LID model.

    Parameters
    ----------
    model_source:
        HuggingFace repo ID or local path, e.g.
        ``"speechbrain/lang-id-voxlingua107-ecapa"``.
    savedir:
        Directory where SpeechBrain caches the downloaded model files.
        Should be a persistent location in production (e.g. a mounted volume).
    device:
        ``"cuda"`` or ``"cpu"``.  SpeechBrain will use the same device as the
        Whisper models so GPU memory is not double-allocated unnecessarily.
    """

    def __init__(self, model_source: str, savedir: str, device: str) -> None:
        # Lazy import -- keep the mock/test startup path free of torch/speechbrain.
        # IMPORTANT: use speechbrain.inference (SpeechBrain >= 1.0), NOT
        # speechbrain.pretrained. The .pretrained path is a deprecated lazy-module
        # shim; on Python 3.14 inspect.getmodule() triggers __getattr__ on that
        # shim, which cascades into loading speechbrain.integrations.k2_fsa,
        # which requires the optional 'k2' package and raises ImportError.
        try:
            from speechbrain.inference import EncoderClassifier  # type: ignore[import]
        except Exception as exc:
            raise ImportError(
                "speechbrain is required for the LID router. "
                "Install it with: pip install speechbrain"
            ) from exc

        run_opts = {"device": device}
        logger.info(
            "Loading LID model %r into %r (savedir=%r)",
            model_source,
            device,
            savedir,
        )

        # On Windows, SpeechBrain's from_hparams() tries to os.symlink() files
        # from the HuggingFace cache into savedir. This fails with WinError 1314
        # ("A required privilege is not held") unless Developer Mode is enabled
        # or the process is running as administrator.
        #
        # Workaround: if model_source looks like a HuggingFace repo (not a local
        # directory), pre-download it with snapshot_download() to get the local
        # snapshot path, then pass that as `source` to from_hparams(). When
        # source is a local directory, SpeechBrain reads directly from it and
        # never creates symlinks.
        local_source = model_source
        if not os.path.isdir(model_source):
            try:
                from huggingface_hub import snapshot_download  # already installed (dep of speechbrain)
                logger.info("Downloading LID model %r to local cache...", model_source)
                local_source = snapshot_download(
                    model_source,
                    cache_dir=savedir,
                )
                logger.info("LID model cached at %r", local_source)
            except Exception:
                # Fall back to letting SpeechBrain handle it directly.
                logger.warning(
                    "snapshot_download failed for %r — falling back to from_hparams default.",
                    model_source,
                )
                local_source = model_source

        self._classifier = EncoderClassifier.from_hparams(
            source=local_source,
            savedir=local_source if os.path.isdir(local_source) else savedir,
            run_opts=run_opts,
        )
        logger.info("LID model ready.")

    def classify(self, audio_path: str) -> tuple[str, float]:
        """Classify the language of an audio file.

        Parameters
        ----------
        audio_path:
            Absolute path to the audio file (any format readable by
            SpeechBrain / torchaudio).

        Returns
        -------
        (lang_code, confidence)
            ``lang_code`` is a bare ISO 639-1 string, e.g. ``"bn"`` or ``"en"``.
            ``confidence`` is a probability in ``[0.0, 1.0]``.
        """
        import torch  # already a transitive dep of speechbrain

        signal = self._classifier.load_audio(audio_path)
        # load_audio returns a 1-D tensor; unsqueeze to [batch=1, time]
        signal = signal.unsqueeze(0)

        prediction = self._classifier.classify_batch(signal)
        # prediction is a tuple: (out_probs, score, index, text_label)
        # text_label is a list of strings, e.g. ["bn - Bengali"]
        text_label: str = prediction[3][0]
        # score is a tensor of log-softmax values; convert to probability
        log_prob: float = float(prediction[1].squeeze())
        confidence = float(torch.exp(torch.tensor(log_prob)))

        lang_code = _parse_lang(text_label)
        logger.debug(
            "LID result: label=%r -> lang=%r, confidence=%.3f",
            text_label,
            lang_code,
            confidence,
        )
        return lang_code, confidence
