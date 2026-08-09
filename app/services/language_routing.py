"""Language routing rule for the two-stage ASR pipeline.

This module contains exactly one public function: resolve_routing_language().

Design note — why this is a separate module:
    The routing rule must be independently unit-testable and must not be baked
    into either the Groq adapter or the FasterWhisper adapter. Keeping it here
    ensures that changing the routing policy (e.g. adding a third language)
    requires editing only this file, with no ripple effect on adapters.

Default-to-Bangla policy (deliberate):
    This is a binary router for a product whose audio traffic is overwhelmingly
    English or Bangla. Any detected language that is not exactly "en" routes to
    the Bangla fine-tuned model. Non-English, non-Bangla audio (e.g. Hindi,
    Spanish) will be transcribed by a Bangla-tuned model and may produce
    degraded output — this is accepted given current traffic patterns.
    See DECISIONS.md ADR #7 for full rationale.

Extending:
    To add a third language (e.g. Hindi with its own model):
    1. Add a branch here: if detected_language == "hi": return "hi"
    2. Add the new model path in config.py.
    3. Add the new adapter case in transcribe_service.get_transcription_adapter_for_language().
    4. Update DECISIONS.md with the new routing rule.
"""

from typing import Literal


def resolve_routing_language(detected_language: str) -> Literal["en", "bn"]:
    """Map a raw detected language code to a binary routing language.

    Args:
        detected_language: The raw language code returned by the detection
            provider (e.g. "en", "bn", "hi", "es", "", "unknown"). Case
            is normalized to lowercase before comparison.

    Returns:
        "en" if detected_language is exactly "en" (English).
        "bn" for ALL other values — including "bn", "hi", "es", "",
            "unknown", and any unrecognized code.

    This implements the deliberate default-to-Bangla policy: when in doubt,
    route to the Bangla fine-tuned model rather than the generic English one,
    because the product''s non-English traffic is overwhelmingly Bangla.

    Examples:
        >>> resolve_routing_language("en")
        ''en''
        >>> resolve_routing_language("bn")
        ''bn''
        >>> resolve_routing_language("hi")   # Hindi -> Bangla model
        ''bn''
        >>> resolve_routing_language("es")   # Spanish -> Bangla model
        ''bn''
        >>> resolve_routing_language("")     # empty -> Bangla model
        ''bn''
        >>> resolve_routing_language("unknown")
        ''bn''
    """
    if detected_language.strip().lower() == "en":
        return "en"
    return "bn"
