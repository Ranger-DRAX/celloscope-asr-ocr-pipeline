# Architectural Decisions

## 1. No-Speech Detection: Combined Threshold Approach

**Decision:** Use both `no_speech_prob` and `avg_logprob` to determine whether audio
contains genuine speech.

**Context:** faster-whisper returns two per-segment metrics:
- `no_speech_prob` — the model's confidence that a segment contains no speech (0.0–1.0)
- `avg_logprob` — average log-probability of the tokens produced (negative; closer to 0 = more confident)

**Rationale:** Using either threshold alone produces false positives:
- High `no_speech_prob` alone can trigger on whispered or faint speech
- Low `avg_logprob` alone can trigger on heavily accented or noisy speech

Combining both (`avg_no_speech >= 0.6 AND avg_logprob <= -1.0`) is more reliable.
When **both** conditions are met simultaneously, we classify the audio as no-speech.

**Thresholds:**
- `no_speech_prob_threshold = 0.6`
- `avg_logprob_threshold = -1.0`

**Known limitation:** These thresholds were tuned against our own silence/noise test
clips, not a published benchmark. They may need adjustment for different acoustic
environments or recording conditions.

---

## 2. Silence Returns HTTP 200, Not an Error

**Decision:** Audio with no detected speech returns HTTP 200 with `has_speech: false`,
`transcript: ""`, and `detected_language: null`.

**Rationale:** The absence of speech is a valid transcription outcome, not a failure.
The model successfully processed the audio and correctly determined there was nothing
to transcribe. Returning 4xx would imply the request itself was malformed, which it
isn't — the audio was valid, just silent.

---

## 3. Lazy Import of faster-whisper

**Decision:** The `faster_whisper` package is only imported inside `get_adapter()`
when `TRANSCRIBE_PROVIDER=faster_whisper`, not at module load time.

**Rationale:** This ensures:
- The mock path (default) never triggers model loading
- Mock and service startup paths remain fast and lightweight
- The `faster_whisper_adapter.py` file is the **only** file in the repo that imports
  `faster_whisper`

---

## 4. Singleton Adapter Pattern

**Decision:** The transcription adapter is created once and reused for all requests
via a module-level singleton in `routes_transcribe.py`.

**Rationale:** The WhisperModel is expensive to initialize (loads ~1GB model into
GPU/CPU memory). Creating it per-request would be prohibitively slow. The singleton
ensures the model is loaded once at first request and reused thereafter.

---

## 5. Layer Separation (Clean Architecture)

**Decision:** Strict three-layer architecture: `api/ → services/ → adapters/`.

**Enforced constraints:**
- `app/services/` has zero imports from `fastapi` (no `UploadFile`, `HTTPException`, `Request`)
- `app/adapters/base.py` has zero third-party imports (Protocol + dataclass only)
- Only `app/adapters/faster_whisper_adapter.py` may import `faster_whisper`

**Verification:** Automated grep checks in Step 12 of the implementation workflow.

---

## 6. Mock-First Development

**Decision:** Build and validate the entire pipeline against the mock adapter before
touching the real faster-whisper adapter.

**Rationale:** This approach:
- Gets a demoable, testable service running immediately
- Doesn't require GPU hardware for development
- Allows CI/CD pipelines to run without model downloads
- Frozen fixture JSONs serve as regression tests

---

## 7. LID-First Bengali Routing

**Decision:** Replace the previous post-hoc Bengali re-run with a Language-ID-first (LID-first) router inside `FasterWhisperAdapter`.

**Context:** The old design ran Whisper on every `language='auto'` request using the generic multilingual model, then checked the detected language and — if Bengali — re-ran on the Bengali-tuned model. This caused two problems:
1. The Bengali re-run attempted to read a temp file that the `finally` block had already deleted (bug).
2. Whisper's internal language detection on short Bengali clips is unreliable, leading to missed Bengali routing.

**Approach:**
- A SpeechBrain ECAPA-TDNN model (`speechbrain/lang-id-voxlingua107-ecapa`) classifies the language before any Whisper decoding.
- If Bengali is detected with confidence ≥ 0.7, the request routes to the Bengali-tuned Whisper model (`WHISPER_MODEL_BN`).
- Below threshold, the primary model handles the request with Whisper's own auto-detection.
- Explicit `language='bn'` or `language='en'` from the caller **bypasses LID entirely** — routing is deterministic.

**Threshold:** `0.7` — set in `LID_BN_THRESHOLD` env var. Below this value, ambiguous clips fall through to the general model rather than risk mis-routing to Bengali.

**Fallback:** If `WHISPER_MODEL_BN` is not configured, Bengali-routed requests silently fall back to the primary model with a Bengali initial prompt. No operator action required; quality degrades gracefully.

**Temp-file fix:** The temp file is now deleted in a single `finally` block wrapping the entire `transcribe()` method body — after both the LID pass and the Whisper decode. This eliminates the race condition in the old double-decode design.

**New file:** `app/adapters/lid_classifier.py` — thin wrapper around SpeechBrain. Lazy-imported (consistent with Decision #3) so the mock path never triggers a model download.

**New config fields:**
- `LID_MODEL` — SpeechBrain HuggingFace repo or local path (default: `speechbrain/lang-id-voxlingua107-ecapa`)
- `LID_BN_THRESHOLD` — confidence threshold (default: `0.7`)
- `LID_SAVEDIR` — local cache directory for SpeechBrain model files (default: `pretrained_models`)
