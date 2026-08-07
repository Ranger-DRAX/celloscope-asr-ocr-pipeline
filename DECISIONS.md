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
- Fresh clones can run tests immediately without installing CUDA/GPU deps
- Docker default path starts instantly
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
