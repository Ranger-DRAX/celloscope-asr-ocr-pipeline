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

---

## 7. Hybrid Language Detection: Groq API + Local Specialized Transcription, with Default-to-Bangla Routing

**Status:** Accepted

### Context

The original single-stage pipeline used local Whisper auto-detect (`language=None`) to identify the spoken language and then transcribed with the same model. This had two problems:

1. **Model mismatch**: The generic multilingual `whisper-small` checkpoint is notably weaker on Bangla (low-resource in training data) than on English. A Bengali-finetuned model (`faster-whisper-bangla-small-int8`) produces significantly better transcripts for Bangla audio.

2. **VRAM budget**: Loading both models simultaneously (English + Bangla) exceeds a practical 4 GB VRAM budget. A routing step is needed to select the correct model *before* transcription, so only one model is in GPU memory at a time.

The product's audio traffic is overwhelmingly English or Bangla. Fast, accurate language identification from a short audio sample is sufficient to make the routing decision.

### Decision

Implement a **two-stage pipeline**:

**Stage 1 — Language detection via Groq API:**
- Send only a short trimmed sample (≤12 s, configurable via `LANGUAGE_DETECTION_SAMPLE_SECONDS`) to Groq's hosted Whisper endpoint (`/openai/v1/audio/transcriptions`, `response_format=verbose_json`).
- Read only the `language` field — discard the transcript. This is detection-only, not transcription.
- Enforce a hard timeout (`LANGUAGE_DETECTION_TIMEOUT_SECONDS`, default 8 s). Groq being slow or down must never hard-fail a transcription request.
- On any `LanguageDetectionError` (timeout, network error, HTTP error, malformed response, missing `language` field): log a warning and fall back to local Whisper auto-detect on the primary model.

**Stage 2 — Routing rule + local transcription:**
- Apply `resolve_routing_language(detected_language)`:
  - `"en"` → route to English model (`whisper-small`).
  - **Anything else** → route to the Bengali fine-tuned model (`faster-whisper-bangla-small-int8`).
- Load the selected model lazily on first use (one model in VRAM at a time).
- When `language` is specified explicitly by the caller (`"en"` or `"bn"`), Stage 1 is skipped entirely — no Groq call is made.

**Routing rule rationale:**
The binary default-to-Bangla rule is intentional, not a limitation. Given current traffic patterns (overwhelmingly English or Bangla), routing everything non-English to the Bangla-tuned model is the most practical policy. A per-language model registry would require additional model downloads, VRAM management complexity, and ongoing maintenance for each new language.

### Alternatives Considered

| Alternative | Reason rejected |
|---|---|
| **Local-only auto-detect** (status quo) | No per-language model routing; Bengali quality suffers on generic `whisper-small`. |
| **Racing both local models** | Doubles GPU memory use; wasteful and slow for the common case. |
| **Full multi-language adapter map** | One specialized model per detected language — operationally complex, each model needs download + VRAM slot. Not justified given current traffic. |
| **Text-based langid (e.g. langdetect)** | Not applicable to raw audio. Would require a preliminary ASR pass first, defeating the purpose. |
| **Use Groq for full transcription** | Per-token cost at production scale; audio data leaves the machine; not suitable for a privacy-conscious on-premise deployment. |

### Consequences

**Benefits:**
- Fast, accurate language ID without loading two local models simultaneously.
- Full audio stays local — only a short detection sample is sent to a third party.
- Explicit `language` bypass saves Groq API cost and latency on known-language requests.
- Graceful degradation: Groq being down degrades to local auto-detect quality (not a failure).

**Trade-offs & risks:**
- Groq is an external dependency on the critical path. Must maintain a working local fallback at all times.
- Short audio sample is sent to Groq for detection — accept this privacy trade-off for detection-only (transcript is discarded).
- Cost/rate-limit exposure proportional to `language="auto"` request volume. Mitigated by short sample + hard timeout.
- **Non-English, non-Bangla audio (e.g. Hindi, Spanish) will be transcribed by the Bangla-tuned model and produce degraded output rather than an explicit "unsupported language" error.** This is a deliberate, accepted trade-off given current traffic patterns. Revisit if language diversity increases significantly.
- `LanguageDetectionFatalError` (both Groq and local fallback fail) is possible but extremely unlikely — the local fallback has no external dependencies.
