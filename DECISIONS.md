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

---

## 8. Qualitative Lab Results: `value: null` Sentinel (Endpoint 2)

**Status:** Accepted

**Context:** The response schema mandates a `value` field. Some lab tests (Urine Albumin = "Nil", HIV = "Non-Reactive", Pregnancy = "Positive") have no meaningful numeric representation.

**Options considered:**
- A) Map qualitative strings to numeric sentinels (e.g., Positive → 1, Negative → 0)
- B) Emit `value: null` for qualitative results

**Decision:** Option B — `value: null`. Mapping qualitative results to arbitrary numerics would fabricate data and could mislead any downstream consumer that does arithmetic. `null` is unambiguous: the test has a result (found in `raw_line`), but it is non-numeric by nature.

**Affected code:** `app/services/document_extraction_service.py` → `_normalize_rows()`, `app/services/normalization.py` → `_QUALITATIVE_SENTINELS`.

---

## 9. Qualified Numerics (`<0.5`, `>200`): Strip Qualifier, Store Threshold (Endpoint 2)

**Status:** Accepted

**Context:** Many lab results appear with inequality qualifiers (`<0.5`, `>200`, `~1.2`). The spec requires a numeric `value` field.

**Options considered:**
- A) Return `null` for all qualified values (same as qualitative policy)
- B) Strip the qualifier and store the threshold as the numeric value
- C) Add a separate `qualifier` field to the schema

**Decision:** Option B. The numeric threshold is a meaningful value (the detection limit of the assay) and is far more useful to a consumer than `null`. The qualifier is fully preserved in `raw_line` — no information is lost. Adding a new schema field (option C) was rejected because it was not specified in the contract and would add complexity for marginal benefit.

**Affected code:** `app/services/normalization.py` → `_QUALIFIER_PATTERN`, `normalize_value()`.

---

## 10. PDF Handling: Native Support via Mistral OCR (Endpoint 2)

**Status:** Accepted

**Context:** Lab reports are sometimes shared as PDFs. PaddleOCR previously required an image, necessitating `pdf2image`.

**Decision:** With the shift to Mistral OCR, we can pass PDFs directly to the provider. The Mistral API accepts PDFs natively. This removes the need for `pdf2image` and poppler dependencies, simplifying our Docker setup and eliminating rasterization overhead.

**Affected code:** `app/services/document_extraction_service.py` (Rasterization logic removed).

---

## 11. Mistral OCR instead of PaddleOCR (Endpoint 2)

**Status:** Accepted

**Context:** The deployment machine has a GTX 1050 Ti with 4 GB VRAM. The Whisper model (Endpoint 1) already occupies most of this budget when loaded. PaddleOCR caused dependency conflicts and environment issues.

**Decision:** Shift to Mistral OCR via API. This:
1. Removes heavy local dependencies (`paddlepaddle`, `paddleocr`).
2. Eliminates VRAM contention with Faster-Whisper.
3. Provides robust out-of-the-box table extraction (Markdown).

**Affected code:** `app/adapters/ocr/mistral_ocr_adapter.py`.

---

## 12. Non-Lab-Report Policy: HTTP 422 with `not_a_lab_report` (Endpoint 2)

**Status:** Accepted

**Context:** A user may accidentally upload a non-medical document (invoice, photo, blank page). The service must decide how to handle this.

**Options considered:**
- A) HTTP 200 with empty/null fields (degrade silently)
- B) HTTP 422 with a structured error containing the classifier reason string
- C) HTTP 400 (client error)

**Decision:** Option B — HTTP 422. The rationale:
- Option A (silent) would return confusingly empty results with no signal to the caller. A consumer would not know whether the document was a lab report with no results or a completely wrong input.
- Option C (400) would imply a malformed request. The request is syntactically valid — it's a legitimate image. The *semantic* issue is the content, which maps more naturally to 422 (Unprocessable Entity).
- HTTP 422 is semantically correct ("the server understands the request but cannot process it") and carries a structured error body (`error: "not_a_lab_report"`, `detail: <classifier reason>`) that gives the caller enough context to prompt the user to re-upload.

The confidence threshold (0.25) is intentionally generous to avoid false positives on degraded scans.

**Affected code:** `app/services/document_classifier.py`, `app/api/routes_documents.py`.

---

## 13. Defaulting to Mock Provider for Document Extraction

**Status:** Accepted

**Context:** Endpoint 2 requires the `MISTRAL_API_KEY` to function. If a user spins up the server without this key, requests would crash or fail with HTTP 500 when interacting with the API.

**Decision:** By default, `.env.example` configures `DOCUMENT_EXTRACTION_PROVIDER=mock`. This ensures that out-of-the-box, the server (or Docker container) starts cleanly and allows UI/integration testing without making external API calls. 

**Trade-offs & risks:**
- **Developer Confusion:** A developer might test the endpoint with a real file and see hardcoded results (e.g., "Fatema Begum") and assume the pipeline is broken, not realizing it's in mock mode. This trade-off is mitigated by clear documentation in the `README.md`.

**Affected code:** `app/config.py`, `.env.example`, `app/adapters/ocr/mock_document_adapter.py`.

