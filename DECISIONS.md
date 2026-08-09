# Architectural Decisions 

This document records the major consequential decisions made during the design and implementation of the Celloscope AI/ML Take-Home Service.

---

## 1. ASR Model Selection: Local Faster-Whisper vs. Remote Transcription

* **What was Picked:** 
  Local CT2-quantized models using `faster-whisper`: `whisper-small` for English, and the fine-tuned `faster-whisper-bangla-small-int8` model for Bengali. Models are initialized lazily as singletons on demand.

* **What was Rejected:** 
  Fully remote transcription APIs (e.g., OpenAI Whisper API, Groq full transcription) or heavy unquantized local models (`whisper-large-v3`).

* **Why / Rationale:** 
  1. **Privacy & On-Premise Compliance:** Transcribing full audio locally ensures voice data never leaves the deployment infrastructure.
  2. **VRAM Constraints:** Consumer deployment hardware (e.g. 4 GB VRAM GPUs) cannot hold multiple large models simultaneously. `int8` quantization with lazy singleton loading ensures low memory footprint (~1.2 GB VRAM peak).
  3. **Cost Efficiency:** Local inference avoids recurring per-minute cloud API transcription charges.

---

## 2. Two-Stage Language Identification: Groq LID Sample vs. Full Local Auto-Detect

* **What was Picked:** 
  A two-stage pipeline for `language="auto"`:
  1. Send a short, trimmed audio sample (≤12 s) to Groq's hosted `whisper-large-v3-turbo` for rapid language detection (LID).
  2. Route `"en"` to `whisper-small`, and all other languages to the fine-tuned Bangla model.
  If Groq times out (8 s limit) or fails, local Whisper auto-detection kicks in automatically as a fallback.

* **What was Rejected:** 
  - Sending full audio files to cloud APIs for transcription.
  - Running local Whisper auto-detection on full audio for every request.

* **Why / Rationale:** 
  1. **Speed & Latency:** Groq LID returns language metadata in <200 ms from a tiny sample, avoiding the latency penalty of full local language scanning.
  2. **Privacy Preservation:** Only a non-sensitive 12 s clip is transmitted; full audio stays local.
  3. **Domain Policy:** Non-English traffic for this system is overwhelmingly Bengali. Routing non-English audio to the fine-tuned Bengali model optimizes for target domain accuracy.

---

## 3. Endpoint 2 OCR Engine: Mistral OCR API vs. Local PaddleOCR

* **What was Picked:** 
  `mistral-ocr-latest` API accessed exclusively via an isolated adapter (`MistralOCRAdapter`).

* **What was Rejected:** 
  Local `PaddleOCR` / `paddlepaddle` engines.

* **Why / Rationale:** 
  1. **Environment & Hardware Stability:** `PaddleOCR` introduced severe C++ runtime attribute conflicts (`ConvertPirAttribute2RuntimeAttribute`) on Windows/PyTorch environments, required heavy native binaries (`cv2`), and competed for local GPU VRAM with Faster-Whisper.
  2. **Native Markdown & PDF Support:** Mistral OCR outputs structured Markdown tables directly and natively handles PDF documents without needing local `pdf2image`, `poppler`, or OpenCV dependencies.
  3. **Clean Adapter Abstraction:** Isolating Mistral inside `adapters/ocr/mistral_ocr_adapter.py` allows the rest of the application to remain 100% cloud-agnostic.

---

## 4. Clean Architecture & Default Mock Provider Pattern

* **What was Picked:** 
  A strict 3-layer Clean Architecture (`api/` → `services/` → `adapters/`) where provider selection is 100% configuration-driven via `.env` (`TRANSCRIBE_PROVIDER=mock|faster_whisper` and `DOCUMENT_EXTRACTION_PROVIDER=mock|mistral_ocr`). Mock adapters serve out-of-the-box fixture responses from disk by default.

* **What was Rejected:** 
  - Monolithic route handlers mixing FastAPI types, business logic, and OCR/ASR SDKs.
  - Mandatory remote API keys or heavy model downloads required for running tests or launching Docker.

* **Why / Rationale:** 
  1. **Assessment Requirement (#11):** `docker compose up` must launch cleanly out-of-the-box without credentials or model downloads. Defaulting to mock adapters fulfills this requirement perfectly.
  2. **Testability & CI:** Disk-replay mock adapters enable sub-second unit and integration test runs without network I/O or GPU access.
  3. **Dependency Isolation:** No provider SDK (`mistralai`, `faster_whisper`) or FastAPI type (`UploadFile`, `HTTPException`) leaks into `services/`.

---

## 5. Medical Extraction Strategy: Evidence Preservation & Strict Numeric Enforcement

* **What was Picked:** 
  An evidence-preserving pipeline where the OCR engine provides raw Markdown text, while application services deterministically parse header metadata, infer column roles (`Test | Value | Unit | Range`), validate numeric rows, and normalize values/units. `raw_line` is preserved **verbatim** for every result row, and non-numeric/qualitative rows (e.g. "Nil", "Positive") are excluded from `results[]` to satisfy the strict numeric requirement. Non-lab documents return HTTP 422 `not_a_lab_report`.

* **What was Rejected:** 
  - Direct end-to-end LLM OCR-to-JSON prompt engineering.
  - Converting qualitative results ("Nil", "Positive") to arbitrary numeric sentinels (`null` or `0`).
  - Silently guessing ambiguous dates or missing metadata.

* **Why / Rationale:** 
  1. **Determinism over Hallucination:** Direct LLM-to-JSON prompts frequently hallucinate medical metrics or misalign table columns. Application-owned parsing ensures reliable column role mapping even when layouts differ (`Test | Value | Unit | Range` vs `Test | Range | Value`).
  2. **Evidence Traceability:** `raw_line` guarantees that human reviewers can verify exact OCR text against normalized values.
  3. **Safety First:** Medical extraction must prefer excluding an ambiguous row over fabricating medical data.
---

## 6. Provider Adapter Interface: Stable Contract Behind Replaceable Providers

- **What was Picked:**
  Both transcription and document extraction use provider interfaces so the API/service layer does not depend on a specific vendor or model.

  Conceptually:

  ```text
  TranscriptionProvider
      ├── FasterWhisperAdapter
      └── MockTranscriptionAdapter

  DocumentExtractionProvider
      ├── MistralOCRAdapter
      └── MockDocumentAdapter




---

## 7. Raw OCR Evidence Preservation

- **What was Picked:**
  The OCR output is treated as immutable evidence. Every structured lab result retains the exact OCR row text in `raw_line`.

- **What was Rejected:**

  - Reconstructing `raw_line` from normalized fields.
  - Cleaning OCR text before storing it.
  - Allowing the parser to overwrite the original OCR representation.

- **Why / Rationale:**

  1. **Traceability:** Reviewers can compare structured values against the original OCR evidence.
  2. **Debuggability:** Parser errors can be distinguished from OCR errors.
  3. **Auditability:** Medical extraction remains explainable.
  4. **Safety:** Normalization cannot silently erase information contained in the original OCR output.

  Structured fields may be normalized, but `raw_line` must remain unchanged.
---

## 8. Dynamic Table Column Mapping Instead of Fixed Positions

- **What was Picked:**
  Table columns are mapped semantically from OCR-detected headers before result rows are parsed.

  Example:

  ```text
  Test | Value | Unit | Reference

## 9. Conservative Result Validation: Exclude Rather Than Guess

- **What was Picked:**
  A row is returned in `results[]` only when the parser can confidently identify a test name and numeric result.

  Rows that are headers, notes, interpretation text, doctor information, or lack a confident numeric value are excluded.

- **What was Rejected:**

  - Returning every OCR line as a medical result.
  - Filling missing values with `0`.
  - Returning `null` for required numeric values.
  - Inferring values from surrounding rows without sufficient evidence.

- **Why / Rationale:**

  The assessment requires every result to contain a numeric value. More importantly, medical extraction should prioritize correctness over completeness.

  Therefore:

  ```text
  uncertain row → exclud
# 10. Deterministic Normalization Instead of LLM-Based Normalization

- **What was Picked:**
  Value, unit, reference-range, and date normalization are implemented as deterministic application logic.

- **What was Rejected:**

  - Asking an LLM to normalize every extracted field.
  - Provider-specific normalization.
  - Silent correction of ambiguous values.

- **Why / Rationale:**

  Deterministic normalization provides:

  1. Repeatable outputs.
  2. Unit-testable behavior.
  3. Easier debugging.
  4. No additional inference/hallucination layer.
  5. Independence from the OCR provider.

  Examples:

  ```text
  12,500     → 12500
  mg/dl      → mg/dL
  150--200   → 150 - 200
  10 Jun 2025 → 2025-06-10
## 11. Non-Lab Documents: Fail Safely

- **What was Picked:**
  Documents that cannot be confidently identified as English medical lab reports return a structured `422 not_a_lab_report` response.

- **What was Rejected:**

  - Returning an empty successful result.
  - Treating arbitrary OCR text as laboratory data.
  - Attempting aggressive extraction from unrelated documents.

- **Why / Rationale:**

  A valid image upload does not necessarily contain a valid lab report.

  Returning structured failure allows downstream clients to distinguish:

  ```text
  invalid upload
## 12. Structured Error Handling at the API Boundary

- **What was Picked:**
  Expected failures are converted into stable structured HTTP responses at the API boundary.

  Examples include:

  ```text
  unsupported_file_type
  document_too_large
  invalid_audio
  provider_timeout
  provider_unavailable
  not_a_lab_report