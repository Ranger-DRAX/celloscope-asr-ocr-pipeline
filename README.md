# Speech & Document Extraction — Celloscope AI Service

FastAPI service for two assessment capabilities:

1. **Endpoint 1 — Speech transcription** for Bengali and English audio.
2. **Endpoint 2 — Medical lab-report extraction** from photographs, scans, and PDFs.

The architecture uses Clean Architecture, configuration-driven provider adapters, deterministic parsing/normalization, structured errors, and mock-first testing.

## Architecture

```text
                         Client
                           |
             +-------------+-------------+
             |                           |
             v                           v
      POST /api/v1/transcribe     POST /api/v1/documents/extract
             |                           |
             v                           v
         API Layer                   API Layer
             |                           |
             v                           v
         Services                    Services
             |                           |
       +-----+-----+             +-------+--------+
       |           |             |                |
       v           v             v                v
    Groq ID    Whisper       Mistral OCR        Mock OCR
       |           |             |                |
       +-----+-----+             +-------+--------+
             |                           |
             v                           v
        ASR Response             Raw OCR Evidence
                                         |
                                         v
                                  Lab Report Parser
                                         |
                              +----------+----------+
                              |                     |
                              v                     v
                         Metadata Parser        Table Parser
                                                     |
                                                     v
                                              Column Mapping
                                                     |
                                                     v
                                              Row Validation
                                                     |
                                                     v
                                           Normalization
                                                     |
                                                     v
                                               Pydantic
                                                     |
                                                     v
                                                 JSON
```

### Layer separation

```text
api/  →  services/  →  adapters/
```

Rules:

- `api/` owns HTTP routing, multipart handling, schemas, and HTTP errors.
- `services/` owns orchestration, parsing, validation, and normalization.
- `adapters/` owns provider/model SDK integration.
- No provider SDK/model library is imported outside `adapters/`.
- No FastAPI `UploadFile`, `Request`, or `HTTPException` is used in `services/`.
- `faster_whisper` is isolated to its adapter.
- `mistralai` is isolated to the Mistral OCR adapter.
- Provider selection is configuration-driven.

---

# Endpoint 1 — Speech Transcription

## API

```text
POST /api/v1/transcribe
Content-Type: multipart/form-data
```

Fields:

| Field | Type | Required | Values |
|---|---|---:|---|
| `audio` | UploadFile | Yes | `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg` |
| `language` | string | No | `bn`, `en`, `auto` |

Maximum upload size: **25 MB**.

### Language routing

For `language=auto`:

```text
short audio sample
      ↓
Groq language detection
      ↓
en → whisper-small
other → Bangla fine-tuned Whisper
```

Only the short detection sample is sent to Groq. Full transcription remains local. Groq failure/timeout falls back to local language detection.

For explicit `bn` or `en`, remote language detection is skipped.

### No-speech behavior

Silence/ambient noise is a valid transcription outcome and returns HTTP 200:

```json
{
  "transcript": "",
  "detected_language": null,
  "has_speech": false
}
```

Current no-speech classification combines:

```text
no_speech_prob >= 0.6
AND
avg_logprob <= -1.0
```

These thresholds are tuned against the project's own fixtures, not a universal benchmark.

---

# Endpoint 2 — Medical Lab Report Extraction

## API

```text
POST /api/v1/documents/extract
Content-Type: multipart/form-data
```

Field:

```text
file=<image or PDF>
```

Supported formats used by the implementation/test suite include:

```text
.jpg
.jpeg
.png
.webp
.pdf
```

Maximum upload size: **25 MB**.

The endpoint targets English-language medical lab reports, including angled photographs, poor lighting, low-quality scans, cropped pages, and multi-page PDFs.

## Provider architecture

```text
DocumentExtractionProvider
        |
        +-----------------------+
        |                       |
        v                       v
MistralOCRAdapter       MockDocumentAdapter
        |                       |
        +-----------+-----------+
                    |
                    v
              Raw OCR Evidence
                    |
                    v
             Lab Report Parser
                    |
          +---------+----------+
          |                    |
          v                    v
      Metadata             Table Parser
      Parser                   |
                              v
                       Column Role Mapping
                              |
                              v
                        Row Validation
                              |
                              v
                       Normalization
                              |
                              v
                          Pydantic
                              |
                              v
                             JSON
```

Mistral OCR is an **evidence provider**, not the final application JSON generator. Parsing and normalization are application-owned and deterministic.

### Provider selection

```env
DOCUMENT_EXTRACTION_PROVIDER=mock
```

or:

```env
DOCUMENT_EXTRACTION_PROVIDER=mistral_ocr
```

Changing provider must not require source-code changes.

### Mistral OCR

The real adapter is isolated at:

```text
app/adapters/ocr/mistral_ocr_adapter.py
```

Configured model:

```text
mistral-ocr-latest
```

PDFs are passed directly to Mistral OCR; no PaddleOCR, `pdf2image`, or Poppler dependency is required for Endpoint 2.

### Mock provider

The mock provider:

- makes no network calls
- loads no OCR model
- requires no Mistral key
- replays deterministic OCR fixtures from disk
- exercises the real parsing/normalization pipeline

---

## Endpoint 2 Response Contract

```json
{
  "meta": {
    "patient_name": "...",
    "age": "...",
    "sex": "...",
    "report_date": "...",
    "lab_name": "...",
    "reference_no": "..."
  },
  "results": [
    {
      "test_name": "Haemoglobin",
      "value": 15.2,
      "unit": "g/dL",
      "reference_range": "13.5 - 19.5",
      "flag": "",
      "raw_line": "Haemoglobin 15.2 13.5--19.5 g/dL"
    }
  ]
}
```

### `raw_line` is evidence

Every emitted result must contain `raw_line` representing the exact OCR text associated with that row.

Never clean, spell-correct, normalize, or silently rewrite `raw_line`.

Structured fields may be normalized independently.

### Numeric value requirement

Every returned `results[]` item must contain a numeric `value`.

Therefore:

- rows without a confident numeric result are excluded;
- qualitative-only results such as `Positive`, `Negative`, `Reactive`, `Non-Reactive`, or `Nil` are **not** converted to arbitrary numeric sentinels;
- the service never fabricates a medical value.

This follows the assessment requirement that every result include a numeric value.

---

## Metadata Parsing

Extract:

```text
patient_name
age
sex
report_date
lab_name
reference_no
```

Use label/context-based parsing rather than fixed line positions.

Missing metadata must not be guessed.

---

## Table Parsing

Different reports may use different column orders, for example:

```text
Tests | Value | Unit | Reference
Tests | Value | Reference | Units
```

Therefore the parser must:

```text
table detection
    ↓
header detection
    ↓
semantic column-role mapping
    ↓
row extraction
```

Header synonyms may include `test`, `tests`, `test name`, `investigation`, `result`, `value`, `reference`, `reference range`, `normal range`, `unit`, and `units`.

This prevents the common failure where OCR is correct but `unit` and `reference_range` are swapped.

---

## Result Filtering

Do not turn arbitrary OCR lines into medical results.

Exclude:

- table headers
- section headers
- notes
- interpretation rows
- doctor/consultant names
- footer text
- empty rows
- rows without a confident numeric result

For example:

```text
5.7 -- 6.4 : Prediabetes
6.5 Or higher : Diabetes
```

are interpretation/reference text, not separate test results.

---

## Normalization Rules

### Values

```text
12.5       → 12.5
12         → 12
12,500     → 12500
1.2 x 10^3 → 1200
```

### Qualified values

```text
<0.5 → 0.5
>200 → 200
```

The original qualifier remains in `raw_line`.

### Ambiguous values

If a value such as:

```text
0.8 - 1.2
```

cannot confidently be identified as one numeric result, do not guess. Exclude the row from `results[]`.

### Units

```text
mg/dl   → mg/dL
gm/dl   → g/dL
g/dL    → g/dL
mmol/L  → mmol/L
10^3/ul → 10³/µL
```

### Reference ranges

```text
150--200 → 150 - 200
150–200  → 150 - 200
```

### Dates

Canonical form:

```text
YYYY-MM-DD
```

Example:

```text
10 Jun 2025 → 2025-06-10
```

Ambiguous dates are not guessed.

---

## Non-Lab Documents

A valid image that is semantically not a lab report should not produce garbage results.

Return structured HTTP 422:

```json
{
  "error": "not_a_lab_report",
  "detail": "..."
}
```

---

# Test Data & Provenance

### 1. Audio Test Data (Endpoint 1)

* **Bangla Audio Source**: `mozilla-foundation/common_voice_11_0` (bn)
* **English Audio Source**: `mozilla-foundation/common_voice_17_0` (en)
* **Location**: `testdata/audio_BN` and `testdata/audio_EN`
* **Reference Transcripts**: Every audio file in `testdata/audio_BN` and `testdata/audio_EN` is accompanied by a ground-truth `.txt` file containing the exact reference transcript to enable accuracy and Word Error Rate (WER) measurement.

### 2. Medical Lab Reports (Endpoint 2)

* **Location**:
  ```text

  ```
* **Selection Rationale**:
  - **Low-Graded Images**: Sourced to test performance under adverse physical conditions (mobile camera photographs, glare, perspective distortion, low lighting, cropped boundaries).
  - **Scanned Images & PDFs**: Chosen to evaluate different multi-column table layouts (`Test | Value | Unit | Range` vs `Test | Range | Value`), scientific notation parsing (`1.2 x 10^3`), and qualified bounds (`<0.5`).

---

# Testing

Run the complete suite:

```powershell
pytest -v
```

Targeted tests should cover:

- provider factories and mock adapters
- file validation and 25 MB limits
- metadata parsing
- table/header detection
- different column orders
- result-row filtering
- numeric value normalization
- unit normalization
- reference-range normalization
- date normalization
- non-lab classification
- Endpoint 2 mock integration
- Endpoint 1 routing/no-speech/fallback behavior

Tests must assert semantic output, not merely `HTTP 200`.

---

# Docker / Clean Clone

The project is fully dockerized to ensure clean-clone execution without credential requirements. 

To start the API on port 8000 using Mock Providers (No API keys or model downloads required):
```bash
docker compose up --build -d
```
The API will be available at `http://localhost:8000/docs`.

To stop the application and clean up containers:
```bash
docker compose down
```

*Note: Real providers (Mistral, Faster-Whisper, Groq) can be activated by providing a `.env` file mapping or explicitly passing variables into the Docker environment.*

Local non-Docker development with Uvicorn remains supported.

---

# Quick Start

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

For real Endpoint 2 extraction:

```env
DOCUMENT_EXTRACTION_PROVIDER=mistral_ocr
MISTRAL_API_KEY=your_key
MISTRAL_OCR_MODEL=mistral-ocr-latest
```

For clean/mock execution:

```env
DOCUMENT_EXTRACTION_PROVIDER=mock
```

Never commit real API keys.

---

# Environment Configuration

```env
TRANSCRIBE_PROVIDER=faster_whisper
WHISPER_MODEL=small
WHISPER_MODEL_BN=pretrained_models/faster-whisper-bangla-small-int8
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=default

MAX_UPLOAD_MB=25

DOCUMENT_EXTRACTION_PROVIDER=mock
MISTRAL_OCR_MODEL=mistral-ocr-latest
MISTRAL_API_KEY=

GROQ_API_KEY=
GROQ_MODEL=whisper-large-v3-turbo
ENABLE_REMOTE_LANGUAGE_DETECTION=true
LANGUAGE_DETECTION_TIMEOUT_SECONDS=8.0
LANGUAGE_DETECTION_SAMPLE_SECONDS=12.0
```

---

# Privacy and Logging

Medical reports and audio can contain sensitive information. Do not log:

- patient names
- complete OCR responses
- complete medical results
- uploaded document contents
- API keys

Operational logs should contain only necessary metadata such as request ID, provider, file type, file size, processing duration, and result count.

---

# Known Limitations

- Extremely blurred/cropped documents can produce partial extraction.
- OCR errors cannot always be corrected safely.
- Non-English lab reports are outside the Endpoint 2 contract.
- Ambiguous values/dates are intentionally not guessed.
- Real Mistral mode requires network access and an API key.
- Non-English/non-Bangla audio follows the accepted default-to-Bangla routing policy.

---

# Engineering Principle

```text
Document
   ↓
OCR
   ↓
Raw evidence
   ↓
Structure detection
   ↓
Semantic column mapping
   ↓
Conservative validation
   ↓
Deterministic normalization
   ↓
Pydantic validation
   ↓
JSON
```


