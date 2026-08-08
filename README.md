# Speech & Document Extraction — Celloscope AI Service

A production-style speech transcription API built with **FastAPI** and **faster-whisper**, featuring **CUDA GPU acceleration**, Groq-powered language detection, Clean Architecture, and an Adapter pattern.

---

## Architecture — Two-Stage Pipeline

```mermaid
graph TD
    A["Client<br/>POST /api/v1/transcribe<br/>multipart: audio + language"] --> B["API Layer<br/>routes_transcribe.py"]
    B --> C{"language == auto?"}

    C -- YES --> D["GroqLanguageDetectorAdapter<br/>groq_language_detector.py<br/>sends short trimmed sample only"]
    D -- detected_language --> E{"LanguageDetectionError?"}
    E -- No --> F["resolve_routing_language<br/>language_routing.py"]
    E -- "Yes / Timeout" --> G["Local FasterWhisper<br/>auto-detect fallback<br/>language_detected_by = local_fallback"]
    G --> F

    C -- "NO: explicit bn or en" --> F

    F -- en --> H["FasterWhisperAdapter<br/>model = whisper-small<br/>(en path)"]
    F -- "bn or any other" --> I["FasterWhisperAdapter<br/>model = faster-whisper-bangla-small-int8<br/>(bn path)"]

    H --> J["JSON Response<br/>transcript, detected_language,<br/>language_detected_by, raw_detected_language,<br/>duration_seconds, provider, has_speech"]
    I --> J
```

**Why Groq for detection only, not transcription?**
Groq's hosted Whisper gives fast, accurate language identification from a short sample (≤12 s). Only that sample is sent to Groq — the full audio stays local. Once the language is known, the appropriate fine-tuned local model handles transcription, keeping all audio data on-premise and avoiding per-token Groq costs at transcription scale.

**Default-to-Bangla policy:**
Any detected language that is not exactly English (`"en"`) routes to the Bangla fine-tuned model. This is a deliberate policy for a product whose non-English traffic is overwhelmingly Bangla. Hindi, Spanish, and other audio will be transcribed by the Bangla model and may produce degraded output — see `DECISIONS.md` ADR #7 for full rationale.

### Layer separation

```
api/  →  services/  →  adapters/
```

- **Dependencies point inward only.**
- **Services Layer**: Pure business & validation logic — zero FastAPI imports. See [`app/services/README.md`](app/services/README.md).
- **Adapters Layer**: Interface in `app/adapters/base.py`. `faster_whisper` is strictly isolated within `faster_whisper_adapter.py`. Groq I/O is isolated within `groq_language_detector.py`.
- **Lazy Provider Loading**: Each local Whisper model loads on first use for that language — never both simultaneously (4 GB VRAM budget).

---

## Features

- ⚡ **Two-Stage ASR Pipeline**: Groq language detection → local faster-whisper transcription routed by language.
- 🌐 **Groq Language Detection**: Fast hosted Whisper language ID from a short audio sample — with hard timeout and local fallback.
- 🔤 **Binary Routing**: English → `whisper-small`; everything else → `faster-whisper-bangla-small-int8`.
- 🔄 **Automatic Hardware Fallback**: Seamlessly falls back from CUDA to CPU if GPU is unavailable.
- 🎭 **Mock Provider**: Zero-dependency offline mock for development and CI testing.
- 🛡️ **Robust Validation & Error Handling**: Strict file extension, file size (max 25 MB), and audio container validation with structured JSON errors.
- 🤫 **Silence & Ambient Noise Handling**: Returns HTTP 200 with `has_speech: false` — never hallucinates or raises errors.
- 📊 **Detection Provenance**: New `language_detected_by` and `raw_detected_language` fields show exactly how the language was determined.

---

## Quick Start (Local Setup)

### 1. Activate Virtual Environment & Install Dependencies

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

### 2. Configure Environment (`.env`)

Copy `.env.example` to `.env` and fill in your values:

```env
TRANSCRIBE_PROVIDER=faster_whisper
WHISPER_MODEL=small
WHISPER_MODEL_BN=pretrained_models/faster-whisper-bangla-small-int8
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=default
MAX_UPLOAD_MB=25

# Groq language detection (Stage 1)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=whisper-large-v3-turbo
ENABLE_REMOTE_LANGUAGE_DETECTION=true
LANGUAGE_DETECTION_TIMEOUT_SECONDS=8.0
LANGUAGE_DETECTION_SAMPLE_SECONDS=12.0
```

> ⚠️ **Never commit your real `GROQ_API_KEY`** — `.env` is gitignored.

### 3. Run the API Server

```powershell
uvicorn app.main:app --reload
```

Server will start at: `http://127.0.0.1:8000`

---

## Testing & API Documentation

### Interactive Swagger UI
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to test endpoints interactively.

### Running the Test Suite

```powershell
pytest tests/test_transcribe_api.py -v
```

Tests cover (37 tests, ~1 s with mock provider):
- **Validation**: File extension rules and 25 MB size limits.
- **Routing**: `resolve_routing_language` full truth-table (en, bn, hi, es, empty, unknown).
- **Service**: Two-stage pipeline with Groq mocked; fallback on timeout/error.
- **API Integration**: End-to-end endpoint tests for success, silence, invalid formats, language validation, and new schema fields.

---

## API Specification

### `POST /api/v1/transcribe` (Endpoint 1: ASR)

**Content-Type:** `multipart/form-data`

| Form Field | Type       | Required | Default  | Allowed Values / Description               |
|------------|------------|----------|----------|---------------------------------------------|
| `audio`    | UploadFile | Yes      | —        | `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`     |
| `language` | string     | No       | `"auto"` | `"bn"`, `"en"`, or `"auto"` (auto-detect)  |

### `POST /api/v1/documents/extract` (Endpoint 2: Lab Report OCR)

**Content-Type:** `multipart/form-data`

| Form Field | Type       | Required | Default  | Allowed Values / Description               |
|------------|------------|----------|----------|---------------------------------------------|
| `file`     | UploadFile | Yes      | —        | `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`    |

**Notes:**
- **Provider (Mistral OCR)**: This endpoint uses the Mistral SDK (`mistralai`) and relies on `mistral-ocr-latest` to parse documents. The endpoint uses the `raw_line` from Mistral markdown table parsing.
- **Graceful Degradation**: If the uploaded image does not resemble a lab report (e.g., an invoice or nature photo), the endpoint returns an HTTP 422 error (`not_a_lab_report`).
- **Data Fidelity (`raw_line`)**: The `raw_line` field is SACRED. It always contains the verbatim text detected by OCR for that row, preserving any abbreviations, unparseable ranges, or typos for human review.

#### Document Extraction Normalization Rules:
- **Numeric values**: Standalone numbers (e.g., `12.5`, `12,500`) are converted to `float` (`12.5`, `12500.0`).
- **Scientific notation**: Parsed correctly when possible (e.g., `1.2 x 10^3` to `1200`).
- **Comparison values**: A value like `<0.5` remains parsed as `0.5`, with qualitative results (Positive, Nil) yielding `value: null`. Unparseable ranges (e.g., `0.8 - 1.2`) are rejected from `value` entirely.
- **Units**: Extensively normalized (e.g., `gm/dl` → `g/dL`, `10^3/ul` → `10³/µL`).
- **Reference ranges**: Normalized to a standard format (e.g., `70-110` → `70 - 110`).
- **Dates**: Converted to ISO-8601 (`YYYY-MM-DD`). Note: `DD/MM/YYYY` is assumed over `MM/DD/YYYY` when ambiguous.

#### Limitations
- **Severely Degraded Documents**: Documents that are extremely cropped, have unreadable text, or severe blur may yield partial or no results. The endpoint does not hallucinate data.
- **Ambiguous Dates**: If a date is highly ambiguous (e.g., `03/04/2026`) and context doesn't clarify it, the original string might be preserved or normalization might skip depending on confidence.
- **Missing Values**: Results without numeric values (or identifiable ranges) might be dropped or logged as partial extraction without `value`.

#### Sample Success Response — `POST /api/v1/documents/extract` (HTTP 200)

```json
{
  "meta": {
    "patient_name": "John Doe",
    "age": "45 Years",
    "sex": "Male",
    "report_date": "2024-07-15",
    "lab_name": "POPULATION HEALTH DIAGNOSTICS",
    "reference_no": "PHD-2024-00123"
  },
  "results": [
    {
      "test_name": "Haemoglobin",
      "value": 13.5,
      "unit": "g/dL",
      "reference_range": "13.0 - 17.0",
      "flag": "",
      "raw_line": "Haemoglobin 13.5 gm/dl 13.0 - 17.0"
    },
    {
      "test_name": "Blood Glucose (F)",
      "value": 0.5,
      "unit": "mmol/L",
      "reference_range": "3.9 - 6.1",
      "flag": "L",
      "raw_line": "Blood Glucose (F) <0.5 mmol/L 3.9 - 6.1 L"
    }
  ]
}
```

#### Sample Validation Error Response — `POST /api/v1/transcribe` (HTTP 400)

#### Sample Success Response — auto-detect via Groq (HTTP 200)

```json
{
    "transcript": "author of The Danger Trail, Philip Steele's, etc.",
    "detected_language": "en",
    "duration_seconds": 3.3,
    "provider": "faster-whisper-small (cuda)",
    "has_speech": true,
    "language_detected_by": "groq",
    "raw_detected_language": "en"
}
```

#### Sample Success Response — Groq detected Hindi, routed to Bangla model (HTTP 200)

```json
{
    "transcript": "...",
    "detected_language": "bn",
    "duration_seconds": 5.1,
    "provider": "faster-whisper-pretrained_models/faster-whisper-bangla-small-int8 (cuda)",
    "has_speech": true,
    "language_detected_by": "groq",
    "raw_detected_language": "hi"
}
```

#### Sample Success Response — explicit language (HTTP 200)

```json
{
    "transcript": "আমি বাংলায় কথা বলছি।",
    "detected_language": "bn",
    "duration_seconds": 4.2,
    "provider": "faster-whisper-pretrained_models/faster-whisper-bangla-small-int8 (cuda)",
    "has_speech": true,
    "language_detected_by": "user_specified",
    "raw_detected_language": null
}
```

#### Sample Silence / Ambient Noise Response (HTTP 200)

```json
{
    "transcript": "",
    "detected_language": null,
    "duration_seconds": 5.1,
    "provider": "faster-whisper-small (cuda)",
    "has_speech": false,
    "language_detected_by": "local_fallback",
    "raw_detected_language": null
}
```

#### Sample Validation Error Response (HTTP 400)

```json
{
    "detail": {
        "error": "unsupported_format",
        "detail": "Unsupported audio format: '.exe'"
    }
}
```

---

## Configuration Reference

All settings can be overridden via environment variables or `.env`:

| Variable                             | Default                                              | Description                                                        |
|--------------------------------------|------------------------------------------------------|--------------------------------------------------------------------|
| `TRANSCRIBE_PROVIDER`                | `faster_whisper`                                     | `faster_whisper` (real GPU/CPU model) or `mock`                    |
| `WHISPER_MODEL`                      | `small`                                              | English model: `tiny`, `base`, `small`, `medium`, `large-v3`      |
| `WHISPER_MODEL_BN`                   | `None`                                               | Path to Bengali fine-tuned model (CT2 format)                      |
| `WHISPER_DEVICE`                     | `cuda`                                               | `cuda` (GPU) or `cpu`                                              |
| `WHISPER_COMPUTE_TYPE`               | `default`                                            | `default` (auto float32/int8 per GPU), `int8`, `float16`          |
| `MAX_UPLOAD_MB`                      | `25`                                                 | Maximum file size in MB                                            |
| `GROQ_API_KEY`                       | `None`                                               | Groq Cloud API key — required for Stage 1 detection               |
| `GROQ_MODEL`                         | `whisper-large-v3-turbo`                             | Groq Whisper model for language detection                          |
| `ENABLE_REMOTE_LANGUAGE_DETECTION`   | `true`                                               | Set to `false` to always use local auto-detect                     |
| `LANGUAGE_DETECTION_TIMEOUT_SECONDS` | `8.0`                                                | Hard timeout for Groq call — local fallback triggers on expiry    |
| `LANGUAGE_DETECTION_SAMPLE_SECONDS`  | `12.0`                                               | Audio sample duration sent to Groq (only this leaves the machine) |
| `DOCUMENT_EXTRACTION_PROVIDER`       | `mock`                                               | `mock` or `mistral_ocr`.                                          |
| `MISTRAL_API_KEY`                    | `None`                                               | Required if `DOCUMENT_EXTRACTION_PROVIDER` is `mistral_ocr`       |
| `LOG_LEVEL`                          | `INFO`                                               | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`         |

---

## Removing `testdata/` Audio Files from Remote Git

If `testdata/` audio files were committed accidentally to remote Git:

```powershell
# 1. Untrack testdata from Git without deleting local files
git rm -r --cached testdata/

# 2. Preserve mock fixture JSONs
git add testdata/mock_responses/

# 3. Commit and push changes
git commit -m "refactor(git): untrack testdata audio files"
git push origin feat/audio
```