# Speech & Document Extraction — Celloscope AI Service

A production-style speech transcription API built with **FastAPI** and **faster-whisper**, featuring **CUDA GPU acceleration**, Clean Architecture, and an Adapter pattern.

---

## Architecture

```
api/  →  services/  →  adapters/
```

- **Dependencies point inward only**.
- **Services Layer**: Pure business & validation logic — zero FastAPI imports.
- **Adapters Layer**: Interface in `app/adapters/base.py`. `faster_whisper` is strictly isolated within `app/adapters/faster_whisper_adapter.py`.
- **Lazy Provider Loading**: `faster_whisper` is loaded only when configured, keeping mock executions lightweight and GPU-independent.

---

## Features

- ⚡ **CUDA (GPU) Accelerated ASR**: Uses `faster-whisper` (`small` model) with auto-discovered NVIDIA CUDA DLLs on Windows.
- 🔄 **Automatic Hardware Fallback**: Seamlessly falls back from CUDA to CPU if GPU hardware is unavailable.
- 🎭 **Mock Provider**: Zero-dependency offline mock provider for development and CI testing.
- 🛡️ **Robust Validation & Error Handling**: Strict file extension, file size (max 25MB), and audio container validation with structured JSON errors.
- 🤫 **Silence & Ambient Noise Handling**: Returns HTTP 200 with `has_speech: false` instead of hallucinating or raising errors.

---

## Quick Start (Local Setup)

### 1. Activate Virtual Environment & Install Dependencies

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (FastAPI, uvicorn, faster-whisper, nvidia CUDA DLLs)
pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

### 2. Configure Environment (`.env`)

Create or update `.env` in the root directory:

```env
TRANSCRIBE_PROVIDER=faster_whisper
WHISPER_MODEL=small
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=default
MAX_UPLOAD_MB=25
```

### 3. Run the API Server

```powershell
uvicorn app.main:app --reload
```

Server will start at: `http://127.0.0.1:8000`

---

## Testing & API Documentation

### Interactive Swagger UI
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** in your browser to test endpoints interactively.

### Running Automated Test Suite

```powershell
pytest tests/ -v
```

Tests cover:
- **Validation**: File extension rules (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`) and 25MB size limits.
- **Service Layer**: Language mapping (`auto` → `None`, `bn`, `en`).
- **API Integration**: End-to-end endpoint tests for success, silence, invalid formats, and language validation.

---

## API Specification

### `POST /api/v1/transcribe`

**Content-Type:** `multipart/form-data`

| Form Field | Type       | Required | Default  | Allowed Values / Description               |
|------------|------------|----------|----------|--------------------------------------------|
| `audio`    | UploadFile | Yes      | —        | `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`    |
| `language` | string     | No       | `"auto"` | `"bn"`, `"en"`, or `"auto"` (auto-detect)  |

#### Sample Success Response (HTTP 200)

```json
{
    "transcript": "author of The Danger Trail, Philip Steele's, etc.",
    "detected_language": "en",
    "duration_seconds": 3.3,
    "provider": "faster-whisper-small (cuda)",
    "has_speech": true
}
```

#### Sample Silence / Ambient Noise Response (HTTP 200)

```json
{
    "transcript": "",
    "detected_language": null,
    "duration_seconds": 5.1,
    "provider": "faster-whisper-small (cuda)",
    "has_speech": false
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

| Variable             | Default          | Options / Description                                   |
|----------------------|------------------|---------------------------------------------------------|
| `TRANSCRIBE_PROVIDER`| `faster_whisper` | `faster_whisper` (real GPU/CPU model) or `mock`         |
| `WHISPER_MODEL`      | `small`          | `tiny`, `base`, `small`, `medium`, `large-v3`          |
| `WHISPER_DEVICE`     | `cuda`           | `cuda` (GPU) or `cpu`                                  |
| `WHISPER_COMPUTE_TYPE`| `default`       | `default` (auto float32/int8 per GPU), `int8`, `float16`|
| `MAX_UPLOAD_MB`      | `25`             | Maximum file size in MB                                 |

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
