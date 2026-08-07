# Speech & Document Extraction — Celloscope AI Service

A production-style speech transcription API built with **FastAPI** and **faster-whisper**.

## Architecture

```
api/  →  services/  →  adapters/
```

Dependencies always point inward. No FastAPI types leak into the service layer. No whisper imports outside the adapter layer.

## Quick Start (Development)

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the server (mock provider — no GPU needed)
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
```

## API

### POST `/api/v1/transcribe`

**Content-Type:** `multipart/form-data`

| Field    | Type       | Required | Description                          |
|----------|------------|----------|--------------------------------------|
| audio    | UploadFile | Yes      | Audio file (.wav, .mp3, .m4a, .flac, .ogg) |
| language | string     | No       | `"bn"`, `"en"`, or `"auto"` (default) |

**Success Response (200):**

```json
{
    "transcript": "Author of the danger trail...",
    "detected_language": "en",
    "duration_seconds": 3.42,
    "provider": "faster-whisper-small",
    "has_speech": true
}
```

**Silence / Ambient Noise Response (200):**

Silence and ambient noise inputs return HTTP 200 — absence of speech is a valid
outcome, not a failure.

```json
{
    "transcript": "",
    "detected_language": null,
    "duration_seconds": 5.10,
    "provider": "faster-whisper-small",
    "has_speech": false
}
```

**Error Response (400):**

```json
{
    "detail": {
        "error": "unsupported_format",
        "detail": "Unsupported audio format: '.pdf'"
    }
}
```

## Configuration

All settings are driven by environment variables with safe defaults:

| Variable             | Default          | Description                        |
|----------------------|------------------|------------------------------------|
| TRANSCRIBE_PROVIDER  | `mock`           | `mock` or `faster_whisper`         |
| WHISPER_MODEL        | `small`          | Whisper model size                 |
| WHISPER_DEVICE       | `cpu`            | `cpu` or `cuda`                    |
| WHISPER_COMPUTE_TYPE | `int8`           | `int8` (CPU) or `float16` (CUDA)  |
| MAX_UPLOAD_MB        | `25`             | Maximum upload file size           |

## Provider Switching

```bash
# Mock provider (default, no GPU needed)
TRANSCRIBE_PROVIDER=mock uvicorn app.main:app

# Real provider (requires GPU + faster-whisper)
TRANSCRIBE_PROVIDER=faster_whisper WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16 uvicorn app.main:app
```

## Docker

```bash
# Default path (mock provider, zero GPU)
docker compose up

# With .env file for real provider
cp .env.example .env
# Edit .env to set TRANSCRIBE_PROVIDER=faster_whisper
docker compose up
```

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- **Validation**: file extension and size checks
- **Service**: language parameter mapping (`auto` → `None`)
- **API**: end-to-end integration with mock provider (200 for valid/silence, 400 for invalid)
