# `app/services/` — Service Layer

This layer owns all **business logic**. It sits between the API (HTTP) layer and the
adapter (model/external-API) layer. Nothing in here touches FastAPI types — no
`UploadFile`, `HTTPException`, or `Request`. That constraint is intentional: it keeps
the business rules independently testable and easy to reason about.

---

## Module Map

```
app/services/
├── __init__.py             # Package marker — no public exports
├── README.md               # You are here
├── transcribe_service.py   # Orchestration: validate → detect language → route → transcribe
└── language_routing.py     # Pure routing rule: raw detected language → "en" | "bn"
```

---

## Module Responsibilities

### `transcribe_service.py`
The main orchestration module. Entry point: `run_transcription()`.

**Flow:**
```
audio_bytes + language
        |
        v
validate_audio()        <- raises UnsupportedFormatError / FileTooLargeError
        |
        v
language == "auto"?
    YES -> call GroqLanguageDetectorAdapter (with timeout + fallback)
             -> on failure, fall back to local FasterWhisper auto-detect
    NO  -> skip detection; use caller-supplied language as-is
        |
        v
resolve_routing_language()   <- from language_routing.py
        |
        v
get_transcription_adapter_for_language()   <- lazy singleton per language
        |
        v
adapter.transcribe()
        |
        v
TranscriptionResult (+ language_detected_by, raw_detected_language)
```

**Key functions:**

| Function | Purpose |
|---|---|
| `validate_audio(filename, size_bytes, max_mb)` | Validates extension + size. Pure, no side effects. |
| `run_transcription(audio_bytes, filename, language)` | Full two-stage pipeline. Catches detector failures and degrades gracefully. |
| `get_transcription_adapter_for_language(lang)` | Lazy singleton factory. Loads `en` or `bn` model on first use only. |

**Error handling rules:**
- `LanguageDetectionError` from Groq -> log warning + fall back to local auto-detect, never 500.
- Both Groq and local fallback fail -> raise `LanguageDetectionFatalError` (maps to 500 in `errors.py`).
- `UnsupportedFormatError` / `FileTooLargeError` -> re-raised to the API layer, maps to 400.

---

### `language_routing.py`
A single, pure, unit-tested function. No I/O, no settings, no side effects.

```python
resolve_routing_language(detected_language: str) -> Literal["en", "bn"]
```

**Routing rule (deliberate default-to-Bangla policy):**

| `detected_language` value | Routes to |
|---|---|
| `"en"` | `"en"` — primary `whisper-small` model |
| anything else (`"bn"`, `"hi"`, `"es"`, `""`, `"unknown"`, ...) | `"bn"` — `faster-whisper-bangla-small-int8` |

This is **not** a general multi-language router. It is a binary routing rule designed
for a codebase where audio traffic is overwhelmingly English or Bangla. Non-English,
non-Bangla audio will be transcribed by the Bangla-tuned model and may produce
degraded output — this is an accepted trade-off (see `DECISIONS.md` ADR #7).

---

## Layer Rules — Do Not Break These

| Rule | Why |
|---|---|
| **Zero FastAPI imports** in `services/` | Keeps business logic framework-agnostic and independently testable. |
| **Zero adapter imports at module level** | Adapters are lazily loaded to avoid model loading on import. |
| **No HTTP calls** directly from services | All external I/O goes through adapter classes. |
| **Raise domain exceptions, not HTTPException** | The API layer (`routes_transcribe.py` + `errors.py`) maps domain errors to HTTP status codes. |

---

## How to Add a New Service Module

1. Create `app/services/your_service.py`.
2. Keep it framework-agnostic — only stdlib + domain types.
3. Define one or more plain functions or classes.
4. Raise domain-specific exceptions (not `HTTPException`).
5. Add the new exception to HTTP mapping in `app/api/errors.py`.
6. Add tests in `tests/test_transcribe_api.py` (or a new focused file if the surface is large).
7. Update this README with the new module in the Module Map and Responsibilities tables.

---

## Extending the Routing Policy

If you need to add a third language (e.g. Hindi with its own model), the change is
isolated to two places only:

1. **`language_routing.py`** — extend `resolve_routing_language()` with a new branch.
2. **`transcribe_service.py`** — add the new adapter to `get_transcription_adapter_for_language()`.

No changes needed in the API layer or adapter base.
