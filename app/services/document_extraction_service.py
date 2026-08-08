"""Orchestration service for Endpoint 2 — Lab Report Extraction.

Layer constraints (mirrors transcribe_service.py):
  - Zero FastAPI imports. No UploadFile, no HTTPException, no Request.
  - Raise domain exceptions only. The route handler maps them to HTTP codes.
  - All heavy adapter imports are deferred (inside get_ocr_adapter()).
  - Lazy singleton for the real PaddleOCR adapter to avoid reloading the
    model on every request.

Pipeline:
  1. validate_document()     — extension + size check
  2. maybe_rasterize_pdf()   — PDF → PNG bytes (lazy pdf2image import)
  3. get_ocr_adapter()       — returns mock or PaddleOCR adapter
  4. adapter.extract()       — OCR → RawOCRResult
  5. classify_document()     — non-lab-report check → raise if not a lab report
  6. parse_header()          — RawOCRResult → ReportMeta
  7. parse_table()           — RawOCRResult → list[RawResultRow]
  8. _normalize_rows()       — RawResultRow → NormalizedResultRow
  9. Return ExtractionResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.adapters.ocr.base import RawOCRResult, DocumentOCRPort
from app.services.header_parser import ReportMeta, parse_header
from app.services.table_parser import RawResultRow, parse_table
from app.services.document_classifier import classify_document
from app.services.normalization import normalize_value, normalize_unit

logger = logging.getLogger(__name__)


# ── Domain exceptions ─────────────────────────────────────────────────────────

class DocumentUnsupportedFormatError(Exception):
    """File extension or content-type is not accepted."""


class DocumentTooLargeError(Exception):
    """File exceeds the maximum upload size."""


class DocumentCorruptError(Exception):
    """Image/PDF bytes are unreadable or corrupt."""


class NotALabReportError(Exception):
    """Document does not meet the minimum lab-report confidence threshold.

    Attributes:
        reason: Human-readable explanation from the classifier.
        confidence: Float 0–1 classifier score.
    """
    def __init__(self, reason: str, confidence: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.confidence = confidence


# ── Allowed extensions ────────────────────────────────────────────────────────

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS + (".pdf",)


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class NormalizedResultRow:
    """One result row after normalization — ready for the API response."""
    test_name: str
    value: float | None         # numeric value, or None for qualitative results
    unit: str
    reference_range: str
    flag: str
    raw_line: str               # SACRED — verbatim OCR text, never modified


@dataclass
class ExtractionResult:
    """Full extraction output passed from the service to the route handler."""
    meta: ReportMeta
    results: list[NormalizedResultRow]
    provider: str
    classifier_confidence: float


# ── Validation ────────────────────────────────────────────────────────────────

def validate_document(filename: str, size_bytes: int, max_mb: int) -> None:
    """Validate file extension and size.

    Raises:
        DocumentUnsupportedFormatError
        DocumentTooLargeError
    """
    parts = filename.lower().rsplit(".", 1)
    ext = ("." + parts[-1]) if len(parts) == 2 else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentUnsupportedFormatError(
            f"Unsupported document format: '{ext or 'unknown'}'. "
            f"Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if size_bytes > max_mb * 1024 * 1024:
        raise DocumentTooLargeError(
            f"Document exceeds {max_mb}MB limit ({size_bytes / 1024 / 1024:.1f}MB received)"
        )


def maybe_rasterize_pdf(image_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """If the file is a PDF, rasterize page 1 to PNG bytes.

    Uses pdf2image (lazily imported). Only page 1 is processed —
    multi-page PDFs are a known limitation documented in README.md.

    Returns:
        (image_bytes, effective_filename) — PNG bytes + updated filename.

    Raises:
        DocumentCorruptError: if pdf2image cannot open the PDF.
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext != "pdf":
        return image_bytes, filename

    try:
        from pdf2image import convert_from_bytes  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pdf2image is not installed. Run: pip install pdf2image"
        ) from e

    try:
        import io
        pages = convert_from_bytes(image_bytes, first_page=1, last_page=1, dpi=200)
        if not pages:
            raise DocumentCorruptError(f"PDF '{filename}' produced no pages.")
        buf = io.BytesIO()
        pages[0].save(buf, format="PNG")
        png_bytes = buf.getvalue()
        logger.info(
            f"PDF '{filename}' rasterized to PNG "
            f"({len(image_bytes)} → {len(png_bytes)} bytes, page 1 only)"
        )
        new_filename = filename.rsplit(".", 1)[0] + ".png"
        return png_bytes, new_filename
    except Exception as exc:
        raise DocumentCorruptError(
            f"PDF '{filename}' could not be rasterized: {exc}"
        ) from exc


# ── Adapter singleton ─────────────────────────────────────────────────────────

_paddle_adapter: Optional[DocumentOCRPort] = None


def get_ocr_adapter() -> DocumentOCRPort:
    """Return the configured OCR adapter (lazy singleton for PaddleOCR).

    Provider is read from settings.document_ocr_provider at call time.
    """
    global _paddle_adapter
    from app.config import settings

    if settings.document_ocr_provider == "mock":
        from app.adapters.ocr.mock_document_adapter import MockDocumentAdapter
        return MockDocumentAdapter(fixtures_dir=settings.document_fixtures_dir)

    # Real PaddleOCR — load once and cache
    if _paddle_adapter is None:
        from app.adapters.ocr.paddle_ocr_adapter import PaddleOCRAdapter
        logger.info("Lazy-loading PaddleOCRAdapter (CPU)")
        _paddle_adapter = PaddleOCRAdapter(use_gpu=False)
    return _paddle_adapter


def _reset_adapter_singleton() -> None:
    """Reset the PaddleOCR singleton — used in tests."""
    global _paddle_adapter
    _paddle_adapter = None


# ── Main orchestration ────────────────────────────────────────────────────────

def run_extraction(
    image_bytes: bytes,
    filename: str,
) -> ExtractionResult:
    """Run the full lab report extraction pipeline.

    Args:
        image_bytes:  Raw file bytes (image or PDF), already size-validated.
        filename:     Original upload filename.

    Returns:
        ExtractionResult with meta and normalized result rows.

    Raises:
        DocumentCorruptError:          Image/PDF cannot be decoded.
        NotALabReportError:            Classifier confidence below threshold.
        DocumentUnsupportedFormatError: (caller validates before calling this)
        DocumentTooLargeError:          (caller validates before calling this)
    """
    from app.config import settings

    # Step 1: Rasterize PDF if necessary
    image_bytes, filename = maybe_rasterize_pdf(image_bytes, filename)

    # Step 2: OCR
    adapter = get_ocr_adapter()
    try:
        ocr_result: RawOCRResult = adapter.extract(image_bytes, filename)
    except Exception as exc:
        # Re-raise as domain exception (keeps API layer clean)
        raise DocumentCorruptError(
            f"OCR failed for '{filename}': {exc}"
        ) from exc

    # Step 3: Parse table rows (needed for classifier)
    raw_rows = parse_table(ocr_result)

    # Step 4: Classify document
    classification = classify_document(ocr_result, raw_rows)
    if not classification.is_lab_report:
        raise NotALabReportError(
            reason=classification.reason,
            confidence=classification.confidence,
        )

    # Step 5: Parse header
    meta = parse_header(ocr_result)

    # Step 6: Normalize rows
    normalized = _normalize_rows(raw_rows)

    logger.info(
        f"Extraction complete: provider='{ocr_result.provider}', "
        f"{len(normalized)} result rows, classifier_confidence={classification.confidence:.3f}"
    )

    return ExtractionResult(
        meta=meta,
        results=normalized,
        provider=ocr_result.provider,
        classifier_confidence=classification.confidence,
    )


def _normalize_rows(raw_rows: list[RawResultRow]) -> list[NormalizedResultRow]:
    """Apply normalization to each raw result row."""
    normalized = []
    for row in raw_rows:
        val = normalize_value(row.value_raw)
        unit = normalize_unit(row.unit_raw)

        # Build the numeric value field:
        # - numeric if parsed and has a number
        # - None for qualitative (Positive, Nil, etc.) — Decision A
        numeric_value: float | None = val.numeric  # already None for qualitative

        normalized.append(NormalizedResultRow(
            test_name=row.test_name,
            value=numeric_value,
            unit=unit,
            reference_range=row.reference_range,
            flag=row.flag,
            raw_line=row.raw_line,  # SACRED — verbatim, never modified
        ))
    return normalized
