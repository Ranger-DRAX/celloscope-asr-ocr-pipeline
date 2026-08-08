"""OCR adapter contract for Endpoint 2 — Lab Report Extraction.

This module defines the shared data types and Protocol that both the real
PaddleOCR adapter and the mock adapter must satisfy. Nothing downstream of
the adapter boundary (services, API) may import from a specific adapter
implementation — they depend on this interface only.

RawOCRResult is the single handoff type between adapters and services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# ── Primitive OCR output types ────────────────────────────────────────────────

@dataclass
class OCRLine:
    """A single line of text detected by the OCR engine.

    Attributes:
        text:       The verbatim text content of the line.
        confidence: Provider-reported confidence score (0.0–1.0), or None.
        bbox:       Bounding box [x1, y1, x2, y2] in image pixels, or None.
        order:      Reading order index (top-to-bottom, left-to-right). Used
                    to reconstruct document structure when the provider returns
                    unordered results.
    """
    text: str
    confidence: float | None = None
    bbox: list[float] | None = None
    order: int = 0


@dataclass
class OCRTableCell:
    """A single cell within a table structure detected by PP-Structure.

    Attributes:
        row_index:  0-based row index within the table.
        col_index:  0-based column index within the table.
        text:       Verbatim cell text content.
        confidence: Provider-reported confidence score, or None.
        bbox:       Bounding box in image pixels, or None.
    """
    row_index: int
    col_index: int
    text: str
    confidence: float | None = None
    bbox: list[float] | None = None


@dataclass
class RawOCRResult:
    """Complete raw output from one OCR pass on a single document image.

    This is the only type that crosses the adapter → service boundary.
    Services must not import any adapter-specific types.

    Attributes:
        lines:          All text lines detected in reading order (full page,
                        including header, footer, and any non-table text).
        table_cells:    Structured table cells from PP-Structure (empty list
                        if the adapter does not support table detection or if
                        no table was found). When present, these are the
                        canonical source for test result rows.
        page_width:     Image width in pixels (0 if unknown).
        page_height:    Image height in pixels (0 if unknown).
        provider:       Adapter label, e.g. "paddleocr-cpu", "mock".
        raw_provider_output: The full, unmodified provider output (for
                        debugging and fixture generation). May be None for the
                        mock adapter.
    """
    lines: list[OCRLine] = field(default_factory=list)
    table_cells: list[OCRTableCell] = field(default_factory=list)
    page_width: int = 0
    page_height: int = 0
    provider: str = "unknown"
    raw_provider_output: dict | None = None


# ── Adapter Protocol ──────────────────────────────────────────────────────────

class DocumentOCRPort(Protocol):
    """Protocol that all OCR adapters must satisfy.

    Implement this Protocol to add a new OCR backend. The real implementation
    lives in paddle_ocr_adapter.py; the mock in mock_document_adapter.py.
    Both must return a RawOCRResult with the same schema.

    Raises:
        CorruptImageError: if the image bytes cannot be decoded at all.
        Any provider-specific network/model error should be wrapped and
        re-raised as a CorruptImageError with a descriptive message.
    """

    def extract(self, image_bytes: bytes, filename: str) -> RawOCRResult:
        """Run OCR on the given image bytes and return structured output.

        Args:
            image_bytes: Raw bytes of the image (JPEG, PNG, WebP, BMP, PDF
                         already rasterized to PNG).
            filename:    Original filename (used for logging and temp files).

        Returns:
            RawOCRResult with lines in reading order and table_cells if
            the adapter supports table structure detection.
        """
        ...
