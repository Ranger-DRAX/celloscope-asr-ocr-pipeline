"""PaddleOCR adapter — real OCR using PaddleOCR PP-Structure V3 (CPU default).

All imports of paddleocr and paddlepaddle are lazy (inside the method body)
so that the module can be imported in any environment — including those where
paddleocr is not installed — without raising an ImportError at startup.

CPU vs GPU decision (DECISIONS.md §8):
    PaddleOCR defaults to CPU here. The GTX 1050 Ti's 4GB VRAM is already
    budgeted to the faster-whisper models on Endpoint 1. Running PaddleOCR on
    GPU would require the paddlepaddle-gpu wheel AND contend for VRAM. CPU
    inference is slower but keeps the service startable on any machine.

Table detection:
    PP-Structure V3 is used when available for structured table extraction.
    If PP-Structure fails or is unavailable, the adapter falls back to plain
    PaddleOCR line detection (graceful degradation — results still usable,
    just without structured table_cells).

Raises:
    CorruptImageError: if the image bytes cannot be decoded by cv2.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from app.adapters.ocr.base import DocumentOCRPort, RawOCRResult, OCRLine, OCRTableCell

logger = logging.getLogger(__name__)


class CorruptImageError(Exception):
    """Raised when the image bytes cannot be decoded by the OCR engine."""


class PaddleOCRAdapter:
    """Real OCR adapter using PaddleOCR (CPU-first, lazy import).

    Instantiate once and reuse — PaddleOCR model loads on first call to
    extract() and is cached in self._ocr / self._structure thereafter.
    """

    def __init__(self, use_gpu: bool = False, lang: str = "en") -> None:
        self._use_gpu = use_gpu
        self._lang = lang
        self._ocr = None        # lazy-loaded PaddleOCR instance
        self._structure = None  # lazy-loaded PPStructure instance

    def _load_ocr(self):
        """Lazy-load PaddleOCR on first use."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore
                logger.info(
                    f"Loading PaddleOCR (use_gpu={self._use_gpu}, lang={self._lang})"
                )
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self._lang,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
            except ImportError as e:
                raise ImportError(
                    "paddleocr is not installed. Run: pip install paddleocr paddlepaddle"
                ) from e
        return self._ocr

    def _load_structure(self):
        """Lazy-load PPStructure for table detection."""
        if self._structure is None:
            try:
                from paddleocr import PPStructure  # type: ignore
                logger.info(
                    f"Loading PPStructure V3 (use_gpu={self._use_gpu})"
                )
                self._structure = PPStructure(
                    table=True,
                    ocr=True,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
            except (ImportError, Exception) as e:
                logger.warning(
                    f"PPStructure unavailable ({e}); table detection disabled — "
                    "will use plain PaddleOCR line detection only."
                )
                self._structure = "unavailable"  # sentinel: don't retry
        return None if self._structure == "unavailable" else self._structure

    def extract(self, image_bytes: bytes, filename: str) -> RawOCRResult:
        """Run OCR on image bytes and return a RawOCRResult.

        Writes a temporary file (required by PaddleOCR's file-path API),
        runs PP-Structure for table detection, falls back to plain PaddleOCR
        if PP-Structure is unavailable, then cleans up the temp file.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, WebP, BMP, or PDF
                         already rasterized to PNG by the validation layer).
            filename:    Original filename (used for temp file suffix only).

        Returns:
            RawOCRResult with lines and table_cells populated.

        Raises:
            CorruptImageError: if cv2 or PaddleOCR cannot decode the image.
        """
        suffix = Path(filename).suffix or ".png"

        # Write image to a temp file (PaddleOCR requires a file path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            img = cv2.imread(tmp_path)
            if img is None:
                # Try via numpy buffer for formats cv2.imread can't auto-detect
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise CorruptImageError(
                    f"Image bytes for '{filename}' could not be decoded by cv2. "
                    "File may be corrupt or in an unsupported format."
                )

            h, w = img.shape[:2]
            lines: list[OCRLine] = []
            table_cells: list[OCRTableCell] = []
            raw_output: dict = {}

            # Try PP-Structure first for table detection
            structure = self._load_structure()
            if structure is not None:
                try:
                    result = structure(img)
                    raw_output["pp_structure"] = _structure_result_to_dict(result)
                    lines, table_cells = _parse_structure_result(result)
                except Exception as e:
                    logger.warning(
                        f"PPStructure inference failed ({e}); "
                        "falling back to plain PaddleOCR"
                    )
                    result = None

            # Plain PaddleOCR fallback (or primary if PPStructure unavailable)
            if not lines:
                ocr = self._load_ocr()
                ocr_result = ocr.ocr(tmp_path, cls=True)
                raw_output["paddle_ocr"] = ocr_result
                lines = _parse_ocr_result(ocr_result)

            logger.info(
                f"PaddleOCRAdapter: extracted {len(lines)} lines, "
                f"{len(table_cells)} table cells from '{filename}'"
            )
            return RawOCRResult(
                lines=lines,
                table_cells=table_cells,
                page_width=w,
                page_height=h,
                provider=f"paddleocr-{'gpu' if self._use_gpu else 'cpu'}",
                raw_provider_output=raw_output,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── PaddleOCR result parsers ──────────────────────────────────────────────────

def _parse_ocr_result(ocr_result) -> list[OCRLine]:
    """Parse plain PaddleOCR output into OCRLine list."""
    lines = []
    order = 0
    for page in (ocr_result or []):
        if not page:
            continue
        for detection in page:
            if not detection or len(detection) < 2:
                continue
            bbox_raw, (text, conf) = detection[0], detection[1]
            # bbox_raw is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] — flatten to [x1,y1,x2,y2]
            try:
                xs = [pt[0] for pt in bbox_raw]
                ys = [pt[1] for pt in bbox_raw]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            except (TypeError, IndexError):
                bbox = None
            lines.append(OCRLine(text=text or "", confidence=float(conf or 0), bbox=bbox, order=order))
            order += 1
    return lines


def _parse_structure_result(result) -> tuple[list[OCRLine], list[OCRTableCell]]:
    """Parse PPStructure output into (lines, table_cells)."""
    lines: list[OCRLine] = []
    table_cells: list[OCRTableCell] = []
    order = 0

    for region in (result or []):
        region_type = region.get("type", "").lower()
        res = region.get("res", [])
        bbox_raw = region.get("bbox", [0, 0, 0, 0])

        if region_type == "table":
            # res is a list of table cell dicts from PPStructure
            html = region.get("res", {})
            if isinstance(html, dict):
                cells_raw = html.get("cells", [])
            else:
                cells_raw = []

            for cell in cells_raw:
                row_idx = cell.get("row_start", 0)
                col_idx = cell.get("col_start", 0)
                text = cell.get("text", "").strip()
                conf = cell.get("score")
                table_cells.append(
                    OCRTableCell(
                        row_index=row_idx,
                        col_index=col_idx,
                        text=text,
                        confidence=float(conf) if conf is not None else None,
                        bbox=bbox_raw,
                    )
                )
                # Also add as a line so the full text is available to parsers
                if text:
                    lines.append(OCRLine(text=text, confidence=float(conf) if conf else None,
                                         bbox=bbox_raw, order=order))
                    order += 1
        else:
            # Text / title / figure caption — add as plain lines
            for detection in (res or []):
                if not detection or len(detection) < 2:
                    continue
                bbox_pts, (text, conf) = detection[0], detection[1]
                try:
                    xs = [pt[0] for pt in bbox_pts]
                    ys = [pt[1] for pt in bbox_pts]
                    flat_bbox = [min(xs), min(ys), max(xs), max(ys)]
                except (TypeError, IndexError):
                    flat_bbox = bbox_raw
                lines.append(OCRLine(text=text or "", confidence=float(conf or 0),
                                     bbox=flat_bbox, order=order))
                order += 1

    return lines, table_cells


def _structure_result_to_dict(result) -> list:
    """Convert PPStructure result to a JSON-serializable dict for raw_provider_output."""
    output = []
    for region in (result or []):
        entry = {
            "type": region.get("type"),
            "bbox": region.get("bbox"),
        }
        output.append(entry)
    return output
