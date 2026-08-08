"""PaddleOCR adapter — real OCR using PaddleOCR 3.x PP-StructureV3 (CPU default).

Written against paddleocr 3.x / paddlex 3.x (the versions pinned in
requirements.txt). Key differences from the old 2.x API:
    - ``PaddleOCR(...)`` no longer accepts ``use_gpu``/``show_log``; use
      ``device="cpu"`` (or ``"gpu"``).  ``use_angle_cls`` is auto-aliased to
      ``use_textline_orientation``.
    - ``PPStructure`` was renamed ``PPStructureV3``; its predictions are
      paddlex ``LayoutParsingResultV2`` objects (keys ``parsing_res_list``,
      ``table_res_list``), not the old ``[{type, bbox, res}, ...]`` list.
    - Plain OCR returns one ``OCRResult`` per page carrying ``rec_texts``/
      ``rec_scores``/``rec_boxes``, not the old ``[[pts, (text, score)]]``.

All imports of paddleocr / paddlex are lazy (inside the method body) so the
module can be imported in any environment — including those where paddleocr
is not installed — without raising an ImportError at startup.

CPU vs GPU decision (DECISIONS.md §8):
    PaddleOCR defaults to CPU here. The GTX 1050 Ti's 4GB VRAM is already
    budgeted to the faster-whisper models on Endpoint 1. Running PaddleOCR on
    GPU would require the paddlepaddle-gpu wheel AND contend for VRAM. CPU
    inference is slower but keeps the service startable on any machine.

Table detection:
    PP-StructureV3 is used when available for structured table extraction.
    If it fails or is unavailable, the adapter falls back to plain PaddleOCR
    line detection (graceful degradation — results still usable, just
    without structured table_cells).

Raises:
    CorruptImageError: if the image bytes cannot be decoded by cv2.
"""

from __future__ import annotations

import logging
import os
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from app.adapters.ocr.base import DocumentOCRPort, RawOCRResult, OCRLine, OCRTableCell

logger = logging.getLogger(__name__)


class CorruptImageError(Exception):
    """Raised when the image bytes cannot be decoded by the OCR engine."""


class PaddleOCRAdapter:
    """Real OCR adapter using PaddleOCR 3.x (CPU-first, lazy import).

    Instantiate once and reuse — PaddleOCR model loads on first call to
    extract() and is cached in self._ocr / self._structure thereafter.
    """

    def __init__(self, use_gpu: bool = False, lang: str = "en") -> None:
        self._use_gpu = use_gpu
        self._lang = lang
        self._ocr = None        # lazy-loaded PaddleOCR instance
        self._structure = None  # lazy-loaded PPStructureV3 instance

    def _device(self) -> str:
        """Map the legacy use_gpu flag onto paddleocr 3.x's device arg."""
        return "gpu" if self._use_gpu else "cpu"

    def _load_ocr(self):
        """Lazy-load PaddleOCR 3.x on first use."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore
                logger.info(
                    f"Loading PaddleOCR (device={self._device()}, lang={self._lang})"
                )
                # NOTE: use_gpu / show_log were removed in 3.x. The OLD
                # use_angle_cls=True is now use_textline_orientation=True.
                self._ocr = PaddleOCR(
                    lang=self._lang,
                    use_textline_orientation=True,
                    device=self._device(),
                )
            except ImportError as e:
                raise ImportError(
                    "paddleocr is not installed. Run: pip install paddleocr paddlepaddle"
                ) from e
        return self._ocr

    def _load_structure(self):
        """Lazy-load PPStructureV3 for table detection (3.x API)."""
        if self._structure is None:
            try:
                from paddleocr import PPStructureV3  # type: ignore
                logger.info(
                    f"Loading PPStructureV3 (device={self._device()}, lang={self._lang})"
                )
                self._structure = PPStructureV3(
                    lang=self._lang,
                    use_table_recognition=True,
                    use_textline_orientation=True,
                    device=self._device(),
                )
            except Exception as e:
                logger.warning(
                    f"PPStructureV3 unavailable ({e}); table detection disabled — "
                    "will use plain PaddleOCR line detection only."
                )
                self._structure = "unavailable"  # sentinel: don't retry
        return None if self._structure == "unavailable" else self._structure

    def extract(self, image_bytes: bytes, filename: str) -> RawOCRResult:
        """Run OCR on image bytes and return a RawOCRResult.

        Writes a temporary file so cv2 can decode the image, runs
        PP-StructureV3 for table detection (3.x predict returns one page
        result per call), falls back to plain PaddleOCR if PP-StructureV3 is
        unavailable, then cleans up the temp file.

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

        # Write image to a temp file (needed for cv2's path-based decode)
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

            # Try PP-StructureV3 first for structured table detection
            structure = self._load_structure()
            if structure is not None:
                try:
                    pages = structure.predict(img)
                    if pages:
                        raw_output["pp_structure"] = _structure_result_to_dict(pages[0])
                        lines, table_cells = _parse_structure_result(pages[0])
                except Exception as e:
                    logger.warning(
                        f"PPStructureV3 inference failed ({e}); "
                        "falling back to plain PaddleOCR"
                    )

            # Plain PaddleOCR fallback (or primary if PPStructure unavailable)
            if not lines:
                ocr = self._load_ocr()
                try:
                    ocr_result = ocr.predict(img)
                    raw_output["paddle_ocr"] = "predict"
                except Exception as e:
                    logger.warning(f"ocr.predict() failed ({e}); falling back to ocr.ocr()")
                    try:
                        ocr_result = ocr.ocr(img)
                        raw_output["paddle_ocr"] = "ocr_fallback"
                    except Exception as e2:
                        logger.error(f"ocr.ocr() failed: {e2}")
                        ocr_result = None

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


# ── PaddleOCR 3.x result parsers ──────────────────────────────────────────────

def _flatten_bbox(box):
    """Flatten a bbox to [x1, y1, x2, y2].

    Handles both flat [x1,y1,x2,y2] boxes (OCRResult.rec_boxes) and
    4-point polygons (rec_polys). Returns None if the box is unusable.
    """
    if box is None:
        return None
    try:
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            return [float(x) for x in box]
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return [min(xs), min(ys), max(xs), max(ys)]
    except (TypeError, IndexError, ValueError):
        return None


def _parse_ocr_result(ocr_result) -> list[OCRLine]:
    """Parse plain PaddleOCR output (handles both 3.x OCRResult and classic ocr() lists)."""
    detections: list[tuple[float, float, str, float | None, list[float] | None]] = []
    for page in (ocr_result or []):
        if page is None:
            continue

        # Format 1: dict or OCRResult object (PaddleX 3.x API)
        if isinstance(page, dict) or hasattr(page, "get") or hasattr(page, "rec_texts"):
            texts = getattr(page, "rec_texts", None) if not isinstance(page, dict) else page.get("rec_texts")
            scores = getattr(page, "rec_scores", None) if not isinstance(page, dict) else page.get("rec_scores")
            boxes = getattr(page, "rec_boxes", None) if not isinstance(page, dict) else page.get("rec_boxes")

            # Check inside 'doc_preprocessor_res' or dict keys if needed
            if not texts and isinstance(page, dict):
                texts = page.get("rec_texts") or page.get("texts") or []
                scores = page.get("rec_scores") or page.get("scores") or []
                boxes = page.get("rec_boxes") or page.get("boxes") or []

            if texts:
                for i, text in enumerate(texts):
                    conf = float(scores[i]) if scores and i < len(scores) else None
                    bbox = _flatten_bbox(boxes[i] if boxes and i < len(boxes) else None)
                    y = bbox[1] if bbox else 0.0
                    x = bbox[0] if bbox else 0.0
                    detections.append((y, x, text or "", conf, bbox))
                continue

        # Format 2: Classic PaddleOCR output list [[[box], (text, conf)], ...]
        if isinstance(page, (list, tuple)):
            for line in page:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    box_pts = line[0]
                    text_conf = line[1]
                    text = ""
                    conf = None
                    if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                        text = str(text_conf[0])
                        if len(text_conf) >= 2 and isinstance(text_conf[1], (int, float)):
                            conf = float(text_conf[1])
                    bbox = _flatten_bbox(box_pts)
                    y = bbox[1] if bbox else 0.0
                    x = bbox[0] if bbox else 0.0
                    detections.append((y, x, text, conf, bbox))

    # Reading order: top-to-bottom, then left-to-right
    detections.sort(key=lambda d: (d[0], d[1]))
    return [
        OCRLine(text=t, confidence=c, bbox=bb, order=i)
        for i, (_, _, t, c, bb) in enumerate(detections)
    ]


# ── PP-StructureV3 (LayoutParsingResultV2) parser ─────────────────────────────

class _TableHTMLParser(HTMLParser):
    """Tiny HTML-table parser producing a row-major grid of cell texts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th"):
            if self._cell is not None:
                text = "".join(self._cell).strip()
                if self._row is not None:
                    self._row.append(text)
            self._cell = None
        elif tag == "tr":
            if self._row is not None:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _parse_table_html(html: str) -> list[list[str]]:
    """Convert a table's pred_html into a row-major grid of cell texts."""
    parser = _TableHTMLParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        return []
    return parser.rows


def _parse_structure_result(page) -> tuple[list[OCRLine], list[OCRTableCell]]:
    """Parse one PPStructureV3 page result into (lines, table_cells).

    ``page`` is a paddlex LayoutParsingResultV2 (dict-like): its
    ``parsing_res_list`` holds LayoutBlock objects (label/bbox/content/
    order_index) and its ``table_res_list`` holds per-table results whose
    ``.json["res"]`` exposes ``pred_html`` and ``cell_box_list``.
    """
    lines: list[OCRLine] = []
    table_cells: list[OCRTableCell] = []
    order = 0

    blocks = list(page.get("parsing_res_list") or [])
    blocks.sort(key=lambda b: b.order_index if b.order_index is not None else 0)

    table_region_bboxes: list[list[float]] = []
    for block in blocks:
        bbox = list(block.bbox or []) if block.bbox else None
        label = (block.label or "").lower()
        if label in ("table", "table_body"):
            table_region_bboxes.append(bbox or [0, 0, 0, 0])
            continue
        content = block.content or ""
        for text in content.splitlines():
            text = text.strip()
            if not text:
                continue
            lines.append(OCRLine(text=text, confidence=None, bbox=bbox, order=order))
            order += 1

    for i, table in enumerate(page.get("table_res_list") or []):
        try:
            tres = table.json["res"]
        except Exception:
            tres = {}
        pred_html = tres.get("pred_html")
        if not pred_html:
            continue
        cell_boxes = [list(b) for b in (tres.get("cell_box_list") or [])]
        region_bbox = (
            table_region_bboxes[i]
            if i < len(table_region_bboxes)
            else [0, 0, 0, 0]
        )

        grid = _parse_table_html(pred_html)
        cell_count = sum(len(row) for row in grid)
        use_cell_boxes = len(cell_boxes) == cell_count
        k = 0
        for row_idx, row in enumerate(grid):
            for col_idx, text in enumerate(row):
                cell_text = (text or "").strip()
                bbox = cell_boxes[k] if use_cell_boxes else region_bbox
                k += 1
                table_cells.append(OCRTableCell(
                    row_index=row_idx,
                    col_index=col_idx,
                    text=cell_text,
                    confidence=None,
                    bbox=bbox,
                ))
                # Also add as a line so the full text is available to parsers
                if cell_text:
                    lines.append(
                        OCRLine(text=cell_text, confidence=None, bbox=bbox, order=order)
                    )
                    order += 1

    return lines, table_cells


def _structure_result_to_dict(page) -> list:
    """Convert a PPStructureV3 page result to a JSON-serializable summary."""
    output = []
    for block in (page.get("parsing_res_list") or []):
        output.append({
            "type": block.label,
            "bbox": list(block.bbox) if block.bbox else None,
        })
    for table in (page.get("table_res_list") or []):
        try:
            tres = table.json["res"]
        except Exception:
            tres = {}
        output.append({
            "type": "table",
            "html": tres.get("pred_html"),
            "cell_boxes": [
                list(b) for b in (tres.get("cell_box_list") or [])
            ],
        })
    return output
