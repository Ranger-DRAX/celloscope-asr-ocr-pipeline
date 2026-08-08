"""Mock OCR adapter — replays frozen fixture JSON files.

No model load, no network call, no GPU. Fixtures live under
testdata/fixtures/documents/ and are keyed by fixture_name (filename stem).

Fixture files are plain JSON serializations of RawOCRResult.  They are
generated once from the real MistralOCRAdapter and committed — they must not
be regenerated at request time (see DECISIONS.md).

Usage (controlled via DOCUMENT_OCR_PROVIDER=mock in settings):
    The fixture is selected by the filename stem of the uploaded image.
    If no matching fixture exists, the first available fixture is used
    (so unit tests that only care about schema shape don't need a named fixture).

This keeps the mock deterministic: same input filename → same output, always.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.adapters.ocr.base import DocumentOCRPort, RawOCRResult, OCRLine, OCRTableCell

logger = logging.getLogger(__name__)


class MockDocumentAdapter:
    """OCR adapter that replays frozen fixture JSON files.

    Satisfies DocumentOCRPort without loading any model or making any
    network call. Safe to use in CI, docker-compose, and unit tests.
    """

    def __init__(self, fixtures_dir: str = "testdata/fixtures/documents") -> None:
        self._fixtures_dir = Path(fixtures_dir)

    def extract(self, image_bytes: bytes, filename: str) -> RawOCRResult:
        """Return a RawOCRResult by replaying a frozen fixture JSON.

        Selection logic:
        1. Look for <stem>.json where stem = Path(filename).stem.
        2. If not found, use the first *.json file in the fixtures directory.
        3. If the directory is empty, return an empty RawOCRResult (safe degradation).
        """
        stem = Path(filename).stem
        fixture_path = self._fixtures_dir / f"{stem}.json"

        if not fixture_path.exists():
            # Fall back to the first available fixture
            candidates = sorted(self._fixtures_dir.glob("*.json"))
            if not candidates:
                logger.warning(
                    f"MockDocumentAdapter: no fixture found in {self._fixtures_dir}; "
                    "returning empty RawOCRResult"
                )
                return RawOCRResult(provider="mock")
            fixture_path = candidates[0]
            logger.debug(
                f"MockDocumentAdapter: no fixture for '{stem}', using '{fixture_path.name}'"
            )
        else:
            logger.debug(f"MockDocumentAdapter: replaying fixture '{fixture_path.name}'")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return _deserialize(data)


def _deserialize(data: dict) -> RawOCRResult:
    """Deserialize a fixture dict to a RawOCRResult."""
    lines = [
        OCRLine(
            text=cell.get("text", ""),
            confidence=cell.get("confidence"),
            bbox=cell.get("bbox"),
            order=cell.get("order", i),
        )
        for i, cell in enumerate(data.get("lines", []))
    ]
    table_cells = [
        OCRTableCell(
            row_index=cell.get("row_index", 0),
            col_index=cell.get("col_index", 0),
            text=cell.get("text", ""),
            confidence=cell.get("confidence"),
            bbox=cell.get("bbox"),
        )
        for cell in data.get("table_cells", [])
    ]
    return RawOCRResult(
        lines=lines,
        table_cells=table_cells,
        page_width=data.get("page_width", 0),
        page_height=data.get("page_height", 0),
        provider=data.get("provider", "mock"),
        raw_provider_output=data.get("raw_provider_output"),
    )
