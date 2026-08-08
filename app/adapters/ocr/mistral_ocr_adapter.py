"""Mistral OCR Adapter for Document Extraction.

Satisfies DocumentOCRPort. Replaces PaddleOCR.
Uploads document to Mistral OCR API, receives markdown, and parses
it into RawOCRResult (extracting Markdown tables as table cells).
"""

from __future__ import annotations

import base64
import logging
import io
import re

from app.adapters.ocr.base import DocumentOCRPort, RawOCRResult, OCRLine, OCRTableCell

logger = logging.getLogger(__name__)

class MistralOCRAdapter(DocumentOCRPort):
    def __init__(self, api_key: str | None = None) -> None:
        if not api_key:
            raise ValueError("MISTRAL_API_KEY must be provided for MistralOCRAdapter.")
        self.api_key = api_key

        try:
            from mistralai.client import Mistral
            self.client = Mistral(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("mistralai SDK not installed. Run: pip install mistralai") from e

    def extract(self, image_bytes: bytes, filename: str) -> RawOCRResult:
        """Call Mistral OCR API and parse markdown to RawOCRResult."""
        try:
            # Mistral OCR supports pdf and images. We must determine type from filename.
            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "jpeg"
            # Normalize extension for mistral (pdf, jpeg, png, etc.)
            mime_type = "application/pdf" if ext == "pdf" else f"image/{ext if ext != 'jpg' else 'jpeg'}"

            document_data = base64.b64encode(image_bytes).decode('utf-8')
            data_url = f"data:{mime_type};base64,{document_data}"

            logger.info(f"Calling Mistral OCR for document '{filename}'...")
            # Note: We use mistral-ocr-latest model
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": data_url,
                }
            )

            # Mistral returns Pages, we combine them
            markdown_content = ""
            if hasattr(ocr_response, "pages"):
                for page in ocr_response.pages:
                    if hasattr(page, "markdown"):
                        markdown_content += page.markdown + "\n\n"
            else:
                # Fallback if structure is different
                markdown_content = getattr(ocr_response, "content", str(ocr_response))

            logger.debug(f"Mistral OCR returned markdown:\n{markdown_content}")
            return self._parse_markdown_to_raw_result(markdown_content, ocr_response.model_dump())
        except Exception as exc:
            raise Exception(f"Mistral OCR failed: {exc}") from exc

    def _parse_markdown_to_raw_result(self, markdown: str, raw_response: dict) -> RawOCRResult:
        """Parses Mistral markdown into OCRLines and OCRTableCells."""
        lines = []
        table_cells = []

        md_lines = markdown.splitlines()
        
        in_table = False
        row_idx = 0
        order = 0

        for line in md_lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Detect markdown table row (starts and ends with '|' usually, or just contains '|')
            # A simple heuristic: contains '|' and at least one other column separator
            if '|' in line_str and line_str.startswith('|') and line_str.endswith('|'):
                # We are in a table row
                in_table = True
                
                # Check if it's a separator row e.g. |---|---|
                if re.match(r"^\|(?:[:\-\s]*\|)+$", line_str):
                    continue # Skip separator row
                
                cells = [c.strip() for c in line_str.strip('|').split('|')]
                for col_idx, cell_text in enumerate(cells):
                    table_cells.append(
                        OCRTableCell(
                            row_index=row_idx,
                            col_index=col_idx,
                            text=cell_text,
                            confidence=None,
                            bbox=None,
                        )
                    )
                row_idx += 1
                
                # Still add the raw line to OCR lines so it is available as a whole line
                lines.append(OCRLine(text=line_str, confidence=None, bbox=None, order=order))
                order += 1
            else:
                # Not a table row
                if in_table:
                    # Reset table state if we left a table
                    in_table = False
                    # We do not reset row_idx because we accumulate all table cells across the document.
                    # Or we could reset row_idx, but the parser might expect one big list of cells. 
                    # The table_parser groups by row_index. 
                    # To avoid merging two distinct tables, we bump row_idx slightly to leave a gap.
                    row_idx += 10 

                lines.append(OCRLine(text=line_str, confidence=None, bbox=None, order=order))
                order += 1

        return RawOCRResult(
            lines=lines,
            table_cells=table_cells,
            provider="mistral_ocr",
            raw_provider_output=raw_response,
        )
