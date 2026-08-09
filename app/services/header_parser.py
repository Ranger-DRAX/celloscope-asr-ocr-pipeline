"""Header parser — extracts ReportMeta from a RawOCRResult.

Best-effort, field-by-field regex extraction from document text lines.
No guarantees: if a field isn't found the value is None (never fabricated).

Design principles:
- Works on lines only (not table_cells) — headers always appear as free text
  above the result table.
- Each field has its own regex; failure on one field doesn't block the others.
- Stops scanning once it sees what looks like the start of the results table
  (a line matching the column-header pattern).
- All text is stripped and whitespace-normalized before matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.adapters.ocr.base import RawOCRResult
from app.services.normalization import normalize_date


@dataclass
class ReportMeta:
    """Structured header fields extracted from a lab report."""
    patient_name: str | None
    age: str | None
    sex: str | None
    report_date: str | None   # ISO-8601 YYYY-MM-DD if parseable, else raw string
    lab_name: str | None
    reference_no: str | None


# ── Field patterns ────────────────────────────────────────────────────────────

_PATIENT_NAME_PAT = re.compile(
    r"(?:patient\s*(?:name)?|name)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
_AGE_PAT = re.compile(
    r"age\s*[:\-]\s*(.+?)(?:\s+(?:years?|yrs?|y\.?o\.?))?$",
    re.IGNORECASE,
)
_SEX_PAT = re.compile(
    r"(?:sex|gender)\s*[:\-]\s*(male|female|m|f|other)\b",
    re.IGNORECASE,
)
_DATE_PAT = re.compile(
    r"(?:date|report\s*date|collection\s*date|sample\s*date)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
_REFNO_PAT = re.compile(
    r"(?:ref(?:erence)?\.?\s*(?:no\.?|number|#)|lab\s*(?:no\.?|id))\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)

# Heuristic: these keywords in the first meaningful line suggest it's the lab name
_LAB_NAME_STOPWORDS = frozenset(
    {"patient", "age", "sex", "gender", "date", "ref", "lab", "test", "result",
     "report", "name:", "name :"}
)
# Typical result table header keywords — stop header scanning when seen
_TABLE_HEADER_KEYWORDS = re.compile(
    r"\b(?:test\s*name|investigation|parameter|analyte|result|value|reference\s*range)\b",
    re.IGNORECASE,
)


def parse_header(ocr: RawOCRResult) -> ReportMeta:
    """Extract header metadata from OCR lines.

    Scans lines in reading order and stops at the first line that looks like
    a results-table column header. Returns a ReportMeta with None for any
    field not found.
    """
    patient_name: str | None = None
    age: str | None = None
    sex: str | None = None
    report_date: str | None = None
    lab_name: str | None = None
    reference_no: str | None = None

    lab_name_candidate: str | None = None  # first meaningful text line

    for i, line in enumerate(sorted(ocr.lines, key=lambda l: l.order)):
        text = line.text.strip()
        if not text:
            continue

        # Stop scanning when we hit the results table header row
        if _TABLE_HEADER_KEYWORDS.search(text):
            break

        # Lab name heuristic: the first non-empty line is usually the lab name
        if lab_name_candidate is None:
            words_lower = text.lower()
            if not any(sw in words_lower for sw in _LAB_NAME_STOPWORDS):
                lab_name_candidate = text

        # Patient name
        if patient_name is None:
            m = _PATIENT_NAME_PAT.match(text)
            if m:
                patient_name = m.group(1).strip()

        # Age
        if age is None:
            m = _AGE_PAT.match(text)
            if m:
                age = m.group(1).strip()

        # Sex
        if sex is None:
            m = _SEX_PAT.search(text)
            if m:
                raw_sex = m.group(1).strip().lower()
                sex = {"m": "Male", "f": "Female"}.get(raw_sex, raw_sex.title())

        # Date
        if report_date is None:
            m = _DATE_PAT.match(text)
            if m:
                raw_date = m.group(1).strip()
                normalized = normalize_date(raw_date)
                report_date = normalized if normalized else raw_date

        # Reference number
        if reference_no is None:
            m = _REFNO_PAT.match(text)
            if m:
                reference_no = m.group(1).strip()

    # Use candidate for lab name if not found via keywords
    if lab_name is None:
        lab_name = lab_name_candidate

    return ReportMeta(
        patient_name=patient_name,
        age=age,
        sex=sex,
        report_date=report_date,
        lab_name=lab_name,
        reference_no=reference_no,
    )
