"""Document classifier — cheap heuristic to detect non-lab-report inputs.

Runs after parsing, on the already-extracted lines + result rows. The
heuristic is cheap (regex pattern counting) — no model, no network.

Policy (DECISIONS.md §6):
    Option (b) selected: return HTTP 422 with a structured reason string when
    the document does not look like a lab report. This is consistent with
    Endpoint 1's silence detection (clear signal = actionable error) and
    prevents the service from silently returning empty/garbage results for
    completely wrong inputs like invoices, newspaper photos, or blank pages.

Confidence score components (each 0–1, weighted sum):
  - result_row_ratio:  fraction of lines that look like numeric result rows
  - header_hit_ratio: fraction of expected header labels found (Patient Name,
                      Age, Sex, Date, Reference No, Lab Name)
  - has_numeric_table: at least 2 numeric result rows found (binary)

Thresholds are deliberately generous to avoid false positives on degraded scans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.adapters.ocr.base import RawOCRResult
from app.services.table_parser import RawResultRow


@dataclass
class ClassificationResult:
    """Outcome of the document classifier heuristic.

    Attributes:
        is_lab_report:  True if the document looks like a lab report.
        confidence:     Float 0.0–1.0. Not a probability — just a score.
        reason:         Human-readable explanation (used in 422 responses).
    """
    is_lab_report: bool
    confidence: float
    reason: str


# ── Heuristic configuration ───────────────────────────────────────────────────

# Minimum number of numeric result rows required to be considered a lab report
_MIN_NUMERIC_ROWS = 2

# Confidence threshold: below this → not a lab report
_CONFIDENCE_THRESHOLD = 0.25

# Lab report header label keywords
_HEADER_LABELS = [
    re.compile(r"\bpatient\b", re.IGNORECASE),
    re.compile(r"\bage\b", re.IGNORECASE),
    re.compile(r"\b(?:sex|gender)\b", re.IGNORECASE),
    re.compile(r"\bdate\b", re.IGNORECASE),
    re.compile(r"\b(?:ref(?:erence)?[\s.]*no|lab[\s.]*(?:no|id))\b", re.IGNORECASE),
]

# Pattern that looks like a numeric result row value
_NUMERIC_VALUE_PAT = re.compile(
    r"[<>~≤≥]?\s*\d[\d,\.]*(?:\s*[xX×]\s*10\^?\d+)?|\bpositive\b|\bnegative\b|\bnil\b|\btrace\b|\breactive\b",
    re.IGNORECASE,
)


def classify_document(
    ocr: RawOCRResult,
    result_rows: list[RawResultRow],
) -> ClassificationResult:
    """Apply cheap heuristics to decide if the document is a lab report.

    Args:
        ocr:          Full OCR output (lines + table_cells).
        result_rows:  Rows already extracted by table_parser (may be empty).

    Returns:
        ClassificationResult — the caller decides how to act on is_lab_report.
    """
    all_lines = [l.text.strip() for l in ocr.lines if l.text.strip()]
    total_lines = len(all_lines)

    if total_lines == 0:
        return ClassificationResult(
            is_lab_report=False,
            confidence=0.0,
            reason="No text detected in the document.",
        )

    # Component 1: How many header labels are found in the full text?
    full_text = " ".join(all_lines)
    header_hits = sum(1 for pat in _HEADER_LABELS if pat.search(full_text))
    header_hit_ratio = header_hits / len(_HEADER_LABELS)

    # Component 2: Numeric result rows
    numeric_row_count = sum(
        1 for row in result_rows
        if _NUMERIC_VALUE_PAT.search(row.value_raw) or row.value_raw.strip()
    )
    has_enough_rows = numeric_row_count >= _MIN_NUMERIC_ROWS

    # Component 3: Fraction of lines that look like result rows
    lines_looking_like_results = sum(
        1 for line in all_lines if _NUMERIC_VALUE_PAT.search(line)
    )
    result_line_ratio = min(lines_looking_like_results / max(total_lines, 1), 1.0)

    # Weighted confidence
    confidence = (
        0.45 * header_hit_ratio
        + 0.35 * (1.0 if has_enough_rows else 0.0)
        + 0.20 * result_line_ratio
    )

    is_lab_report = confidence >= _CONFIDENCE_THRESHOLD

    if is_lab_report:
        reason = (
            f"Classified as lab report (confidence={confidence:.2f}): "
            f"{header_hits}/{len(_HEADER_LABELS)} header labels found, "
            f"{numeric_row_count} result rows."
        )
    else:
        reason = (
            f"Document does not appear to be a lab report (confidence={confidence:.2f}): "
            f"only {header_hits}/{len(_HEADER_LABELS)} header labels found, "
            f"{numeric_row_count} numeric result rows (need ≥{_MIN_NUMERIC_ROWS})."
        )

    return ClassificationResult(
        is_lab_report=is_lab_report,
        confidence=round(confidence, 4),
        reason=reason,
    )
