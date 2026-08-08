"""Table parser — extracts RawResultRows from a RawOCRResult.

Strategy (priority order):
1. If table_cells are populated (PP-Structure found a table), use them —
   they have explicit row/col indices and are the most reliable source.
2. If no table_cells are available, fall back to line-by-line heuristics:
   scan lines past the column-header row and try to split each line into
   the expected columns (test name | value | unit | range | flag).

raw_line is SACRED: every RawResultRow carries the exact, verbatim OCR text
for the source row. It is never cleaned up, never re-flowed, never truncated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.adapters.ocr.base import RawOCRResult, OCRTableCell


@dataclass
class RawResultRow:
    """One result row as extracted directly from OCR, before normalization.

    raw_line is the verbatim concatenation of all cells in the row (or the
    full text line in line-heuristic mode). It must never be modified.
    """
    test_name: str
    value_raw: str          # verbatim value text from the cell
    unit_raw: str
    reference_range: str
    flag: str
    raw_line: str           # SACRED — exact OCR text, preserved verbatim


# ── Column-header detection ───────────────────────────────────────────────────

# Keywords that identify the header row of the results table
_HEADER_KEYWORDS = re.compile(
    r"\b(?:test\s*name|investigation|parameter|analyte|result|value|ref(?:erence)?\s*range|unit|flag)\b",
    re.IGNORECASE,
)


def parse_table(ocr: RawOCRResult) -> list[RawResultRow]:
    """Extract result rows from OCR output.

    Tries structured table_cells first; falls back to line heuristics.
    Always returns raw, un-normalized rows.
    """
    if ocr.table_cells:
        return _parse_from_table_cells(ocr.table_cells)
    return _parse_from_lines(ocr)


# ── Strategy 1: Structured table cells ───────────────────────────────────────

def _parse_from_table_cells(cells: list[OCRTableCell]) -> list[RawResultRow]:
    """Reconstruct rows from PP-Structure table cells."""
    # Group cells by row_index
    rows: dict[int, dict[int, str]] = {}
    for cell in cells:
        rows.setdefault(cell.row_index, {})[cell.col_index] = cell.text.strip()

    if not rows:
        return []

    # Detect header row (row 0 or the first row containing table header keywords)
    sorted_row_indices = sorted(rows.keys())
    header_row_idx = sorted_row_indices[0]
    for idx in sorted_row_indices:
        row_text = " ".join(rows[idx].values())
        if _HEADER_KEYWORDS.search(row_text):
            header_row_idx = idx
            break

    # Identify column mapping from the header row
    col_map = _infer_column_map(rows.get(header_row_idx, {}))

    results: list[RawResultRow] = []
    for row_idx in sorted_row_indices:
        if row_idx == header_row_idx:
            continue
        row = rows[row_idx]
        test_name = row.get(col_map.get("test", -1), "").strip()
        value_raw = row.get(col_map.get("value", -1), "").strip()
        unit_raw = row.get(col_map.get("unit", -1), "").strip()
        ref_range = row.get(col_map.get("range", -1), "").strip()
        flag = row.get(col_map.get("flag", -1), "").strip()

        # Skip fully empty rows
        if not any([test_name, value_raw, unit_raw, ref_range]):
            continue

        # raw_line: verbatim concatenation of all cells in the row, tab-separated
        raw_line = "\t".join(row.get(c, "") for c in sorted(row.keys()))

        results.append(RawResultRow(
            test_name=test_name,
            value_raw=value_raw,
            unit_raw=unit_raw,
            reference_range=ref_range,
            flag=flag,
            raw_line=raw_line,
        ))

    return results


_TEST_NAME_COLS = re.compile(r"test|investigation|parameter|analyte", re.IGNORECASE)
_VALUE_COLS = re.compile(r"result|value", re.IGNORECASE)
_UNIT_COLS = re.compile(r"unit", re.IGNORECASE)
_RANGE_COLS = re.compile(r"ref(?:erence)?\s*range|normal|range", re.IGNORECASE)
_FLAG_COLS = re.compile(r"flag|remark|status|interpretation", re.IGNORECASE)


def _infer_column_map(header_row: dict[int, str]) -> dict[str, int]:
    """Map logical column names to physical column indices from header text."""
    col_map: dict[str, int] = {}
    for col_idx, text in header_row.items():
        if _TEST_NAME_COLS.search(text) and "test" not in col_map:
            col_map["test"] = col_idx
        elif _VALUE_COLS.search(text) and "value" not in col_map:
            col_map["value"] = col_idx
        elif _UNIT_COLS.search(text) and "unit" not in col_map:
            col_map["unit"] = col_idx
        elif _RANGE_COLS.search(text) and "range" not in col_map:
            col_map["range"] = col_idx
        elif _FLAG_COLS.search(text) and "flag" not in col_map:
            col_map["flag"] = col_idx

    # Default fallbacks if header detection failed
    sorted_cols = sorted(header_row.keys())
    defaults = ["test", "value", "unit", "range", "flag"]
    for i, key in enumerate(defaults):
        if key not in col_map and i < len(sorted_cols):
            col_map[key] = sorted_cols[i]

    return col_map


# ── Strategy 2: Line heuristics ───────────────────────────────────────────────

# A line that looks like a result row: starts with alphabetic test name,
# followed by a numeric-looking value somewhere in the line.
_RESULT_LINE_PAT = re.compile(
    r"^([A-Za-z][A-Za-z0-9\s\(\)/\-\.]+?)\s{2,}"  # test name (2+ spaces as separator)
    r"([<>~]?\d[\d,\.]*(?:\s*[xX×]\s*10\^?\d+)?)"  # value
    r"(?:\s+([^\s]+))?"                              # unit (optional)
    r"(?:\s+(\d[\d\.\-\s]+\d))?"                    # reference range (optional)
    r"(?:\s+([A-Za-z*]+))?",                         # flag (optional)
)


def _parse_from_lines(ocr: RawOCRResult) -> list[RawResultRow]:
    """Fallback: extract result rows from text lines using heuristics."""
    lines_sorted = sorted(ocr.lines, key=lambda l: l.order)

    # Find the header row
    header_idx = None
    for i, line in enumerate(lines_sorted):
        if _HEADER_KEYWORDS.search(line.text):
            header_idx = i
            break

    if header_idx is None:
        return []  # Can't identify where results start

    results: list[RawResultRow] = []
    for line in lines_sorted[header_idx + 1:]:
        text = line.text.strip()
        if not text:
            continue
        m = _RESULT_LINE_PAT.match(text)
        if m:
            results.append(RawResultRow(
                test_name=(m.group(1) or "").strip(),
                value_raw=(m.group(2) or "").strip(),
                unit_raw=(m.group(3) or "").strip(),
                reference_range=(m.group(4) or "").strip(),
                flag=(m.group(5) or "").strip(),
                raw_line=text,  # SACRED: the verbatim line
            ))

    return results
