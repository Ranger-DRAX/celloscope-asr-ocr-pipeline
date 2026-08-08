"""Normalization module for Endpoint 2 — Lab Report Extraction.

Pure functions only — no I/O, no adapter imports, no FastAPI dependencies.
Each function is independently unit-testable.

## Canonical forms (documented here and in README.md)

### Values
Inputs handled:
  - Plain decimal:           "12.5"      → parsed=True,  numeric=12.5
  - Qualified numeric:       "<0.5"      → parsed=True,  numeric=0.5  (qualifier in raw_line only)
  - Qualified numeric:       ">200"      → parsed=True,  numeric=200.0
  - Approx numeric:          "~1.2"      → parsed=True,  numeric=1.2
  - Thousands separator:     "12,500"    → parsed=True,  numeric=12500.0
  - Scientific-ish:          "1.2 x 10^3" / "1.2×10^3" → parsed=True, numeric=1200.0
  - Qualitative:             "Positive"  → parsed=True,  numeric=None, qualitative="Positive"
  - Ranges (reference):      "0.8 - 1.2" → parsed=False (ranges belong in reference_range)
  - Genuinely unparseable:   "??"        → parsed=False, original retained

### Units — canonical mapping table
  gm/dl, g/dL, GM/DL, Gm/dL  → "g/dL"
  mg/dl, mg/dL, MG/DL         → "mg/dL"
  mmol/l, mmol/L, MMOL/L      → "mmol/L"
  iu/l, IU/L, U/L, u/l        → "IU/L"
  10^3/ul, 10^3/µl, x10^3/µl, 10^3/uL, ×10³/µL → "10³/µL"
  10^6/ul, 10^6/µl             → "10⁶/µL"
  %                            → "%"
  (empty / absent)             → ""

### Dates — canonical form: ISO-8601 YYYY-MM-DD
  DD/MM/YYYY → YYYY-MM-DD      (DD/MM/YYYY assumed, NOT M/D/Y — documented below)
  DD-MM-YYYY → YYYY-MM-DD
  YYYY-MM-DD → YYYY-MM-DD      (already canonical)
  DD MMM YYYY, DD-MMM-YYYY     → YYYY-MM-DD
  Ambiguous (03/04/25):        → None (day-first locale not assumed; too ambiguous)

Ambiguity policy for dates: when a date string is 2-digit-year only (e.g. 03/04/25)
the function returns None rather than guessing a century or locale. The raw string
is preserved in the calling service for `raw_line`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Value normalization ───────────────────────────────────────────────────────

# Known qualitative sentinel strings (case-insensitive, exact match after strip)
_QUALITATIVE_SENTINELS = frozenset(
    {
        "positive", "negative", "nil", "trace", "reactive", "non-reactive",
        "nonreactive", "absent", "present", "normal", "abnormal", "borderline",
        "detected", "not detected", "equivocal", "inconclusive", "invalid",
        "see note", "+", "++", "+++", "++++", "1+", "2+", "3+", "4+",
    }
)

# Qualifier prefix characters (stripped from numeric values, kept in raw_line)
_QUALIFIER_PATTERN = re.compile(r"^([<>~≤≥±])\s*")

# Thousands-separator number: digits with commas, optional decimal
_THOUSANDS_SEP_PATTERN = re.compile(r"^(\d{1,3}(?:,\d{3})+(?:\.\d+)?)$")

# Scientific-ish notation: "1.2 x 10^3", "1.2×10^3", "1.2E3", "1.2e+3"
_SCI_PATTERN = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)\s*[×xXeE]\s*10\^?([+-]?\d+)$"
)
_STD_SCI_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)$")

# Range detection — "0.8 - 1.2", "0.8-1.2", "0.8 – 1.2"
_RANGE_PATTERN = re.compile(
    r"^\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?$"
)


@dataclass
class NormalizedValue:
    """Result of normalize_value().

    Attributes:
        parsed:       True if the input was successfully interpreted.
        numeric:      Extracted float, or None for qualitative results.
                      Qualifier characters are stripped (see raw_qualifier).
        qualitative:  Original string for non-numeric categorical results;
                      None for numeric results.
        raw_qualifier: The qualifier character found (<, >, ~, etc.), or "".
                      Downstream consumers can read this from raw_line instead.
        original:     The original input string, always preserved.
    """
    parsed: bool
    numeric: float | None
    qualitative: str | None
    raw_qualifier: str
    original: str


def normalize_value(raw: str) -> NormalizedValue:
    """Normalize a raw value string from a lab report result cell.

    Decision (from DECISIONS.md §8):
    - Qualitative results (Positive, Negative, Nil…) → parsed=True, numeric=None.
    - Qualified numerics (<0.5, >200) → strip qualifier, store threshold as numeric.
    - Ranges ("0.8 - 1.2") → parsed=False (these belong in reference_range).
    - Genuinely unparseable → parsed=False, numeric=None.
    - Never raises; always returns a NormalizedValue.
    """
    s = raw.strip()
    if not s:
        return NormalizedValue(parsed=False, numeric=None, qualitative=None,
                               raw_qualifier="", original=raw)

    # 1. Qualitative sentinel check (case-insensitive)
    if s.lower() in _QUALITATIVE_SENTINELS:
        return NormalizedValue(parsed=True, numeric=None, qualitative=s,
                               raw_qualifier="", original=raw)

    # 2. Range detection — must NOT be parsed as a value
    if _RANGE_PATTERN.match(s):
        return NormalizedValue(parsed=False, numeric=None, qualitative=None,
                               raw_qualifier="", original=raw)

    # 3. Strip qualifier prefix
    qualifier = ""
    m = _QUALIFIER_PATTERN.match(s)
    if m:
        qualifier = m.group(1)
        s = s[m.end():].strip()

    # 4. Thousands-separator number (e.g. "12,500")
    if _THOUSANDS_SEP_PATTERN.match(s):
        try:
            numeric = float(s.replace(",", ""))
            return NormalizedValue(parsed=True, numeric=numeric, qualitative=None,
                                   raw_qualifier=qualifier, original=raw)
        except ValueError:
            pass

    # 5. Scientific-ish notation: "1.2 x 10^3"
    m = _SCI_PATTERN.match(s)
    if m:
        try:
            mantissa = float(m.group(1))
            exponent = int(m.group(2))
            numeric = mantissa * (10 ** exponent)
            return NormalizedValue(parsed=True, numeric=numeric, qualitative=None,
                                   raw_qualifier=qualifier, original=raw)
        except (ValueError, OverflowError):
            pass

    # 6. Standard scientific notation: "1.2E3"
    m = _STD_SCI_PATTERN.match(s)
    if m:
        try:
            numeric = float(s)
            return NormalizedValue(parsed=True, numeric=numeric, qualitative=None,
                                   raw_qualifier=qualifier, original=raw)
        except ValueError:
            pass

    # 7. Plain float / int
    try:
        numeric = float(s)
        return NormalizedValue(parsed=True, numeric=numeric, qualitative=None,
                               raw_qualifier=qualifier, original=raw)
    except ValueError:
        pass

    # 8. Unrecognized — preserve verbatim
    return NormalizedValue(parsed=False, numeric=None, qualitative=None,
                           raw_qualifier=qualifier, original=raw)


# ── Unit normalization ────────────────────────────────────────────────────────

# Maps lowercased, whitespace-stripped raw unit strings to canonical form.
# Extend this table when new unit variants appear in real data.
_UNIT_CANONICAL: dict[str, str] = {
    # Hemoglobin / concentration — g/dL
    "g/dl": "g/dL",
    "gm/dl": "g/dL",
    "gm/dl.": "g/dL",
    "g/dl.": "g/dL",
    "gm/l": "g/dL",       # occasionally mistyped
    "g/l": "g/L",
    "gm": "g/dL",

    # mg/dL
    "mg/dl": "mg/dL",
    "mg/l": "mg/L",
    "mg/l.": "mg/L",

    # mmol/L
    "mmol/l": "mmol/L",
    "mmol/l.": "mmol/L",

    # IU/L, U/L
    "iu/l": "IU/L",
    "iu/ml": "IU/mL",
    "u/l": "IU/L",
    "u/ml": "IU/mL",
    "mu/l": "mIU/L",
    "miu/l": "mIU/L",

    # Cell counts
    "10^3/ul": "10³/µL",
    "10^3/µl": "10³/µL",
    "x10^3/ul": "10³/µL",
    "x10^3/µl": "10³/µL",
    "×10^3/µl": "10³/µL",
    "×10³/µl": "10³/µL",
    "10³/µl": "10³/µL",
    "10^3/ul.": "10³/µL",
    "10^6/ul": "10⁶/µL",
    "10^6/µl": "10⁶/µL",
    "cells/cumm": "cells/mm³",
    "cells/mm3": "cells/mm³",
    "/cumm": "cells/mm³",

    # Percentage
    "%": "%",
    "percent": "%",

    # fl (femtolitre — MCV)
    "fl": "fL",
    "fl.": "fL",

    # pg (picogram — MCH)
    "pg": "pg",
    "pg.": "pg",

    # mEq/L
    "meq/l": "mEq/L",

    # µmol/L
    "µmol/l": "µmol/L",
    "umol/l": "µmol/L",

    # ng/mL
    "ng/ml": "ng/mL",

    # µg/dL
    "µg/dl": "µg/dL",
    "ug/dl": "µg/dL",

    # Empty
    "": "",
    "-": "",
    "n/a": "",
    "na": "",
}


def normalize_unit(raw: str) -> str:
    """Normalize a unit string to a canonical form.

    Case-insensitive lookup in the canonical table. If not found, the
    original (stripped) string is returned as-is — no guessing.

    Args:
        raw: Raw unit string from OCR (e.g. "gm/dl", "10^3/µL").

    Returns:
        Canonical unit string (e.g. "g/dL", "10³/µL"), or the original
        stripped string if the unit is not in the lookup table.
    """
    key = raw.strip().lower()
    return _UNIT_CANONICAL.get(key, raw.strip())


# ── Date normalization ────────────────────────────────────────────────────────

# Month name → zero-padded month number
_MONTH_NAMES: dict[str, str] = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}

# DD/MM/YYYY or DD-MM-YYYY (4-digit year, unambiguous)
_DMY4_PATTERN = re.compile(
    r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$"
)
# YYYY-MM-DD or YYYY/MM/DD (already ISO)
_YMD4_PATTERN = re.compile(
    r"^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$"
)
# DD MMM YYYY or DD-MMM-YYYY ("15 Jan 2024", "15-Jan-2024")
_DMONTHY_PATTERN = re.compile(
    r"^(\d{1,2})[\s\-\.]([A-Za-z]{3,9})[\s\-\.](\d{4})$"
)
# MMM DD, YYYY ("Jan 15, 2024")
_MONDY_PATTERN = re.compile(
    r"^([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})$"
)
# Ambiguous 2-digit year: DD/MM/YY or MM/DD/YY
_AMBIGUOUS_2Y_PATTERN = re.compile(
    r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2}$"
)


def normalize_date(raw: str) -> Optional[str]:
    """Normalize a date string to ISO-8601 YYYY-MM-DD format.

    Assumption (documented): When the format is DD/MM/YYYY or DD-MM-YYYY, we
    interpret the first field as DAY, second as MONTH (day-first locale). This
    assumption is documented in DECISIONS.md and README.md.

    Ambiguous 2-digit-year dates (e.g. "03/04/25") return None — we do not
    guess a century or locale. The raw string is preserved by the caller.

    Args:
        raw: Raw date string from OCR.

    Returns:
        ISO-8601 date string "YYYY-MM-DD", or None if genuinely ambiguous
        or unparseable.
    """
    s = raw.strip()
    if not s:
        return None

    # Already ISO: YYYY-MM-DD / YYYY/MM/DD
    m = _YMD4_PATTERN.match(s)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        if _valid_date(y, mo, d):
            return f"{y}-{mo}-{d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = _DMY4_PATTERN.match(s)
    if m:
        d, mo, y = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        if _valid_date(y, mo, d):
            return f"{y}-{mo}-{d}"

    # DD MMM YYYY
    m = _DMONTHY_PATTERN.match(s)
    if m:
        d = m.group(1).zfill(2)
        month_str = m.group(2).lower()
        y = m.group(3)
        mo = _MONTH_NAMES.get(month_str)
        if mo and _valid_date(y, mo, d):
            return f"{y}-{mo}-{d}"

    # MMM DD, YYYY
    m = _MONDY_PATTERN.match(s)
    if m:
        month_str = m.group(1).lower()
        d = m.group(2).zfill(2)
        y = m.group(3)
        mo = _MONTH_NAMES.get(month_str)
        if mo and _valid_date(y, mo, d):
            return f"{y}-{mo}-{d}"

    # Ambiguous 2-digit year → do NOT guess
    if _AMBIGUOUS_2Y_PATTERN.match(s):
        return None

    return None


def _valid_date(y: str, m: str, d: str) -> bool:
    """Return True if (year, month, day) is a plausible calendar date."""
    try:
        yi, mi, di = int(y), int(m), int(d)
        if not (1900 <= yi <= 2100):
            return False
        if not (1 <= mi <= 12):
            return False
        if not (1 <= di <= 31):
            return False
        return True
    except ValueError:
        return False
