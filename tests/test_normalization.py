import pytest
from app.services.normalization import normalize_value, normalize_unit, normalize_date

def test_normalize_value():
    # Plain decimal
    v = normalize_value("12.5")
    assert v.parsed is True
    assert v.numeric == 12.5
    assert v.qualitative is None
    
    # Qualified numeric
    v = normalize_value("<0.5")
    assert v.parsed is True
    assert v.numeric == 0.5
    assert v.raw_qualifier == "<"

    v = normalize_value(">200")
    assert v.parsed is True
    assert v.numeric == 200.0
    assert v.raw_qualifier == ">"

    # Thousands separator
    v = normalize_value("12,500")
    assert v.parsed is True
    assert v.numeric == 12500.0

    # Scientific-ish
    v = normalize_value("1.2 x 10^3")
    assert v.parsed is True
    assert v.numeric == 1200.0

    v = normalize_value("1.2×10^3")
    assert v.parsed is True
    assert v.numeric == 1200.0

    # Qualitative
    v = normalize_value("Positive")
    assert v.parsed is True
    assert v.numeric is None
    assert v.qualitative.lower() == "positive"

    # Range
    v = normalize_value("0.8 - 1.2")
    assert v.parsed is False

    # Unparseable
    v = normalize_value("??")
    assert v.parsed is False
    assert v.numeric is None


def test_normalize_unit():
    assert normalize_unit("gm/dl") == "g/dL"
    assert normalize_unit("g/dL") == "g/dL"
    assert normalize_unit("GM/DL") == "g/dL"
    assert normalize_unit("10^3/ul") == "10³/µL"
    assert normalize_unit("x10^3/µl") == "10³/µL"
    assert normalize_unit("%") == "%"
    assert normalize_unit("unknown_unit") == "unknown_unit"


def test_normalize_date():
    assert normalize_date("15/07/2024") == "2024-07-15"
    assert normalize_date("15-07-2024") == "2024-07-15"
    assert normalize_date("2024-07-15") == "2024-07-15"
    assert normalize_date("15 Jan 2024") == "2024-01-15"
    assert normalize_date("15-Jan-2024") == "2024-01-15"
    assert normalize_date("Jan 15, 2024") == "2024-01-15"
    
    # Ambiguous 2-digit year
    assert normalize_date("03/04/25") is None
    
    # Invalid date
    assert normalize_date("32/13/2024") is None
