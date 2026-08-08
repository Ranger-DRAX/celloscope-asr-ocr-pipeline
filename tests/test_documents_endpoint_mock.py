import pytest
from fastapi.testclient import TestClient
import os

from app.main import app
from app.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def force_mock_provider(monkeypatch):
    """Force mock provider for all tests in this file."""
    monkeypatch.setattr(settings, "document_ocr_provider", "mock")

def _post_file(filename: str, content: bytes = b"fake-image", ctype: str = "image/jpeg"):
    return client.post(
        "/api/v1/documents/extract",
        files={"file": (filename, content, ctype)},
    )

def test_extract_clean_scanned_report():
    # Will hit testdata/fixtures/documents/scanned_clean_01.json
    response = _post_file("scanned_clean_01.jpg")
    
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Check Meta
    meta = data["meta"]
    assert meta["patient_name"] == "John Doe"
    assert meta["age"] == "45 Years"
    assert meta["sex"] == "Male"
    assert meta["reference_no"] == "PHD-2024-00123"
    assert meta["report_date"] == "2024-07-15"
    
    # Check Results
    results = data["results"]
    assert len(results) > 0
    
    # Check Haemoglobin (numeric)
    hb = next((r for r in results if r["test_name"] == "Haemoglobin"), None)
    assert hb is not None
    assert hb["value"] == 13.5
    assert hb["unit"] == "g/dL"  # Normalized from gm/dl
    assert hb["reference_range"] == "13.0 - 17.0"
    
    # Check Blood Glucose (qualified numeric)
    bg = next((r for r in results if r["test_name"] == "Blood Glucose (F)"), None)
    assert bg is not None
    assert bg["value"] == 0.5
    assert bg["unit"] == "mmol/L"
    assert "<" in bg["raw_line"]
    
    # Check Urine Albumin (qualitative)
    ua = next((r for r in results if r["test_name"] == "Urine Albumin"), None)
    assert ua is not None
    assert ua["value"] is None
    assert "Nil" in ua["raw_line"]

def test_extract_lowgrade_report():
    # Will hit testdata/fixtures/documents/lowgrade_01.json
    response = _post_file("lowgrade_01.jpg")
    
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Check Meta
    meta = data["meta"]
    assert meta["patient_name"] == "Fatema Begum"
    assert meta["sex"] == "Female" # Normalization from F
    
    # Check Results
    results = data["results"]
    assert len(results) > 0
    
    # Check RBC Count (scientific)
    rbc = next((r for r in results if r["test_name"] == "RBC Count"), None)
    assert rbc is not None
    assert rbc["value"] == 1200.0  # 1.2 x 10^3
    assert rbc["unit"] == "10⁶/µL" # Normalized from 10^6/ul
    
def test_extract_non_lab_report():
    # Will hit testdata/fixtures/documents/non_lab_report.json
    response = _post_file("non_lab_report.jpg")
    
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error"] == "not_a_lab_report"
    assert "Classified as lab report" not in data["detail"]["detail"]
