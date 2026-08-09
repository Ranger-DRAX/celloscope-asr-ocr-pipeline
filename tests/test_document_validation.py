import pytest
from app.services.document_extraction_service import (
    validate_document,
    DocumentUnsupportedFormatError,
    DocumentTooLargeError
)

def test_validate_document_valid():
    validate_document("report.jpg", 1024 * 1024, max_mb=25)
    validate_document("scan.PDF", 1024 * 1024, max_mb=25)

def test_validate_document_unsupported_format():
    with pytest.raises(DocumentUnsupportedFormatError):
        validate_document("report.txt", 1024, max_mb=25)
        
    with pytest.raises(DocumentUnsupportedFormatError):
        validate_document("report.docx", 1024, max_mb=25)

def test_validate_document_too_large():
    with pytest.raises(DocumentTooLargeError):
        validate_document("huge.jpg", 30 * 1024 * 1024, max_mb=25)
