"""API routes for Endpoint 2 — Lab Report Extraction."""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.api.schemas_documents import DocumentExtractionResponse, ReportMetaSchema, ResultRowSchema
from app.services.document_extraction_service import (
    validate_document,
    run_extraction,
    DocumentUnsupportedFormatError,
    DocumentTooLargeError,
    DocumentCorruptError,
    NotALabReportError,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

class ErrorDetail(BaseModel):
    error: str
    detail: str

class ErrorResponse(BaseModel):
    detail: ErrorDetail

@router.post(
    "/api/v1/documents/extract",
    response_model=DocumentExtractionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def extract_document(
    file: UploadFile = File(...),
):
    """Extract metadata and test results from a lab report image/PDF."""
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_upload", "detail": f"Failed to read file stream: {e}"},
        )

    try:
        validate_document(
            filename=file.filename or "unknown",
            size_bytes=len(file_bytes),
            max_mb=settings.max_upload_mb,
        )
    except DocumentUnsupportedFormatError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_format", "detail": str(e)},
        )
    except DocumentTooLargeError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "file_too_large", "detail": str(e)},
        )

    try:
        result = run_extraction(
            image_bytes=file_bytes,
            filename=file.filename or "unknown",
        )
    except DocumentCorruptError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "corrupt_document", "detail": str(e)},
        )
    except NotALabReportError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "not_a_lab_report", "detail": str(e)},
        )
    except Exception as e:
        logger.error(f"Document extraction pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": "An unexpected error occurred during extraction."},
        )

    return DocumentExtractionResponse(
        meta=ReportMetaSchema(
            patient_name=result.meta.patient_name,
            age=result.meta.age,
            sex=result.meta.sex,
            report_date=result.meta.report_date,
            lab_name=result.meta.lab_name,
            reference_no=result.meta.reference_no,
        ),
        results=[
            ResultRowSchema(
                test_name=r.test_name,
                value=r.value,
                unit=r.unit,
                reference_range=r.reference_range,
                flag=r.flag,
                raw_line=r.raw_line,
            )
            for r in result.results
        ]
    )
