"""Pydantic schemas for the document extraction endpoint."""

from pydantic import BaseModel, Field

class ReportMetaSchema(BaseModel):
    patient_name: str | None = Field(default=None)
    age: str | None = Field(default=None)
    sex: str | None = Field(default=None)
    report_date: str | None = Field(default=None)
    lab_name: str | None = Field(default=None)
    reference_no: str | None = Field(default=None)

class ResultRowSchema(BaseModel):
    test_name: str
    value: float
    unit: str
    reference_range: str
    flag: str
    raw_line: str

class DocumentExtractionResponse(BaseModel):
    meta: ReportMetaSchema
    results: list[ResultRowSchema]
