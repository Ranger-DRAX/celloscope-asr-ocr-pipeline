from fastapi import FastAPI

from app.api.routes_transcribe import router as transcribe_router

app = FastAPI(title="Celloscope AI Service")
app.include_router(transcribe_router)
