FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any are needed
# (e.g. ffmpeg might be needed for faster-whisper/audio handling)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set default environment variables
ENV LOG_LEVEL=INFO
ENV DOCUMENT_EXTRACTION_PROVIDER=mock
ENV TRANSCRIBE_PROVIDER=faster_whisper

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
