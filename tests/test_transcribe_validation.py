"""Unit tests for audio validation — no adapter needed."""

import pytest

from app.services.transcribe_service import (
    validate_audio,
    UnsupportedFormatError,
    FileTooLargeError,
)


class TestValidateAudio:
    """Covers: valid extension + size, bad extension, oversized file."""

    def test_valid_wav_under_limit(self):
        """Valid .wav file under the MB limit passes without exception."""
        validate_audio("recording.wav", 1_000_000, max_mb=25)

    def test_valid_mp3_under_limit(self):
        """Valid .mp3 file passes."""
        validate_audio("recording.mp3", 5_000_000, max_mb=25)

    def test_valid_flac_under_limit(self):
        """Valid .flac file passes."""
        validate_audio("clip.flac", 10_000_000, max_mb=25)

    def test_valid_ogg_under_limit(self):
        """Valid .ogg file passes."""
        validate_audio("clip.ogg", 100_000, max_mb=25)

    def test_valid_m4a_under_limit(self):
        """Valid .m4a file passes."""
        validate_audio("clip.m4a", 2_000_000, max_mb=25)

    def test_unsupported_exe_raises(self):
        """An .exe file is rejected with UnsupportedFormatError."""
        with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
            validate_audio("malware.exe", 100, max_mb=25)

    def test_unsupported_txt_raises(self):
        """A .txt file is rejected."""
        with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
            validate_audio("notes.txt", 100, max_mb=25)

    def test_unsupported_pdf_raises(self):
        """A .pdf file is rejected."""
        with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
            validate_audio("document.pdf", 100, max_mb=25)

    def test_file_over_limit_raises(self):
        """A file exceeding max_mb raises FileTooLargeError."""
        over_limit = 26 * 1024 * 1024  # 26 MB
        with pytest.raises(FileTooLargeError, match="File exceeds 25MB limit"):
            validate_audio("large.wav", over_limit, max_mb=25)

    def test_file_exactly_at_limit_passes(self):
        """A file exactly at max_mb should pass (not strictly greater)."""
        exactly = 25 * 1024 * 1024
        validate_audio("exact.mp3", exactly, max_mb=25)
