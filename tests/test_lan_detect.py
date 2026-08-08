import os
import sys
import time

# 1. Suppress Hugging Face hub warnings & symlink notices
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"  # Set to "1" once tiny model finishes downloading once

# 2. Add CUDA DLL paths for GPU scripts (kept for safety)
venv_base = sys.prefix
nvidia_base_path = os.path.join(venv_base, 'Lib', 'site-packages', 'nvidia')
cublas_bin = os.path.join(nvidia_base_path, 'cublas', 'bin')
cudnn_bin = os.path.join(nvidia_base_path, 'cudnn', 'bin')
os.environ['PATH'] = f"{cublas_bin};{cudnn_bin};" + os.environ.get('PATH', '')

from faster_whisper import WhisperModel

print("Initializing lightweight language detector on CPU...")
# Using device="cpu" avoids all GPU/CUDA DLL issues for detection
detector = WhisperModel("tiny", device="cuda", compute_type="int8")

def detect_audio_language(audio_path):
    start_time = time.time()
    # Transcribe with beam_size=1 just to fetch language metadata instantly
    _, info = detector.transcribe(audio_path, beam_size=1)
    detection_time = time.time() - start_time
    
    return info.language, info.language_probability, detection_time

# Test file
audio_file = "testdata/audio_BN/sample_6.mp3"

print(f"Detecting language for {audio_file}...")
lang, prob, elapsed = detect_audio_language(audio_file)

print(f"\nResults:")
print(f" Detected Language : {lang.upper()}")
print(f" Confidence          : {prob * 100:.2f}%")
print(f" Detection Speed     : {elapsed:.3f} seconds")

# Pipeline Routing Logic Example
if lang == "bn":
    print("\n Routing -> faster-whisper-bangla-small-int8 (Local GPU)")
elif lang == "en":
    print("\n Routing -> Standard English Whisper Model")
else:
    print(f"\n Routing -> Fallback Model for language code: {lang}")