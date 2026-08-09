import os
import sys
import time
import traceback
os.environ["HF_HUB_OFFLINE"] = "1"
# faster-whisper/ctranslate2 loads cublas64_12.dll, cudnn64_9.dll, nvrtc64_120_0.dll
# at import time, but they ship inside pip nvidia-* wheels and are NOT on PATH.
# Inject their bin dirs into PATH before importing faster_whisper.
def _add_nvidia_dlls():
    nvidia_root = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.isdir(nvidia_root):
        return
    for sub in ("cublas", "cudnn", "cuda_nvrtc"):
        bin_dir = os.path.join(nvidia_root, sub, "bin")
        if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

_add_nvidia_dlls()

from faster_whisper import WhisperModel

# 1. Point this exactly to the folder name you just created
model_path = "pretrained_models/faster-whisper-bangla-small-int8"
# model_path="models/whisper-small-bn-ct2"

print("Loading local Bangla model into GPU...")
# 2. compute_type="int8" is the magic that makes it fit your 4GB VRAM
model = WhisperModel(
    model_path, 
    device="cuda", 
    compute_type="int8" 
)
print("Model loaded successfully! No VRAM crash!")

# Replace this with the name of a real Bengali audio file you have in your folder
audio_file = "testdata/audio_BN/sample_6.mp3"
# audio_file = "testdata/audio_EN/arctic_a0004_1592748489.flac"



try:
    print(f"Transcribing {audio_file}...")
    start_time = time.time()
    
    # language="bn" forces it to transcribe in Bengali
    segments, info = model.transcribe(audio_file, language="bn")
    
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
    print(f"Transcription took {time.time() - start_time:.2f} seconds")

except Exception as e:
    print(f"Failed to transcribe {type(e).__name__}: {e}")
    traceback.print_exc()
