import os
from groq import Groq

# FIX: Pass the string directly to api_key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

audio_path = "testdata/audio_EN/arctic_a0004_1592748489.flac"

try:
    with open(audio_path, "rb") as file:
        print("Sending audio to Groq API...")
        
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        
        # Extract detected language
        detected_lang = transcription.language 
        print(f"Detected Language via API: {detected_lang}")

except Exception as e:
    print(f"API Error: {e}")