import asyncio
import os

# Force Mistral provider and make sure API key is read from .env if present
os.environ["DOCUMENT_EXTRACTION_PROVIDER"] = "mistral_ocr"
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from app.main import app

def test_endpoint():
    print("Initializing test client...")
    client = TestClient(app)
    
    print("Testing /api/v1/documents/extract with Mistral provider...")
    
    test_files = [
        "testdata/medical_LAB_Reports/PDF/9109 Abn Sample Report 20180503 CBC with Differential Blood.pdf",
        "testdata/medical_LAB_Reports/PDF/sterling-accuris-pathology-sample-report-unlocked.pdf",
        "testdata/medical_LAB_Reports/Images/Scanned-Images/1.webp"
    ]
    
    import json
    for idx, file_path in enumerate(test_files):
        print(f"\n--- Testing {file_path} ---")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(file_path)
        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/webp"
        files = {'file': (filename, file_bytes, mime_type)}
        
        response = client.post("/api/v1/documents/extract", files=files)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            output_path = f"testdata/output_sample_{idx}.json"
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(response.json(), out_f, indent=2, ensure_ascii=False)
            print(f"Extraction successful! Output saved to {output_path}")
        else:
            print("Error:")
            print(response.text)

if __name__ == "__main__":
    test_endpoint()
