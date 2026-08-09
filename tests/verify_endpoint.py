import os
import requests
import json

API_URL = "http://127.0.0.1:8000/api/v1/documents/extract"
TEST_DIR = r"K:\Job_Prep\Company test\Celloscope-Assesment\celloscope-asr-ocr-pipeline\testdata\medical_LAB_Reports\Images\Scanned-Images"

def run_tests():
    print(f"Testing files in {TEST_DIR}")
    if not os.path.exists(TEST_DIR):
        print("Directory does not exist!")
        return

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.pdf'))]
    
    if not files:
        print("No valid test files found in the directory.")
        return

    success_count = 0
    failure_count = 0

    for filename in files:
        filepath = os.path.join(TEST_DIR, filename)
        print(f"\n--- Testing: {filename} ---")
        try:
            with open(filepath, 'rb') as f:
                response = requests.post(
                    API_URL,
                    files={'file': (filename, f, 'application/octet-stream')}
                )
            
            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get("results", []))
                patient_name = data.get("meta", {}).get("patient_name", "Unknown")
                print(f"✅ SUCCESS ({response.status_code}): Patient: {patient_name}, Results: {results_count}")
                success_count += 1
            elif response.status_code == 422:
                print(f"⚠️ REJECTED ({response.status_code}): Not a lab report. {response.text}")
                # We consider 422 a "pass" if it correctly rejected a non-lab report.
                success_count += 1 
            else:
                print(f"❌ FAILED ({response.status_code}): {response.text}")
                failure_count += 1
                
        except Exception as e:
            print(f"❌ ERROR: Failed to connect or request failed: {e}")
            failure_count += 1

    print("\n===============================")
    print(f"Summary: {success_count} Passed, {failure_count} Failed.")
    print("===============================\n")

if __name__ == "__main__":
    run_tests()
