"""
Test script for document upload and embedding pipeline.
Uploads a sample document to the /chatbots endpoint and verifies embedding storage.
"""
import os
import requests
import time

API_URL = "http://localhost:8000/chatbot/chatbots"
SAMPLE_DOC_PATH = "sample_test.txt"
QDRANT_STORAGE_PATH = "qdrant_storage"

# Create a sample document
with open(SAMPLE_DOC_PATH, "w", encoding="utf-8") as f:
    f.write(("This is a test document for embedding pipeline.\n") * 100)

# Prepare form data
form_data = {
    "chatbot_name": (None, "TestBot"),
    "description": (None, "Test chatbot for embedding pipeline"),
    "instructions": (None, "Just a test."),
    "is_quiz_mode": (None, "false"),
    "is_active": (None, "true"),
    "recommended_model": (None, "gpt-3.5-turbo"),
}

# Attach file
with open(SAMPLE_DOC_PATH, "rb") as file_data:
    files = {
        "file": (os.path.basename(SAMPLE_DOC_PATH), file_data, "text/plain"),
    }

    print("Uploading sample document to /chatbot/chatbots ...")
    response = requests.post(API_URL, files=files, data=form_data)

print(f"Status code: {response.status_code}")

if response.status_code == 200:
    try:
        print("Response:", response.json())
    except Exception as e:
        print(f"Response: {response.text}")
else:
    print(f"Error: {response.text}")
    print("\n❌ Upload failed. Check the server logs above.")
    os.remove(SAMPLE_DOC_PATH)
    exit(1)

# Wait for embeddings
print("\nWaiting for embeddings to be generated...")
time.sleep(5)

# Check Qdrant storage
if os.path.exists(QDRANT_STORAGE_PATH):
    stored_files = os.listdir(QDRANT_STORAGE_PATH)
    if stored_files:
        print(f"✅ Embeddings stored in '{QDRANT_STORAGE_PATH}/':")
        for fname in stored_files:
            print(f"   - {fname}")
    else:
        print(f"⚠️ Qdrant storage folder exists but is empty.")
else:
    print(f"❌ Qdrant storage folder '{QDRANT_STORAGE_PATH}/' not found.")

# Cleanup
os.remove(SAMPLE_DOC_PATH)
print("🧹 Sample document removed.")
