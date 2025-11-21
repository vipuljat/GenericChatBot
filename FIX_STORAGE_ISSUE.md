# 🔧 FIX: Corrupted Qdrant Storage

## Problem
The Qdrant storage has become corrupted with invalid metadata. Embeddings cannot be stored or retrieved.

## Root Cause
- Qdrant's local storage file (`qdrant_storage/`) has validation errors
- Old metadata format is incompatible with current Pydantic version
- Storage file is locked when server is running

## Solution (3 Easy Steps)

### Step 1: Stop the Backend Server
```bash
# In the terminal running uvicorn, press Ctrl+C
```

### Step 2: Delete Corrupted Storage
```bash
# Run the cleanup script
python cleanup_storage.py

# OR manually delete the folder
Remove-Item -Recurse -Force qdrant_storage
```

### Step 3: Restart and Re-upload
```bash
# Start the server
uvicorn main:app --reload

# In another terminal, re-upload documents
python test_hr_chatbot.py
```

## Verification

After restarting, check if it works:
```bash
# This should show empty storage initially
python diagnose_storage.py

# Upload a document
python test_hr_chatbot.py

# Check again - should now show embeddings
python diagnose_storage.py
```

## Why This Happened

The Qdrant storage was created with an older version that stored metadata in a format incompatible with the current Pydantic validation rules. When the server tries to load this old metadata, it fails validation.

## Prevention

This shouldn't happen again because:
1. The code now handles corrupted storage better
2. Fresh installations won't have legacy metadata
3. The cleanup script is available if needed

## Quick Commands

```bash
# Stop server, clean, restart
Ctrl+C  # Stop server
python cleanup_storage.py
uvicorn main:app --reload

# Re-test
python test_hr_chatbot.py
```

## If Problems Persist

1. Make sure NO processes are using the `qdrant_storage/` folder
2. Check Task Manager for any Python/Uvicorn processes
3. Manually delete the folder:
   ```bash
   Remove-Item -Recurse -Force qdrant_storage
   ```
4. Restart your computer if files remain locked
