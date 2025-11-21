"""
Force cleanup of qdrant_storage with retries
"""
import os
import shutil
import time
import sys

storage_path = "qdrant_storage"

print("="*60)
print("FORCE CLEANUP QDRANT STORAGE")
print("="*60)

if not os.path.exists(storage_path):
    print(f"✅ Storage folder doesn't exist, nothing to clean")
    sys.exit(0)

print(f"\n⚠️  Attempting to remove: {storage_path}")
print("   This may take multiple attempts if files are locked...\n")

max_attempts = 5
for attempt in range(1, max_attempts + 1):
    try:
        print(f"Attempt {attempt}/{max_attempts}...")
        
        # First, try to rename (sometimes works when delete doesn't)
        temp_path = f"{storage_path}_old_{int(time.time())}"
        if os.path.exists(storage_path):
            os.rename(storage_path, temp_path)
            print(f"   Renamed to: {temp_path}")
            
        # Now try to delete
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            print(f"   ✅ Deleted successfully!")
            break
            
    except PermissionError as e:
        print(f"   ❌ Locked: {e}")
        if attempt < max_attempts:
            print(f"   Waiting 2 seconds before retry...")
            time.sleep(2)
        else:
            print("\n" + "="*60)
            print("❌ CLEANUP FAILED")
            print("="*60)
            print("\n⚠️  The storage is still locked. Please:")
            print("   1. Stop the backend server (Ctrl+C in uvicorn terminal)")
            print("   2. Close any other Python processes")
            print("   3. Run this script again")
            print(f"\n   OR manually delete: {storage_path}")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        if attempt == max_attempts:
            sys.exit(1)
        time.sleep(2)

print("\n" + "="*60)
print("✅ CLEANUP SUCCESSFUL")
print("="*60)
print("\nNow you can:")
print("   1. Start the server: uvicorn main:app --reload")
print("   2. Run test: python test_hr_chatbot.py")
