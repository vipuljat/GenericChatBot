"""
Cleanup script to fix corrupted Qdrant storage
IMPORTANT: Stop the backend server before running this!
"""
import os
import shutil

storage_path = "qdrant_storage"

print("="*60)
print("QDRANT STORAGE CLEANUP")
print("="*60)
print("\n⚠️  IMPORTANT: Make sure the backend server is STOPPED!")
print("   Press Ctrl+C now if the server is still running.")
print("   Otherwise, press Enter to continue...")
input()

if os.path.exists(storage_path):
    try:
        print(f"\n🗑️  Removing corrupted storage: {storage_path}")
        shutil.rmtree(storage_path)
        print("✅ Storage removed successfully")
    except PermissionError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 The storage is locked because the server is still running.")
        print("   Steps to fix:")
        print("   1. Stop the backend server (Ctrl+C in the terminal)")
        print("   2. Run this script again: python cleanup_storage.py")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)
else:
    print(f"✅ Storage folder doesn't exist (nothing to clean)")

print("\n" + "="*60)
print("CLEANUP COMPLETE")
print("="*60)
print("\n📝 Next steps:")
print("   1. Start the backend server: uvicorn main:app --reload")
print("   2. Re-upload documents to create fresh embeddings")
print("   3. Run test: python test_hr_chatbot.py")
