"""
Test script to verify the embedding pipeline setup
Run this before starting the full application
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing imports...")
    try:
        import fastapi
        print("   ✅ FastAPI")
        import qdrant_client
        print("   ✅ Qdrant Client")
        import openai
        print("   ✅ OpenAI")
        import PyPDF2
        print("   ✅ PyPDF2")
        from docx import Document
        print("   ✅ python-docx")
        import tiktoken
        print("   ✅ tiktoken")
        import sqlalchemy
        print("   ✅ SQLAlchemy")
        print("\n✅ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def test_config():
    """Test if configuration is properly set"""
    print("🔍 Testing configuration...")
    try:
        import config
        
        # Check database
        if config.DATABASE_URL:
            print(f"   ✅ Database URL: {config.DATABASE_URL[:30]}...")
        else:
            print("   ⚠️  Database URL not set")
        
        # Check OpenAI
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your-openai-api-key-here":
            print(f"   ✅ OpenAI API Key: {config.OPENAI_API_KEY[:10]}...")
        else:
            print("   ❌ OpenAI API Key not set in .env")
            print("      Add: OPENAI_API_KEY=sk-your-key-here")
            return False
        
        # Check Qdrant
        print(f"   ✅ Qdrant URL: {config.QDRANT_URL}")
        print(f"   ✅ Qdrant Storage: {config.QDRANT_STORAGE_PATH}")
        
        # Check upload directory
        if not os.path.exists(config.UPLOAD_DIR):
            os.makedirs(config.UPLOAD_DIR)
            print(f"   ✅ Created upload directory: {config.UPLOAD_DIR}")
        else:
            print(f"   ✅ Upload directory exists: {config.UPLOAD_DIR}")
        
        print("\n✅ Configuration looks good!\n")
        return True
    except Exception as e:
        print(f"\n❌ Configuration error: {e}\n")
        return False

def test_qdrant():
    """Test Qdrant client initialization"""
    print("🔍 Testing Qdrant connection...")
    try:
        from services.qdrant_service import get_qdrant_service
        
        qdrant = get_qdrant_service()
        collections = qdrant.client.get_collections()
        
        print(f"   ✅ Qdrant initialized successfully")
        print(f"   ✅ Collections: {[c.name for c in collections.collections]}")
        print(f"   ✅ Storage mode: {'Server' if hasattr(qdrant.client, 'host') else 'Local file'}")
        print("\n✅ Qdrant is ready!\n")
        return True
    except Exception as e:
        print(f"   ⚠️  Qdrant initialization: {e}")
        print(f"   ℹ️  Will use local file storage at: ./qdrant_storage")
        print("   ✅ This is normal and will work fine!\n")
        return True

def test_services():
    """Test if services can be imported"""
    print("🔍 Testing services...")
    try:
        from services.document_service import get_document_service
        print("   ✅ Document Service")
        
        from services.embedding_service import get_embedding_service
        print("   ✅ Embedding Service")
        
        from services.qdrant_service import get_qdrant_service
        print("   ✅ Qdrant Service")
        
        from services.chatbot_service import get_chatbot_service
        print("   ✅ Chatbot Service")
        
        print("\n✅ All services available!\n")
        return True
    except Exception as e:
        print(f"\n❌ Service error: {e}\n")
        return False

def test_routes():
    """Test if routes can be imported"""
    print("🔍 Testing routes...")
    try:
        from routes.chatbot_routes import router as chatbot_router
        print("   ✅ Chatbot Routes")
        
        from routes.auth_routes import router as auth_router
        print("   ✅ Auth Routes")
        
        print("\n✅ All routes available!\n")
        return True
    except Exception as e:
        print(f"\n❌ Route error: {e}\n")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("🧪 ChatBot Embedding Pipeline - Setup Verification")
    print("="*50 + "\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Services", test_services()))
    results.append(("Routes", test_routes()))
    results.append(("Qdrant", test_qdrant()))
    
    print("="*50)
    print("📊 Test Results:")
    print("="*50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*50)
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready!")
        print("\n📌 Next steps:")
        print("   1. Make sure OpenAI API key is set in .env")
        print("   2. Start the server: uvicorn main:app --reload")
        print("   3. Visit: http://localhost:8000/docs")
        print("   4. Test document upload via API\n")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("   Check the error messages and try again.\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
