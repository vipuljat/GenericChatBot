"""
Diagnostic script to check Qdrant storage and embeddings
"""
import sys
import os

# Check if qdrant_storage folder exists
storage_path = "qdrant_storage"
print("="*60)
print("QDRANT STORAGE DIAGNOSTICS")
print("="*60)

if os.path.exists(storage_path):
    print(f"✅ Storage folder exists: {storage_path}")
    
    # List all files in storage
    files = []
    for root, dirs, filenames in os.walk(storage_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            size = os.path.getsize(filepath)
            files.append((filepath, size))
    
    if files:
        print(f"\n📁 Found {len(files)} files:")
        for filepath, size in files:
            print(f"   - {filepath} ({size} bytes)")
    else:
        print("\n⚠️  Storage folder exists but is empty")
else:
    print(f"❌ Storage folder does not exist: {storage_path}")
    print("   Embeddings may not be getting stored!")

# Try to connect to Qdrant and check collections
print("\n" + "="*60)
print("QDRANT CONNECTION TEST")
print("="*60)

try:
    from services.qdrant_service import get_qdrant_service
    
    qdrant = get_qdrant_service()
    print("✅ Qdrant service initialized")
    
    # Get collections
    collections = qdrant.client.get_collections()
    print(f"\n📚 Collections: {len(collections.collections)}")
    
    for collection in collections.collections:
        print(f"\n   Collection: {collection.name}")
        
        # Get collection info
        info = qdrant.client.get_collection(collection.name)
        print(f"   - Vector count: {info.points_count}")
        print(f"   - Vector size: {info.config.params.vectors.size}")
        
        # Try to scroll through points
        if info.points_count > 0:
            points = qdrant.client.scroll(
                collection_name=collection.name,
                limit=5,
                with_payload=True
            )
            
            print(f"   - Sample points:")
            for point in points[0]:
                chatbot_id = point.payload.get('chatbot_id', 'Unknown')
                doc_name = point.payload.get('doc_name', 'Unknown')
                print(f"     • Point ID: {point.id}")
                print(f"       Chatbot: {chatbot_id}, Document: {doc_name}")
        else:
            print("   ⚠️  No points in collection!")
            
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

# Check database for documents
print("\n" + "="*60)
print("DATABASE CHECK")
print("="*60)

try:
    from core.database import SessionLocal
    from models.chatbot import Chatbot
    from models.documents import ChatbotDocument
    
    db = SessionLocal()
    
    # Get all chatbots
    chatbots = db.query(Chatbot).all()
    print(f"\n🤖 Chatbots: {len(chatbots)}")
    
    for bot in chatbots:
        print(f"\n   ID: {bot.id}, Name: {bot.chatbot_name}")
        
        # Get documents for this chatbot
        docs = db.query(ChatbotDocument).filter(
            ChatbotDocument.chatbot_id == bot.id
        ).all()
        
        print(f"   Documents: {len(docs)}")
        for doc in docs:
            print(f"     - {doc.doc_name}")
            print(f"       Path: {doc.file_path}")
            print(f"       Exists: {os.path.exists(doc.file_path)}")
    
    db.close()
    
except Exception as e:
    print(f"❌ Database error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
