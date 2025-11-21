"""
Direct test of Qdrant search functionality
"""
from services.qdrant_service import get_qdrant_service
from services.embedding_service import get_embedding_service

print("="*60)
print("DIRECT QDRANT SEARCH TEST")
print("="*60)

# Get services
qdrant = get_qdrant_service()
embedding_service = get_embedding_service()

print(f"\n✅ Qdrant service initialized")
print(f"   Client type: {type(qdrant.client)}")
print(f"   Has search: {hasattr(qdrant.client, 'search')}")

# Check collections
collections = qdrant.client.get_collections()
print(f"\n📚 Collections: {[c.name for c in collections.collections]}")

# Get collection info
collection_name = "chatbot_documents"
info = qdrant.client.get_collection(collection_name)
print(f"\n📊 Collection '{collection_name}':")
print(f"   Points: {info.points_count}")
print(f"   Vector size: {info.config.params.vectors.size}")

if info.points_count > 0:
    # Sample some points
    points = qdrant.client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True
    )
    
    print(f"\n📄 Sample points:")
    for point in points[0]:
        print(f"   - Chatbot {point.payload.get('chatbot_id')}: {point.payload.get('doc_name')}")
        print(f"     Text preview: {point.payload.get('text', '')[:100]}...")
    
    # Test search
    print(f"\n🔍 Testing search...")
    query = "leave policy"
    print(f"   Query: '{query}'")
    
    query_embedding = embedding_service.generate_embedding(query)
    print(f"   Embedding generated: {len(query_embedding)} dimensions")
    
    # Get a chatbot_id to search
    chatbot_id = points[0][0].payload.get('chatbot_id')
    print(f"   Searching in chatbot: {chatbot_id}")
    
    results = qdrant.search_similar(
        query_embedding=query_embedding,
        chatbot_id=chatbot_id,
        limit=3
    )
    
    print(f"\n✅ Search results: {len(results)} found")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"      Score: {result['score']:.4f}")
        print(f"      Document: {result['metadata'].get('doc_name')}")
        print(f"      Text: {result['metadata'].get('text', '')[:150]}...")
else:
    print("\n⚠️  No points in collection yet!")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
