"""Functional Qdrant vector store operations."""

from typing import List, Dict, Any, Optional
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from stateful_services.database import qdrant_manager
from utils.logging import log
import re


def sanitize_collection_name(name: str) -> str:
    """Sanitize chatbot name for Qdrant collection."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = f'_{sanitized}'
    return sanitized[:255] or 'chatbot_default'


def create_collection(
    collection_name: str,
    vector_size: int,
    force_recreate: bool = False
):
    """Create or recreate Qdrant collection."""
    client = qdrant_manager.get_client()
    collection_name = sanitize_collection_name(collection_name)
    
    existing = [c.name for c in client.get_collections().collections]
    
    if collection_name in existing:
        if force_recreate:
            client.delete_collection(collection_name)
            log.info(f"🗑️  Deleted existing collection: {collection_name}")
        else:
            log.info(f"Collection already exists: {collection_name}")
            return
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    log.info(f"✓ Created collection: {collection_name} (size={vector_size})")


def upsert_points(
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 100
):
    """Insert points into Qdrant in batches."""
    client = qdrant_manager.get_client()
    collection_name = sanitize_collection_name(collection_name)
    
    log.info(f"Upserting {len(points)} points to {collection_name}...")
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        log.info(f"✓ Batch {i//batch_size + 1}: {len(batch)} points")
    
    log.info(f"✓ Total {len(points)} points upserted")


def search_similar(
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    min_score: float = 0.5,
    filters: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors in Qdrant.
    Returns ONLY chunks - NO LLM CALLS.
    """
    client = qdrant_manager.get_client()
    collection_name = sanitize_collection_name(collection_name)
    
    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        log.warning(f"Collection not found: {collection_name}")
        return []
    
    # Build filters
    query_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        query_filter = Filter(must=conditions)
    
    # Search
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter
    )
    
    # Filter by score and format
    chunks = []
    for point in results.points:
        if point.score >= min_score:
            payload = point.payload or {}
            chunks.append({
                'text': payload.get('text', ''),
                'source_file': payload.get('source_file'),
                'document_type': payload.get('document_type'),
                'chunk_index': payload.get('chunk_index'),
                'score': float(point.score),
                'metadata': payload
            })
    
    log.info(f"Found {len(chunks)} chunks (score >= {min_score}) from {collection_name}")
    return chunks


def delete_collection(collection_name: str) -> bool:
    """Delete Qdrant collection."""
    client = qdrant_manager.get_client()
    collection_name = sanitize_collection_name(collection_name)
    
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
        log.info(f"🗑️  Deleted collection: {collection_name}")
        return True
    
    log.warning(f"Collection not found: {collection_name}")
    return False

