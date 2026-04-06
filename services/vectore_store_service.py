"""Functional Qdrant vector store operations."""

from typing import List, Dict, Any, Optional, Set
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    FilterSelector
)
from stateful_services.database import qdrant_manager
from utils.logging import log
from utils.retrieval_metadata import extract_content_tokens
import re


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name for Qdrant.
    Qdrant collection names must contain only letters, digits, and underscores.
    
    Examples:
        "HR Conversation" -> "HR_Conversation"
        "My-Cool Bot!" -> "My_Cool_Bot_"
        "123start" -> "_123start"
    """
    # Replace spaces and hyphens with underscores
    sanitized = name.replace(' ', '_').replace('-', '_')
    # Remove any non-alphanumeric characters except underscores
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = f'_{sanitized}'
    # Limit length and provide default
    return sanitized[:255] or 'chatbot_default'


def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)
        collections = client.get_collections().collections
        exists = any(c.name == sanitized_name for c in collections)
        log.debug(f"Collection '{sanitized_name}' exists: {exists}")
        return exists
    except Exception as e:
        log.error(f"Error checking collection existence: {e}")
        return False


def create_collection(
    collection_name: str,
    vector_size: int,
    force_recreate: bool = False
):
    """Create or recreate Qdrant collection."""
    client = qdrant_manager.get_client()
    sanitized_name = sanitize_collection_name(collection_name)
    
    existing = [c.name for c in client.get_collections().collections]
    
    if sanitized_name in existing:
        if force_recreate:
            client.delete_collection(sanitized_name)
            log.info(f"🗑️  Deleted existing collection: {sanitized_name}")
        else:
            log.info(f"Collection already exists: {sanitized_name}")
            return
    
    client.create_collection(
        collection_name=sanitized_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    log.info(f"✓ Created collection: {sanitized_name} (size={vector_size})")


def get_collection_info(collection_name: str) -> dict:
    """
    Get collection information including point count and vector size.
    """
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)
        collection_info = client.get_collection(sanitized_name)
        
        # Get vector size from config
        vector_size = None
        if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
            if hasattr(collection_info.config.params, 'vectors'):
                vectors_config = collection_info.config.params.vectors
                if hasattr(vectors_config, 'size'):
                    vector_size = vectors_config.size
        
        info = {
            'points_count': collection_info.points_count,
            'vectors_count': collection_info.indexed_vectors_count,
            'vector_size': vector_size
        }
        log.info(f"Collection '{sanitized_name}' info: {info}")
        return info
    except Exception as e:
        log.error(f"Error getting collection info: {e}")
        return {'points_count': 0, 'vector_size': None}


def upsert_points(
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 100
):
    """Insert points into Qdrant in batches."""
    client = qdrant_manager.get_client()
    sanitized_name = sanitize_collection_name(collection_name)
    
    log.info(f"Upserting {len(points)} points to {sanitized_name}...")
    
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=sanitized_name, points=batch)
        log.info(f"✓ Batch {i//batch_size + 1}: {len(batch)} points")
    
    log.info(f"✓ Total {len(points)} points upserted to {sanitized_name}")


def delete_points_by_source_file(
    collection_name: str,
    source_file: str
) -> int:
    """
    Delete all points that have a specific source_file in their payload.
    Returns the number of points deleted.
    """
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)
        
        # Check if collection exists
        existing = [c.name for c in client.get_collections().collections]
        if sanitized_name not in existing:
            log.warning(f"Collection not found: {sanitized_name}")
            return 0
        
        # Create filter for the source file
        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file)
                )
            ]
        )
        
        # Count points before deletion (for logging)
        count_result = client.count(
            collection_name=sanitized_name,
            count_filter=delete_filter,
            exact=True
        )
        points_to_delete = count_result.count
        
        if points_to_delete == 0:
            log.info(f"No points found for source_file: {source_file}")
            return 0
        
        # Delete points matching the filter
        client.delete(
            collection_name=sanitized_name,
            points_selector=FilterSelector(filter=delete_filter)
        )
        
        log.info(f"🗑️  Deleted {points_to_delete} points for source_file: {source_file}")
        return points_to_delete
    
    except Exception as e:
        log.error(f"Error deleting points by source_file: {e}")
        return 0


def get_existing_source_files(collection_name: str) -> Set[str]:
    """
    Get all unique source_file values from the collection.
    Useful for checking which documents are already indexed.
    """
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)
        
        # Check if collection exists
        existing = [c.name for c in client.get_collections().collections]
        if sanitized_name not in existing:
            log.warning(f"Collection not found: {sanitized_name}")
            return set()
        
        # Scroll through all points to get unique source files
        source_files = set()
        offset = None
        batch_size = 100
        
        while True:
            results, offset = client.scroll(
                collection_name=sanitized_name,
                limit=batch_size,
                offset=offset,
                with_payload=["source_file"],
                with_vectors=False
            )
            
            for point in results:
                if point.payload and 'source_file' in point.payload:
                    source_files.add(point.payload['source_file'])
            
            if offset is None:
                break
        
        log.info(f"Found {len(source_files)} unique source files in {sanitized_name}")
        return source_files
    
    except Exception as e:
        log.error(f"Error getting existing source files: {e}")
        return set()


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
    sanitized_name = sanitize_collection_name(collection_name)
    
    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if sanitized_name not in existing:
        log.warning(f"Collection not found: {sanitized_name}")
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
        collection_name=sanitized_name,
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
                'metadata': payload,
                'retrieval_keywords': payload.get('retrieval_keywords', []),
                'section_heading': payload.get('section_heading'),
                'section_heading_tokens': payload.get('section_heading_tokens', []),
                'section_labels': payload.get('section_labels', []),
                'contains_team_members': bool(payload.get('contains_team_members')),
                'contains_team_lead': bool(payload.get('contains_team_lead')),
                'contains_leadership': bool(payload.get('contains_leadership')),
                'contains_department': bool(payload.get('contains_department')),
                'contains_core_team': bool(payload.get('contains_core_team')),
                'contains_qa_pair': bool(payload.get('contains_qa_pair')),
                'contains_name_role_pairs': bool(payload.get('contains_name_role_pairs')),
                'contains_people_list': bool(payload.get('contains_people_list')),
            })
    
    log.info(f"Found {len(chunks)} chunks (score >= {min_score}) from {sanitized_name}")
    return chunks


def _char_trigrams(token: str) -> set:
    """Return the set of character trigrams for a token."""
    if len(token) < 3:
        return {token}
    return {token[i:i + 3] for i in range(len(token) - 2)}


def _fuzzy_token_recall(query_tokens: set, token_space: set, threshold: float = 0.35) -> float:
    """
    For each query token that has no exact match, check if any token in
    token_space is 'close enough' via character trigram Jaccard similarity.
    Returns the count of fuzzy-matched query tokens (for use in recall calc).
    Only applies to tokens long enough for trigrams to be meaningful (>= 4 chars).
    """
    unmatched = query_tokens - token_space
    if not unmatched:
        return 0.0

    fuzzy_matched = 0
    for qt in unmatched:
        if len(qt) < 4:
            continue
        qt_grams = _char_trigrams(qt)
        for ct in token_space:
            if len(ct) < 4:
                continue
            ct_grams = _char_trigrams(ct)
            union = qt_grams | ct_grams
            if not union:
                continue
            jaccard = len(qt_grams & ct_grams) / len(union)
            if jaccard >= threshold:
                fuzzy_matched += 1
                break  # one match per query token is enough

    return float(fuzzy_matched)


def search_lexical(
    collection_name: str,
    query_text: str,
    limit: int = 10,
    filters: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Lightweight lexical retrieval over chunk payloads.
    This complements dense retrieval for short or structured lookup queries.
    """
    client = qdrant_manager.get_client()
    sanitized_name = sanitize_collection_name(collection_name)

    existing = [c.name for c in client.get_collections().collections]
    if sanitized_name not in existing:
        log.warning(f"Collection not found: {sanitized_name}")
        return []

    query_tokens = set(extract_content_tokens(query_text))
    if not query_tokens:
        return []

    scroll_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        scroll_filter = Filter(must=conditions)

    lexical_hits: List[Dict[str, Any]] = []
    offset = None
    batch_size = 128

    while True:
        results, offset = client.scroll(
            collection_name=sanitized_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in results:
            payload = point.payload or {}
            chunk_keywords = set(payload.get("retrieval_keywords") or [])
            heading_tokens = set(payload.get("section_heading_tokens") or [])
            chunk_text = payload.get("text", "") or ""
            chunk_tokens = set(extract_content_tokens(chunk_text))
            token_space = chunk_keywords | chunk_tokens | heading_tokens
            if not token_space:
                continue

            exact_matches = len(query_tokens & token_space)
            fuzzy_matches = _fuzzy_token_recall(query_tokens, token_space)
            recall = (exact_matches + fuzzy_matches) / len(query_tokens)
            if recall <= 0:
                continue

            phrase = " ".join(extract_content_tokens(query_text))
            normalized_chunk = " ".join(extract_content_tokens(chunk_text))
            phrase_bonus = 0.25 if phrase and phrase in normalized_chunk else 0.0
            people_bonus = 0.1 if payload.get("contains_people_list") else 0.0
            lexical_score = min(1.0, recall + phrase_bonus + people_bonus)

            lexical_hits.append({
                "text": chunk_text,
                "source_file": payload.get("source_file"),
                "document_type": payload.get("document_type"),
                "chunk_index": payload.get("chunk_index"),
                "score": lexical_score,
                "raw_score": lexical_score,
                "metadata": payload,
                "retrieval_keywords": payload.get("retrieval_keywords", []),
                "section_heading": payload.get("section_heading"),
                "section_heading_tokens": payload.get("section_heading_tokens", []),
                "section_labels": payload.get("section_labels", []),
                "contains_team_members": bool(payload.get("contains_team_members")),
                "contains_team_lead": bool(payload.get("contains_team_lead")),
                "contains_leadership": bool(payload.get("contains_leadership")),
                "contains_department": bool(payload.get("contains_department")),
                "contains_core_team": bool(payload.get("contains_core_team")),
                "contains_qa_pair": bool(payload.get("contains_qa_pair")),
                "contains_name_role_pairs": bool(payload.get("contains_name_role_pairs")),
                "contains_people_list": bool(payload.get("contains_people_list")),
            })

        if offset is None:
            break

    lexical_hits.sort(key=lambda chunk: float(chunk.get("score", 0.0) or 0.0), reverse=True)
    hits = lexical_hits[:limit]
    log.info(f"Found {len(hits)} lexical chunks from {sanitized_name}")
    return hits


def delete_collection(collection_name: str) -> bool:
    """Delete Qdrant collection."""
    client = qdrant_manager.get_client()
    sanitized_name = sanitize_collection_name(collection_name)
    
    existing = [c.name for c in client.get_collections().collections]
    if sanitized_name in existing:
        client.delete_collection(sanitized_name)
        log.info(f"🗑️  Deleted collection: {sanitized_name}")
        return True
    
    log.warning(f"Collection not found: {sanitized_name}")
    return False


def get_points_count(collection_name: str) -> int:
    """Get the total number of points in a collection."""
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)
        
        existing = [c.name for c in client.get_collections().collections]
        if sanitized_name not in existing:
            return 0
        
        collection_info = client.get_collection(sanitized_name)
        return collection_info.points_count
    except Exception as e:
        log.error(f"Error getting points count: {e}")
        return 0


def get_points_by_source_file(
    collection_name: str,
    source_file: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get all points for a specific source file.
    Useful for debugging and verification.
    """
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)
        
        existing = [c.name for c in client.get_collections().collections]
        if sanitized_name not in existing:
            log.warning(f"Collection not found: {sanitized_name}")
            return []
        
        # Create filter for the source file
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file)
                )
            ]
        )
        
        # Scroll through points with this filter
        results, _ = client.scroll(
            collection_name=sanitized_name,
            scroll_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        points = []
        for point in results:
            points.append({
                'id': point.id,
                'payload': point.payload
            })
        
        log.info(f"Found {len(points)} points for source_file: {source_file}")
        return points
    
    except Exception as e:
        log.error(f"Error getting points by source file: {e}")
        return []


def fetch_chunk_by_index(
    collection_name: str,
    source_file: str,
    chunk_index: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch a single chunk by source_file + chunk_index.
    Used to retrieve adjacent chunks (e.g. chunk N+1 when chunk N is selected).
    Returns None if not found.
    """
    try:
        client = qdrant_manager.get_client()
        sanitized_name = sanitize_collection_name(collection_name)

        results, _ = client.scroll(
            collection_name=sanitized_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return None

        payload = results[0].payload or {}
        return {
            "id": results[0].id,
            "text": payload.get("text", ""),
            "source_file": payload.get("source_file", source_file),
            "chunk_index": payload.get("chunk_index", chunk_index),
            "document_type": payload.get("document_type", "unknown"),
            "score": 0.0,
            "raw_score": 0.0,
            "adjacent_chunk": True,
            "retrieval_keywords": payload.get("retrieval_keywords", []),
            "section_heading": payload.get("section_heading"),
            "section_heading_tokens": payload.get("section_heading_tokens", []),
            "section_labels": payload.get("section_labels", []),
            "contains_team_members": bool(payload.get("contains_team_members")),
            "contains_team_lead": bool(payload.get("contains_team_lead")),
            "contains_leadership": bool(payload.get("contains_leadership")),
            "contains_department": bool(payload.get("contains_department")),
            "contains_core_team": bool(payload.get("contains_core_team")),
            "contains_qa_pair": bool(payload.get("contains_qa_pair")),
            "contains_name_role_pairs": bool(payload.get("contains_name_role_pairs")),
            "contains_people_list": bool(payload.get("contains_people_list")),
        }
    except Exception as exc:
        log.warning(f"fetch_chunk_by_index failed (non-fatal): {exc}")
        return None
