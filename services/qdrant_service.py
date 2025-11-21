from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import config
from typing import List, Dict, Optional
import logging
import os
import uuid

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for managing Qdrant vector database operations"""
    
    def __init__(self):
        self.client = None
        self.collection_name = config.QDRANT_COLLECTION_NAME
        self._reinitialize_client()
    
    def _reinitialize_client(self):
        """Initialize or reinitialize the Qdrant client"""
        self.client = self._initialize_client()
        self._ensure_collection_exists()
    
    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client"""
        try:
            if config.QDRANT_API_KEY:
                # Cloud/Remote instance
                client = QdrantClient(
                    url=config.QDRANT_URL,
                    api_key=config.QDRANT_API_KEY
                )
                logger.info("Qdrant client initialized (cloud mode)")
            elif config.QDRANT_URL == "localhost" or config.QDRANT_URL.startswith("127.0.0.1"):
                # Always use local file-based storage for localhost (no Docker/server needed)
                storage_path = config.QDRANT_STORAGE_PATH or "./qdrant_storage"
                os.makedirs(storage_path, exist_ok=True)
                
                try:
                    client = QdrantClient(path=storage_path)
                    logger.info(f"Qdrant client initialized (local storage mode: {storage_path})")
                except Exception as e:
                    # If storage is corrupted, use in-memory storage as fallback
                    logger.error(f"Corrupted storage detected: {e}")
                    logger.warning("Using in-memory storage (data will be lost on restart)")
                    logger.warning("Run: python force_cleanup.py to fix storage")
                    client = QdrantClient(":memory:")
                    logger.info("Qdrant client initialized (in-memory mode)")
            else:
                # Remote instance without API key
                client = QdrantClient(
                    host=config.QDRANT_URL,
                    port=config.QDRANT_PORT
                )
                logger.info("Qdrant client initialized (remote server mode)")
            
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def store_embeddings(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict],
        chatbot_id: int
    ) -> bool:
        """
        Store embeddings in Qdrant with metadata
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries for each embedding
            chatbot_id: ID of the chatbot these embeddings belong to
            
        Returns:
            bool: Success status
        """
        try:
            points = []
            for idx, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
                # Generate UUID for point ID
                point_id = str(uuid.uuid4())
                
                # Add chatbot_id to metadata for filtering
                metadata['chatbot_id'] = chatbot_id
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Stored {len(points)} embeddings for chatbot {chatbot_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        chatbot_id: int,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query vector
            chatbot_id: Filter by chatbot ID
            limit: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Validate client has search method
            if not hasattr(self.client, 'search'):
                logger.warning("Client missing search method, reinitializing...")
                self._reinitialize_client()
            
            # Use search method with correct parameters for Qdrant 1.14.x
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chatbot_id",
                            match=models.MatchValue(value=chatbot_id)
                        )
                    ]
                ),
                limit=limit
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.payload
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching embeddings: {str(e)}")
            return []
    
    def delete_chatbot_embeddings(self, chatbot_id: int) -> bool:
        """Delete all embeddings for a specific chatbot"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chatbot_id",
                                match=models.MatchValue(value=chatbot_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted embeddings for chatbot {chatbot_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return False

# Singleton instance
_qdrant_service = None

def get_qdrant_service() -> QdrantService:
    """Get or create Qdrant service instance"""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    # Validate client is still functional
    elif not hasattr(_qdrant_service.client, 'search'):
        logger.warning("Singleton client is invalid, recreating...")
        _qdrant_service = QdrantService()
    return _qdrant_service

def reset_qdrant_service():
    """Reset the singleton instance (useful after errors or reloads)"""
    global _qdrant_service
    _qdrant_service = None
