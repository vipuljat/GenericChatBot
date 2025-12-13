# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from qdrant_client.http import models
# from qdrant_client.http.exceptions import ResponseHandlingException
# import config
# from typing import List, Dict
# import logging
# import os
# import uuid

# logger = logging.getLogger(__name__)


# class QdrantService:
#     """Service for managing Qdrant vector database operations"""

#     def __init__(self):
#         self.client = None
#         self.collection_name = config.QDRANT_COLLECTION_NAME
#         self._reinitialize_client()

#     def _reinitialize_client(self):
#         """Initialize or reinitialize the Qdrant client"""
#         self.client = self._initialize_client()
#         self._ensure_collection_exists()

#     def _initialize_client(self) -> QdrantClient:
#         """Initialize Qdrant client"""
#         try:
#             if config.QDRANT_API_KEY:
#                 # Cloud mode with API key
#                 client = QdrantClient(
#                     url=config.QDRANT_URL,
#                     api_key=config.QDRANT_API_KEY
#                 )
#                 logger.info("Qdrant client initialized (cloud mode)")

#             elif config.QDRANT_URL.startswith("http://") or config.QDRANT_URL.startswith("https://"):
#                 # Docker container mode (HTTP URL)
#                 client = QdrantClient(
#                     url=f"{config.QDRANT_URL}:{config.QDRANT_PORT}"
#                 )
#                 logger.info(f"Qdrant client initialized (Docker mode: {config.QDRANT_URL}:{config.QDRANT_PORT})")

#             elif config.QDRANT_STORAGE_PATH:
#                 # Local file-based storage mode
#                 storage_path = config.QDRANT_STORAGE_PATH
#                 os.makedirs(storage_path, exist_ok=True)
#                 client = QdrantClient(path=storage_path)
#                 logger.info(f"Qdrant client initialized (local storage mode: {storage_path})")

#             else:
#                 # Default: in-memory mode
#                 client = QdrantClient(":memory:")
#                 logger.info("Qdrant client initialized (in-memory mode)")

#             return client

#         except Exception as e:
#             logger.error(f"Failed to initialize Qdrant client: {str(e)}")
#             raise

#     def _ensure_collection_exists(self):
#         """Create collection if it doesn't exist"""
#         try:
#             if self.client is None:
#                 logger.warning("Qdrant client was None during collection ensure; reinitializing.")
#                 self._reinitialize_client()
#                 if self.client is None:
#                     raise RuntimeError("Qdrant client failed to initialize")
#             collections = self.client.get_collections().collections
#             collection_names = [c.name for c in collections]

#             if self.collection_name not in collection_names:
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(
#                         size=1536,
#                         distance=Distance.COSINE
#                     )
#                 )
#                 logger.info(f"Created collection: {self.collection_name}")
#             else:
#                 logger.info(f"Collection {self.collection_name} already exists")

#         except Exception as e:
#             logger.error(f"Error ensuring collection exists: {str(e)}")
#             raise

#     def store_embeddings(
#         self,
#         embeddings: List[List[float]],
#         metadata_list: List[Dict],
#         chatbot_id: int
#     ) -> bool:
#         """Store embeddings in Qdrant with metadata"""
#         try:
#             if self.client is None:
#                 logger.warning("Qdrant client None before storing embeddings; reinitializing.")
#                 self._reinitialize_client()
#                 if self.client is None:
#                     raise RuntimeError("Qdrant client unavailable for storing embeddings")
#             points = []

#             for embedding, metadata in zip(embeddings, metadata_list):
#                 point_id = str(uuid.uuid4())
#                 metadata["chatbot_id"] = chatbot_id

#                 point = PointStruct(
#                     id=point_id,
#                     vector=embedding,
#                     payload=metadata
#                 )
#                 points.append(point)

#             batch_size = 100
#             for i in range(0, len(points), batch_size):
#                 batch = points[i:i + batch_size]
#                 self.client.upsert(
#                     collection_name=self.collection_name,
#                     points=batch
#                 )

#             logger.info(f"Stored {len(points)} embeddings for chatbot {chatbot_id}")
#             return True

#         except Exception as e:
#             logger.error(f"Error storing embeddings: {str(e)}")
#             return False

#     def search_similar(
#         self,
#         query_embedding: List[float],
#         chatbot_id: int,
#         limit: int = 5
#     ) -> List[Dict]:
#         """Search for similar embeddings"""
#         try:
#             if not hasattr(self.client, "query_points"):
#                 logger.warning("Client missing query_points, reinitializing...")
#                 self._reinitialize_client()
#             if self.client is None:
#                 raise RuntimeError("Qdrant client unavailable for search")

#             # Correct qdrant-client 1.14.x usage: pass embedding as `query`
#             try:
#                 search_result = self.client.query_points(
#                     collection_name=self.collection_name,
#                     query=query_embedding,
#                     query_filter=models.Filter(
#                         must=[
#                             models.FieldCondition(
#                                 key="chatbot_id",
#                                 match=models.MatchValue(value=chatbot_id)
#                             )
#                         ]
#                     ),
#                     limit=limit
#                 )
#             except ResponseHandlingException as re:
#                 # Fallback for older Qdrant server versions that lack /points/query endpoint
#                 if "404" in str(re):
#                     logger.warning("query_points endpoint returned 404 - falling back to legacy search() API")
#                     # Legacy search signature
#                     search_result = self.client.search(
#                         collection_name=self.collection_name,
#                         query_vector=query_embedding,
#                         query_filter=models.Filter(
#                             must=[
#                                 models.FieldCondition(
#                                     key="chatbot_id",
#                                     match=models.MatchValue(value=chatbot_id)
#                                 )
#                             ]
#                         ),
#                         limit=limit
#                     )
#                 else:
#                     raise

#             results = []
#             # Both query_points and search return objects with .points list (search returns List[ScoredPoint])
#             points_attr = getattr(search_result, 'points', search_result)
#             for hit in points_attr:
#                 results.append({
#                     "id": getattr(hit, 'id', None),
#                     "score": getattr(hit, 'score', None),
#                     "metadata": getattr(hit, 'payload', {})
#                 })

#             return results

#         except Exception as e:
#             logger.error(f"Error searching embeddings: {str(e)}")
#             return []

#     def delete_chatbot_embeddings(self, chatbot_id: int) -> bool:
#         """Delete all embeddings for a specific chatbot"""
#         try:
#             if self.client is None:
#                 logger.warning("Qdrant client None before delete; reinitializing.")
#                 self._reinitialize_client()
#                 if self.client is None:
#                     raise RuntimeError("Qdrant client unavailable for delete operation")
#             self.client.delete(
#                 collection_name=self.collection_name,
#                 points_selector=models.FilterSelector(
#                     filter=models.Filter(
#                         must=[
#                             models.FieldCondition(
#                                 key="chatbot_id",
#                                 match=models.MatchValue(value=chatbot_id)
#                             )
#                         ]
#                     )
#                 )
#             )
#             logger.info(f"Deleted embeddings for chatbot {chatbot_id}")
#             return True

#         except Exception as e:
#             logger.error(f"Error deleting embeddings: {str(e)}")
#             return False


# # Singleton instance
# _qdrant_service = None


# def get_qdrant_service() -> QdrantService:
#     global _qdrant_service
#     if _qdrant_service is None:
#         _qdrant_service = QdrantService()
#     return _qdrant_service


# def reset_qdrant_service():
#     global _qdrant_service
#     _qdrant_service = None


# def qdrant_healthcheck() -> bool:
#     """Simple health check to verify Qdrant connectivity."""
#     svc = get_qdrant_service()
#     try:
#         if svc.client is None:
#             return False
#         svc.client.get_collections()
#         return True
#     except Exception as e:
#         logger.error(f"Qdrant health check failed: {e}")
#         return False
