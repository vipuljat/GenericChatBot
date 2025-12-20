from sqlalchemy.orm import Session
from models.chatbot import Chatbot
from models.documents import ChatbotDocument
from fastapi import UploadFile
from typing import Optional, List
import logging
from services.document_service import get_document_service
# from services.embedding_service import get_embedding_service
# from services.qdrant_service import get_qdrant_service

logger = logging.getLogger(__name__)

class ChatbotService:
    """Service for managing chatbot operations"""
    
    def __init__(self):
        self.document_service = get_document_service()
        # self.embedding_service = get_embedding_service()
        # self.qdrant_service = get_qdrant_service()
    
    def _process_and_store_embeddings(
        self,
        file_path: str,
        doc_id: int,
        chatbot_id: int,
        doc_name: str
    ) -> bool:
        """
        Process document and store embeddings in Qdrant
        
        Args:
            file_path: Path to the document file
            doc_id: Document ID from database
            chatbot_id: Chatbot ID
            doc_name: Original document name
            
        Returns:
            bool: Success status
        """
        try:
            # Step 1: Process document - extract text and chunk it
            logger.info(f"Processing document: {doc_name}")
            chunk_texts, chunk_metadata = self.document_service.process_document(
                file_path=file_path,
                doc_id=doc_id,
                chatbot_id=chatbot_id,
                doc_name=doc_name
            )
            
            if not chunk_texts:
                logger.warning(f"No chunks generated for document {doc_name}")
                return False
            
            # Step 2: Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            # Step 3: Store embeddings in Qdrant
            logger.info("Storing embeddings in Qdrant")
            success = self.qdrant_service.store_embeddings(
                embeddings=embeddings,
                metadata_list=chunk_metadata,
                chatbot_id=chatbot_id
            )
            
            if success:
                logger.info(f"Successfully stored embeddings for document {doc_name}")
            else:
                logger.error(f"Failed to store embeddings for document {doc_name}")
            
            return success
        except Exception as e:
            logger.error(f"Error in embedding pipeline: {str(e)}")
            return False

def create_chatbot_with_documents(
    db: Session,
    chatbot_name: str,
    description: str,
    instructions: str,
    conversation_starters: Optional[list],
    is_quiz_mode: bool,
    is_active: bool,
    recommended_model: Optional[str],
    file: UploadFile,
    quiz_file: Optional[UploadFile]
) -> int:
    """
    Create a new chatbot with documents and process embeddings
    
    Args:
        db: Database session
        chatbot_name: Name of the chatbot
        description: Description of the chatbot
        instructions: Instructions for the chatbot
        conversation_starters: List of conversation starters
        is_quiz_mode: Whether quiz mode is enabled
        is_active: Whether chatbot is active
        recommended_model: Recommended AI model
        file: Main document file
        quiz_file: Quiz document file (optional)
        
    Returns:
        int: Created chatbot ID
    """
    try:
        chatbot_service = ChatbotService()
        document_service = get_document_service()
        
        # Step 1: Create chatbot entry in database
        logger.info(f"Creating chatbot: {chatbot_name}")
        chatbot = Chatbot(
            chatbot_name=chatbot_name,
            description=description,
            instructions=instructions,
            conversation_starters=conversation_starters,
            is_quiz_mode=is_quiz_mode,
            is_active=is_active,
            recommended_model=recommended_model
        )
        db.add(chatbot)
        db.commit()
        db.refresh(chatbot)
        
        chatbot_id = chatbot.id
        logger.info(f"Created chatbot with ID: {chatbot_id}")
        
        # Step 2: Save and process main document
        if file:
            logger.info(f"Processing main document: {file.filename}")
            
            # Save file to disk
            file_path = document_service.save_uploaded_file(file)
            
            # Create document entry in database
            normal_doc = ChatbotDocument(
                chatbot_id=chatbot_id,
                doc_name=file.filename,
                file_path=file_path,
                is_quiz_document=False
            )
            db.add(normal_doc)
            db.commit()
            db.refresh(normal_doc)
            
            # Process and store embeddings
            chatbot_service._process_and_store_embeddings(
                file_path=file_path,
                doc_id=int(normal_doc.id),
                chatbot_id=int(chatbot_id),
                doc_name=str(file.filename)
            )
        
        # Step 3: Save and process quiz document (if provided)
        if is_quiz_mode and quiz_file:
            logger.info(f"Processing quiz document: {quiz_file.filename}")
            
            # Save file to disk
            quiz_path = document_service.save_uploaded_file(quiz_file)
            
            # Create quiz document entry in database
            quiz_doc = ChatbotDocument(
                chatbot_id=chatbot_id,
                doc_name=quiz_file.filename,
                file_path=quiz_path,
                is_quiz_document=True
            )
            db.add(quiz_doc)
            db.commit()
            db.refresh(quiz_doc)
            
            # Process and store embeddings for quiz document
            chatbot_service._process_and_store_embeddings(
                file_path=quiz_path,
                doc_id=int(quiz_doc.id),
                chatbot_id=int(chatbot_id),
                doc_name=str(quiz_file.filename)
            )
        
        db.commit()
        logger.info(f"Chatbot {chatbot_id} created successfully with embeddings")
        return int(chatbot_id)
    
    except Exception as e:
        logger.error(f"Error creating chatbot: {str(e)}")
        db.rollback()
        raise

def get_chatbot_service() -> ChatbotService:
    """Get chatbot service instance"""
    return ChatbotService()
