"""
Refactored Chatbot Service - FUNCTIONAL APPROACH
"""

# ============================================================================
# services/chatbot_service.py - FUNCTIONAL CHATBOT CRUD
# ============================================================================

"""Functional chatbot CRUD service."""

import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, BackgroundTasks
from qdrant_client.models import PointStruct

from stateful_services.db_schema import Chatbot, Question
from utils.document_service import extract_text_from_document, get_file_extension
from utils.logging import log

# Import functional services
from utils.embedding import process_document_for_embedding
from services.vectore_store_service import (
    create_collection,
    upsert_points,
    delete_collection
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_questions(db: Session, chatbot_id: uuid.UUID, questions: dict):
    """Save or update questions for quiz mode."""
    existing = db.query(Question).filter(
        Question.chatbot_id == chatbot_id
    ).first()
    
    if existing:
        existing.question_data = questions
    else:
        new_questions = Question(
            chatbot_id=chatbot_id,
            question_data=questions
        )
        db.add(new_questions)
    
    db.commit()
    log.info(f"✓ Questions saved for chatbot {chatbot_id}")


def process_documents_background(
    chatbot_name: str,
    doc_contents: List[bytes],
    doc_names: List[str]
):
    """
    Background task: Process documents and store embeddings.
    
    KEY: Embeddings generated ONCE per chunk with cost logging.
    """
    try:
        log.info(f"📄 Processing {len(doc_contents)} documents for '{chatbot_name}'")
        
        all_points = []
        point_id = 0
        
        # Process each document
        for doc_content, doc_name in zip(doc_contents, doc_names):
            try:
                log.info(f"Processing: {doc_name}")
                
                # Extract text
                text = extract_text_from_document(doc_content, doc_name)
                
                if not text or not text.strip():
                    log.warning(f"⚠️  No text from {doc_name}")
                    continue
                
                log.info(f"✓ Extracted {len(text)} chars from {doc_name}")
                
                doc_type = get_file_extension(doc_name) or 'unknown'
                
                # Chunk and generate embeddings (costs logged inside)
                chunks_with_embeddings = process_document_for_embedding(
                    text=text,
                    metadata={
                        'chatbot_name': chatbot_name,
                        'source_file': doc_name,
                        'document_type': doc_type
                    }
                )
                
                if not chunks_with_embeddings:
                    log.warning(f"⚠️  No chunks from {doc_name}")
                    continue
                
                log.info(f"✓ {len(chunks_with_embeddings)} chunks with embeddings")
                
                # Prepare Qdrant points
                for chunk in chunks_with_embeddings:
                    point = PointStruct(
                        id=point_id,
                        vector=chunk['embedding'],
                        payload={
                            'text': chunk['text'],
                            'chatbot_name': chatbot_name,
                            'source_file': doc_name,
                            'document_type': doc_type,
                            'chunk_index': chunk['chunk_index'],
                            'total_chunks': chunk['total_chunks'],
                            'char_count': chunk['char_count']
                        }
                    )
                    all_points.append(point)
                    point_id += 1
            
            except Exception as e:
                log.error(f"❌ Error processing {doc_name}: {e}", exc_info=True)
                continue
        
        if not all_points:
            log.error(f"❌ No embeddings generated for '{chatbot_name}'")
            return
        
        # Create collection and upload
        vector_size = len(all_points[0].vector)
        create_collection(
            collection_name=chatbot_name,
            vector_size=vector_size,
            force_recreate=True
        )
        
        upsert_points(
            collection_name=chatbot_name,
            points=all_points
        )
        
        log.info(f"✅ Stored {len(all_points)} embeddings for '{chatbot_name}'")
    
    except Exception as e:
        log.error(f"❌ Background processing failed: {e}", exc_info=True)


# ============================================================================
# MAIN CRUD FUNCTIONS
# ============================================================================

def create_chatbot(
    db: Session,
    chatbot_name: str,
    status: str,
    description: Optional[str] = None,
    instruction: Optional[str] = None,
    doc_contents: Optional[List[bytes]] = None,
    doc_names: Optional[List[str]] = None,
    generated_by: Optional[uuid.UUID] = None,
    meta_data: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = "general",
    questions: Optional[dict] = None,
    background_tasks: Optional[BackgroundTasks] = None
) -> Chatbot:
    """Create chatbot and schedule document processing."""
    try:
        # Check uniqueness
        existing = db.query(Chatbot).filter(
            Chatbot.chatbot_name == chatbot_name
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Chatbot '{chatbot_name}' already exists"
            )
        
        # Create chatbot record
        chatbot = Chatbot(
            chatbot_name=chatbot_name,
            description=description,
            instruction=instruction,
            pdf_names=doc_names or [],
            status=status,
            mode=mode,
            generated_by=generated_by,
            meta_data=meta_data or {}
        )
        
        db.add(chatbot)
        db.commit()
        db.refresh(chatbot)
        
        log.info(f"✓ Chatbot created: {chatbot_name} (ID: {chatbot.chatbot_id})")
        
        # Save questions if quiz mode
        if questions and mode in ("quiz", "people_analyzer"):
            save_questions(db, chatbot.chatbot_id, questions)
        
        # Schedule document processing
        if doc_contents and doc_names and background_tasks:
            background_tasks.add_task(
                process_documents_background,
                chatbot_name=chatbot_name,
                doc_contents=doc_contents,
                doc_names=doc_names
            )
            log.info(f"📋 Scheduled: {len(doc_names)} documents")
        
        return chatbot
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"❌ Create failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def update_chatbot(
    db: Session,
    chatbot_id: uuid.UUID,
    chatbot_name: Optional[str] = None,
    status: Optional[str] = None,
    description: Optional[str] = None,
    instruction: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    doc_contents: Optional[List[bytes]] = None,
    doc_names: Optional[List[str]] = None,
    questions: Optional[dict] = None,
    background_tasks: Optional[BackgroundTasks] = None
) -> Chatbot:
    """Update chatbot."""
    try:
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_id == chatbot_id
        ).first()
        
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Check name uniqueness if changing
        if chatbot_name and chatbot_name != chatbot.chatbot_name:
            exists = db.query(Chatbot).filter(
                Chatbot.chatbot_name == chatbot_name,
                Chatbot.chatbot_id != chatbot_id
            ).first()
            if exists:
                raise HTTPException(
                    status_code=400,
                    detail="Chatbot name already exists"
                )
            chatbot.chatbot_name = chatbot_name
        
        # Update fields
        if status:
            chatbot.status = status
        if description is not None:
            chatbot.description = description
        if instruction is not None:
            chatbot.instruction = instruction
        if meta_data is not None:
            chatbot.meta_data = meta_data
        if mode:
            chatbot.mode = mode
        if doc_names:
            chatbot.pdf_names = doc_names
        
        db.commit()
        db.refresh(chatbot)
        
        # Update questions if quiz mode
        if questions and chatbot.mode == "quiz":
            save_questions(db, chatbot.chatbot_id, questions)
        
        # Process new documents
        if doc_contents and doc_names and background_tasks:
            background_tasks.add_task(
                process_documents_background,
                chatbot_name=chatbot.chatbot_name,
                doc_contents=doc_contents,
                doc_names=doc_names
            )
        
        log.info(f"✓ Chatbot updated: {chatbot.chatbot_id}")
        return chatbot
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"❌ Update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def delete_chatbot_by_id(db: Session, chatbot_id: uuid.UUID) -> bool:
    """Delete chatbot and its vector collection."""
    try:
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_id == chatbot_id
        ).first()
        
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        chatbot_name = chatbot.chatbot_name
        
        # Delete vector collection
        delete_collection(chatbot_name)
        
        # Delete questions
        db.query(Question).filter(Question.chatbot_id == chatbot_id).delete()
        
        # Delete chatbot
        db.delete(chatbot)
        db.commit()
        
        log.info(f"🗑️  Chatbot deleted: {chatbot_name}")
        return True
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"❌ Delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


