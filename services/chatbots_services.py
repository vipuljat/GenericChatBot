"""Chatbot service with multi-format document processing and chunked embeddings."""

import uuid
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import config
from sqlalchemy.orm import Session
from fastapi import HTTPException, BackgroundTasks
from qdrant_client.models import Distance, VectorParams, PointStruct
from stateful_services.db_schema import Chatbot
from stateful_services.database import qdrant_manager
from utils.document_service import extract_text_from_document, get_file_extension
from utils.embedding import process_document_for_embedding
from utils.embedding import generate_embedding
from utils.logging import log

genai.configure(api_key=config.GEMINI_API_KEY)

def _sanitize_collection_name(chatbot_name: str) -> str:
    """
    Sanitize chatbot name to create a valid Qdrant collection name.
    
    Qdrant collection names must:
    - Start with a letter or underscore
    - Contain only letters, numbers, underscores, and hyphens
    - Be between 1-255 characters
    
    Args:
        chatbot_name: Original chatbot name
        
    Returns:
        Sanitized collection name
    """
    import re
    
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', chatbot_name)
    
    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = f'_{sanitized}'
    
    # Limit length to 255 characters
    sanitized = sanitized[:255]
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'chatbot_default'
    
    return sanitized


def _process_documents_and_store_embeddings(
    chatbot_name: str,
    doc_contents: List[bytes],
    doc_names: List[str]
):
    """
    Background task: Extract text from documents, chunk, generate embeddings, and store in Qdrant.
    
    Supports: PDF, DOCX, DOC, TXT, RTF
    
    Args:
        chatbot_name: Name of the chatbot (used as collection name)
        doc_contents: List of document file contents
        doc_names: List of document filenames
    """
    try:
        log.info(
            f"Starting document processing for chatbot '{chatbot_name}' "
            f"({len(doc_contents)} documents)"
        )
        
        client = qdrant_manager.get_client()
        if not client:
            log.error("Qdrant client unavailable for background task")
            return
        
        # Use chatbot name directly as collection name (sanitized)
        collection_name = _sanitize_collection_name(chatbot_name)
        
        log.info(f"Using collection name: {collection_name}")
        
        # Process each document
        all_points = []
        point_id = 0
        
        for doc_content, doc_name in zip(doc_contents, doc_names):
            try:
                log.info(f"Processing document: {doc_name}")
                
                # Extract text from document (supports multiple formats)
                text = extract_text_from_document(doc_content, doc_name)
                
                if not text or not text.strip():
                    log.warning(f"No text extracted from {doc_name}")
                    continue
                
                log.info(
                    f"Extracted {len(text)} characters from {doc_name}"
                )
                
                doc_type = get_file_extension(doc_name) or 'unknown'
                
                # Chunk and generate embeddings
                chunks_with_embeddings = process_document_for_embedding(
                    text=text,
                    metadata={
                        'chatbot_name': chatbot_name,
                        'source_file': doc_name,
                        'document_type': doc_type
                    }
                )
                
                if not chunks_with_embeddings:
                    log.warning(f"No chunks generated from {doc_name}")
                    continue
                
                log.info(
                    f"Generated {len(chunks_with_embeddings)} chunks "
                    f"with embeddings for {doc_name}"
                )
                
                # Prepare Qdrant points
                for chunk_data in chunks_with_embeddings:
                    point = PointStruct(
                        id=point_id,
                        vector=chunk_data['embedding'],
                        payload={
                            'text': chunk_data['text'],
                            'chatbot_name': chatbot_name,
                            'source_file': doc_name,
                            'document_type': doc_type,
                            'chunk_index': chunk_data['chunk_index'],
                            'total_chunks': chunk_data['total_chunks'],
                            'char_count': chunk_data['char_count']
                        }
                    )
                    all_points.append(point)
                    point_id += 1
                    
            except Exception as e:
                log.error(f"Error processing document {doc_name}: {e}", exc_info=True)
                continue
        
        if not all_points:
            log.error(f"No embeddings generated for chatbot '{chatbot_name}'")
            return
        
        # Get vector dimension from first embedding
        vector_size = len(all_points[0].vector)
        
        # Create or recreate collection
        existing_collections = [col.name for col in client.get_collections().collections]
        
        if collection_name in existing_collections:
            log.info(f"Recreating existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        log.info(f"Created Qdrant collection: {collection_name}")
        
        # Upload points to Qdrant in batches
        batch_size = 100
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            log.info(
                f"Uploaded batch {i//batch_size + 1} "
                f"({len(batch)} points) to Qdrant"
            )
        
        log.info(
            f"Successfully stored {len(all_points)} embeddings in Qdrant "
            f"for chatbot '{chatbot_name}' (collection: {collection_name})"
        )
        
    except Exception as e:
        log.error(
            f"Background task failed for chatbot '{chatbot_name}': {e}",
            exc_info=True
        )


def create_chatbot_service(
    db: Session,
    chatbot_name: str,
    status: str,
    description: Optional[str] = None,
    instruction: Optional[str] = None,
    doc_contents: Optional[List[bytes]] = None,
    doc_names: Optional[List[str]] = None,
    generated_by: Optional[uuid.UUID] = None,
    meta_data: Optional[Dict[str, Any]] = None,
    background_tasks: Optional[BackgroundTasks] = None
) -> Chatbot:
    """
    Create a chatbot and schedule background embedding task.
    
    Chatbot names must be unique.
    Supports multiple document formats: PDF, DOCX, DOC, TXT, RTF
    
    Args:
        db: Database session
        chatbot_name: Name of the chatbot (must be unique)
        status: Status of the chatbot
        description: Optional description
        instruction: Optional instructions
        doc_contents: List of document file contents (bytes)
        doc_names: List of document filenames
        generated_by: UUID of creator
        meta_data: Additional metadata
        background_tasks: FastAPI background tasks
        
    Returns:
        Created chatbot instance
        
    Raises:
        HTTPException: If chatbot name already exists
    """
    try:
        # Check if chatbot name already exists
        existing_chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_name == chatbot_name
        ).first()
        
        if existing_chatbot:
            raise HTTPException(
                status_code=400,
                detail=f"Chatbot with name '{chatbot_name}' already exists. Please use a unique name."
            )
        
        new_chatbot = Chatbot(
            chatbot_name=chatbot_name,
            description=description,
            instruction=instruction,
            pdf_names=doc_names or [],  # Store all document names
            status=status,
            generated_by=generated_by,
            meta_data=meta_data or {}
        )
        
        db.add(new_chatbot)
        db.commit()
        db.refresh(new_chatbot)
        
        log.info(
            f"Chatbot '{chatbot_name}' created with ID {new_chatbot.chatbot_id}"
        )
        
        # Schedule background embedding task if documents exist
        if doc_contents and doc_names and background_tasks:
            background_tasks.add_task(
                _process_documents_and_store_embeddings,
                chatbot_name=chatbot_name,
                doc_contents=doc_contents,
                doc_names=doc_names
            )
            log.info(
                f"Background task started for processing {len(doc_names)} "
                f"documents of '{chatbot_name}'"
            )
        
        return new_chatbot
        
    except HTTPException:
        # Re-raise HTTP exceptions (like duplicate name)
        raise
    except Exception as e:
        db.rollback()
        log.error(f"Failed to create chatbot '{chatbot_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create chatbot: {str(e)}"
        )


def search_similar_chunks(
    chatbot_name: str,
    query: str,
    top_k: int = 5,
    document_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks in Qdrant for a given query.
    
    Args:
        chatbot_name: Name of the chatbot (collection name)
        query: Search query text
        top_k: Number of results to return
        document_type: Optional filter by document type (pdf, docx, txt, etc.)
        
    Returns:
        List of similar chunks with scores and metadata
    """
    try:
        client = qdrant_manager.get_client()
        if not client:
            raise HTTPException(
                status_code=503,
                detail="Qdrant service unavailable"
            )
        
        collection_name = _sanitize_collection_name(chatbot_name)
        
        # Check if collection exists
        existing_collections = [col.name for col in client.get_collections().collections]
        if collection_name not in existing_collections:
            log.warning(f"Collection {collection_name} not found")
            return []
        
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        # Build filter if document_type specified
        query_filter = None
        if document_type:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value=document_type)
                    )
                ]
            )
        
        
        from qdrant_client.models import  VectorInput

        # New search using the current API
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,  # Directly pass the list[float] embedding vector
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter,
        )

        results = response.points
        
        # Format results
        chunks = []
        for result in results:
            chunks.append({
                'text': result.payload.get('text'),
                'source_file': result.payload.get('source_file'),
                'document_type': result.payload.get('document_type'),
                'chunk_index': result.payload.get('chunk_index'),
                'total_chunks': result.payload.get('total_chunks'),
                'score': float(result.score),
                'metadata': result.payload
            })
        
        log.info(
            f"Found {len(chunks)} similar chunks for query in chatbot '{chatbot_name}'"
        )
        
        return chunks
        
    except Exception as e:
        log.error(f"Error searching similar chunks: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error searching chunks: {str(e)}"
        )


def get_chatbot_context(
    chatbot_name: str,
    query: str, 
    max_context_length: int = 3000,
    document_type: Optional[str] = None,
    min_chunk_chars: int = 20  # Skip very short or meaningless chunks
) -> str:
    """
    Get relevant context for a query by searching similar chunks.
    Now robust: skips empty/None text chunks and logs sample content for debugging.
    """
    try:
        chunks = search_similar_chunks(
            chatbot_name=chatbot_name, 
            query=query, 
            top_k=10,
            document_type=document_type
        )
        
        if not chunks:
            log.info(f"No chunks retrieved for query in chatbot '{chatbot_name}'")
            return ""
        
        # Debug: Log first few chunks to see if text is present
        for i, chunk in enumerate(chunks[:3]):
            text_snippet = str(chunk.get('text', '') or '')[:100].replace('\n', ' ')
            log.info(
                f"Sample chunk {i} - source: {chunk.get('source_file')}, "
                f"text_len: {len(str(chunk.get('text', '') or ''))}, "
                f"score: {chunk.get('score'):.3f}, snippet: {text_snippet}"
            )
        
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            raw_text = chunk.get('text')
            text = str(raw_text).strip() if raw_text is not None else ""
            
            # Skip empty or too-short chunks
            if not text or len(text) < min_chunk_chars:
                continue
                
            source = chunk.get('source_file', 'Unknown')
            doc_type = chunk.get('document_type', 'unknown')
            
            chunk_text = f"[Source: {source} ({doc_type})]\n{text}\n"
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length > max_context_length:
                log.info(f"Context length limit reached ({total_length}/{max_context_length} chars)")
                break
            
            context_parts.append(chunk_text)
            total_length += chunk_length
        
        context = "\n---\n".join(context_parts)
        
        log.info(
            f"Built context of {total_length} chars from {len(context_parts)} valid chunks "
            f"for chatbot '{chatbot_name}'"
        )
        
        return context
        
    except Exception as e:
        log.error(f"Error building chatbot context: {e}", exc_info=True)
        return ""

def delete_chatbot_service(db: Session, chatbot_name: str) -> bool:
    """
    Delete a chatbot and its associated Qdrant collection.
    
    Args:
        db: Database session
        chatbot_name: Name of the chatbot to delete
        
    Returns:
        True if successfully deleted, False otherwise
        
    Raises:
        HTTPException: If chatbot not found or deletion fails
    """
    try:
        # Find chatbot
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_name == chatbot_name
        ).first()
        
        if not chatbot:
            raise HTTPException(
                status_code=404,
                detail=f"Chatbot '{chatbot_name}' not found"
            )
        
        # Delete from Qdrant
        client = qdrant_manager.get_client()
        if client:
            collection_name = _sanitize_collection_name(chatbot_name)
            existing_collections = [col.name for col in client.get_collections().collections]
            
            if collection_name in existing_collections:
                client.delete_collection(collection_name=collection_name)
                log.info(f"Deleted Qdrant collection: {collection_name}")
        
        # Delete from database
        db.delete(chatbot)
        db.commit()
        
        log.info(f"Chatbot '{chatbot_name}' deleted successfully")
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"Failed to delete chatbot '{chatbot_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete chatbot: {str(e)}"
        )
        
        
        
"""
Clean RAG (Retrieval-Augmented Generation) Service Function
Handles query processing, context retrieval, and response generation in one call.
"""



def rag_query_service(
    chatbot_name: str,
    query: str,
    chatbot_instructions: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None,
    top_k: int = 5,
    max_context_length: int = 3000,
    max_retries: int = 3
) -> Dict:
    """
    Full RAG query service with intelligent fallback for casual conversation.
    Uses best-practice prompt engineering for grounded, safe, and natural responses.
    """

    try:
        log.info(f"Retrieving context for chatbot '{chatbot_name}' | Query: {query[:80]}")

        context = get_chatbot_context(
            chatbot_name=chatbot_name,
            query=query,
            max_context_length=max_context_length
        )

        resolved_model = model_name or config.GEMINI_MODEL

        # ---------------------------------------------------------------------
        # CASUAL / NO-CONTEXT FALLBACK
        # ---------------------------------------------------------------------
        if not context:
            log.info("No relevant context found. Switching to conversational mode.")

            casual_system_prompt = """
            You are a friendly, polite, and helpful conversational assistant.

            Guidelines:
            - Respond naturally and warmly.
            - Keep responses concise and human-like.
            - If the user greets you, greet them back.
            - If the user asks a general question, answer to the best of your ability.
            - Do NOT mention documents, sources, databases, or training data.
            - Do NOT say you could not find context or documents.
            """

            casual_prompt = f"""
            User Message:
            {query}

            Assistant Response:
            """

            try:
                model = genai.GenerativeModel(resolved_model)
                response_obj = model.generate_content(
                    f"{casual_system_prompt}\n{casual_prompt}"
                )

                return {
                    "response": response_obj.text.strip(),
                    "sources": [],
                    "context_used": False,
                    "num_chunks_used": 0
                }

            except Exception as e:
                log.error(f"Casual response generation failed: {e}")
                return {
                    "response": "Hi! 😊 How can I help you today?",
                    "sources": [],
                    "context_used": False,
                    "num_chunks_used": 0
                }

        # ---------------------------------------------------------------------
        # RAG MODE (CONTEXT FOUND)
        # ---------------------------------------------------------------------
        system_prompt = chatbot_instructions or """
            You are a knowledgeable and reliable assistant.

            Core Principles:
            - Use ONLY the information provided in the context to answer.
            - If the answer is not clearly supported by the context, say so honestly.
            - Do NOT fabricate details or make assumptions.
            - Maintain a professional, clear, and concise tone.
            - Do NOT mention documents, sources, embeddings, or retrieval.
            - Answer as if you naturally know the information.
            """

        rag_prompt = f"""
            ==============================
            CONTEXT (AUTHORITATIVE)
            ==============================
            {context}

            ==============================
            USER QUESTION
            ==============================
            {query}

            ==============================
            INSTRUCTIONS
            ==============================
            - Provide a direct and accurate answer grounded in the context above.
            - If the context does not fully answer the question, clearly state the limitation.
            - Do NOT reference the existence of documents or context.
            - Do NOT include phrases like:
            "Based on the documents..."
            "According to the provided sources..."
            - Keep the response well-structured and easy to understand.

            ASSISTANT RESPONSE:
            """

        log.info(f"Generating RAG response using model: {resolved_model}")

        response_obj = _generate_with_retry(
            prompt=rag_prompt,
            conversation_history=conversation_history,
            model_name=resolved_model,
            max_retries=max_retries
        )

        # ---------------------------------------------------------------------
        # SOURCE METADATA (FOR CLIENT USE ONLY)
        # ---------------------------------------------------------------------
        chunks = search_similar_chunks(
            chatbot_name=chatbot_name,
            query=query,
            top_k=top_k
        )

        sources = [
            {
                "source_file": chunk["source_file"],
                "document_type": chunk["document_type"],
                "relevance_score": round(chunk["score"], 3),
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
            if chunk.get("text") and len(str(chunk["text"]).strip()) >= 20
        ]

        log.info("RAG query completed successfully")

        return {
            "response": response_obj.text.strip(),
            "sources": sources,
            "context_used": True,
            "num_chunks_used": len(sources)
        }

    except Exception as e:
        log.error("RAG query service failed", exc_info=True)
        return {
            "response": "Sorry, something went wrong while processing your request. Please try again.",
            "sources": [],
            "context_used": False,
            "error": str(e)
        }
def _generate_with_retry(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]],
    model_name: str,
    max_retries: int
):
    backoff = 2
    last_error = None
    import time
    for attempt in range(1, max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name)
            if conversation_history:
                chat = model.start_chat(history=[
                    {"role": msg["role"], "parts": [msg["content"]]} for msg in conversation_history
                ])
                return chat.send_message(prompt)
            return model.generate_content(prompt)
        except Exception as e:
            msg = str(e)
            last_error = e
            if ("429" in msg or "quota" in msg.lower() or "rate" in msg.lower()) and attempt < max_retries:
                log.warning(f"Rate limit hit (attempt {attempt}); retrying after {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            break
    raise last_error or RuntimeError("Gemini generation failed after retries")