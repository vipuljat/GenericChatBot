"""Chatbot service with multi-format document processing and chunked embeddings."""

import time
import uuid
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import config
from sqlalchemy.orm import Session
from fastapi import HTTPException, BackgroundTasks
from qdrant_client.models import Distance, VectorParams, PointStruct
from stateful_services.db_schema import Chatbot, Question
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
    mode: Optional[str] = "general",
    questions: Optional[dict] = None,   
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
            mode=mode,
            generated_by=generated_by,
            meta_data=meta_data or {}
        )
        
        db.add(new_chatbot)
        db.commit()
        db.refresh(new_chatbot)
        
        log.info(
            f"Chatbot '{chatbot_name}' created with ID {new_chatbot.chatbot_id}"
        )

        if questions and mode == "quiz":
            existing_questions= db.query(Question).filter(Question.chatbot_id == new_chatbot.chatbot_id).first()
            if not existing_questions:
                new_questions = Question(
                    chatbot_id=new_chatbot.chatbot_id,
                    question_data=questions
                )
                db.add(new_questions)
                db.commit()
                db.refresh(new_questions)
            else:
                existing_questions.question_data = questions
                db.commit()
                db.refresh(existing_questions)    
                
                   
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
    document_type: Optional[str] = None,
    min_score: float = 0.5  # New: minimum similarity score threshold
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks in Qdrant for a given query.
    
    Args:
        chatbot_name: Name of the chatbot (collection name)
        query: Search query text
        top_k: Number of results to return
        document_type: Optional filter by document type (pdf, docx, txt, etc.)
        min_score: Minimum similarity score to consider a chunk relevant
        
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
        
        # Format results, filtering by min_score
        chunks = []
        for result in results:
            if float(result.score) >= min_score:
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
            f"Found {len(chunks)} similar chunks (after min_score filter) for query in chatbot '{chatbot_name}'"
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
    min_chunk_chars: int = 20,  # Skip very short or meaningless chunks
    min_score: float = 0.5  # New: pass min_score to search
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
            document_type=document_type,
            min_score=min_score  # New: pass min_score
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
    top_k: int = 10,
    max_context_length: int = 4000,
    max_retries: int = 3,
    min_relevance_score: float = 0.55  # Increased threshold for more reliable relevance
) -> Dict:
    """
    Enhanced RAG service that behaves like a natural, consistent chatbot.
    Handles greetings, casual chat, and document-based questions seamlessly.
    """
    try:
        log.info(f"Processing query for chatbot '{chatbot_name}': {query[:60]}...")

        # Retrieve potentially relevant context
        context = get_chatbot_context(
            chatbot_name=chatbot_name,
            query=query,
            max_context_length=max_context_length,
            min_score=min_relevance_score  # Only include truly relevant chunks
        )

        resolved_model = model_name or config.GEMINI_MODEL

        # Base personality/instructions
        base_instructions = chatbot_instructions or (
            "You are a friendly, knowledgeable, and helpful assistant. "
            "You answer clearly, concisely, and naturally. "
            "You remember the conversation and respond in a consistent tone."
        )

        # Unified prompt – works whether context is present or not
        prompt = f"""Relevant Document Context (use ONLY if directly relevant to the user's question):
{context if context else "(No relevant document information found)"}

Conversation Instructions:
{base_instructions}

Important Rules:
- Answer naturally, as if the knowledge is your own. Never say "based on the documents", "according to the context", or mention sources unless the user explicitly asks for them.
- If the user's message is a greeting (hi, hello, how are you, etc.), casual chat, or off-topic, respond warmly and conversationally. Ignore the document context in these cases.
- If the question is clearly related to the documents and the context above helps, use it to give an accurate answer.
- Keep responses engaging, friendly, and appropriate to the conversation flow.
- Do not hallucinate information not supported by the context when answering document-related questions.

User's message: {query}

Respond directly and naturally."""

        log.info(f"Generating response with {resolved_model} (context length: {len(context) if context else 0})")

        response_obj = _generate_with_retry(
            prompt=prompt,
            conversation_history=conversation_history,
            model_name=resolved_model,
            max_retries=max_retries
        )

        # Only return sources if we actually used meaningful context
        has_meaningful_context = bool(context and context.strip() and context != "(No relevant document information found)")
        
        if has_meaningful_context:
            chunks = search_similar_chunks(
                chatbot_name=chatbot_name,
                query=query,
                top_k=top_k,
                min_score=min_relevance_score
            )
            sources = [
                {
                    "source_file": chunk['source_file'],
                    "document_type": chunk['document_type'],
                    "relevance_score": round(chunk['score'], 3),
                    "chunk_index": chunk['chunk_index']
                }
                for chunk in chunks
                if chunk.get('text') and len(str(chunk['text']).strip()) >= 20
            ]
        else:
            sources = []

        return {
            "response": response_obj.text.strip(),
            "sources": sources,
            "context_used": has_meaningful_context,
            "num_chunks_used": len(sources)
        }

    except Exception as e:
        log.error(f"RAG query failed: {str(e)}", exc_info=True)
        return {
            "response": "Sorry, I'm having trouble right now. Please try again in a moment!",
            "sources": [],
            "context_used": False,
            "error": str(e)
        }
        
def _map_role_for_gemini(role_name: str) -> str:
    """Maps internal or non-standard roles to the Gemini API's required roles ('user', 'model')."""
    role_name_lower = role_name.lower()

    if role_name_lower == 'user':
        return 'user'

    if role_name_lower in ['model', 'assistant','bot', 'ai', 'system']:
        return 'model'
    return 'user'

# --- Main Generation Function ---
def _generate_with_retry(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]],
    model_name: str,
    max_retries: int
):
    """
    Generates content using the Gemini API, implementing retry logic 
    for rate limit errors.
    """
    backoff = 2
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name)
            
            if conversation_history:
                # 1. Map and format the history list for the Gemini API
                history_for_gemini = [
                    {
                        # Use the helper function to ensure valid roles
                        "role": _map_role_for_gemini(msg["role"]), 
                        # Use "text" key inside parts, as required by the API's Content object format
                        "parts": [{"text": msg["content"]}] 
                    } 
                    for msg in conversation_history
                ]
                
                # 2. Start chat with the corrected history
                chat = model.start_chat(history=history_for_gemini)
                return chat.send_message(prompt)
            
            # If no conversation history, use generate_content directly
            return model.generate_content(prompt)
            
        except Exception as e:
            msg = str(e)
            last_error = e
            
            # Rate Limit Retry Logic
            if ("429" in msg or "quota" in msg.lower() or "rate" in msg.lower()) and attempt < max_retries:
                log.warning(f"Rate limit hit (attempt {attempt}); retrying after {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            
            # Break if it's a permanent error (like 400 InvalidArgument) or max retries reached
            break
            
    # Raise the last error encountered (which was likely the 400 InvalidArgument error)
    raise last_error or RuntimeError("Gemini generation failed after retries")