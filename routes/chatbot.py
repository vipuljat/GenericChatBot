"""Chatbot API routes with multi-format document upload support."""

from typing import Any, Dict, Optional, List
import uuid
import json

from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.params import Body
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.quiz_services import quiz_query_service
from stateful_services.database import get_db
from stateful_services.db_schema import Chatbot, Question
from services.chatbots_services import (
    create_chatbot_service,
    delete_chatbot_service,
    rag_query_service,
    search_similar_chunks,
    get_chatbot_context,
    update_chatbot_service
)
from services.document_service import render_quiz_questions
from utils.document_service import is_supported_document, get_supported_extensions
from utils.logging import log

router = APIRouter()


@router.post("/create", summary="Create a chatbot with multiple documents")
async def create_chatbot(
    chatbot_name: str = Form(..., description="Unique name for the chatbot"),
    status: str = Form(..., description="Status (active/inactive)"),
    description: Optional[str] = Form(None, description="Description of the chatbot"),
    instruction: Optional[str] = Form(None, description="Instructions for the chatbot"),
    generated_by: Optional[str] = Form(None, description="UUID of creator"),
    meta_data: Optional[str] = Form(None, description="JSON metadata"),
    documents: List[UploadFile] = File(..., description="Documents (PDF, DOCX, DOC, TXT, RTF)"),
    mode: Optional[str] = Form("general", description="Chatbot mode (e.g., general, quiz, hybrid)"),
    questions: Optional[dict] = Form(None, description="Quiz questions if in quiz mode"),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a chatbot with multiple documents.
    
    **Supported Formats:**
    - PDF (.pdf)
    - Word Documents (.docx, .doc)
    - Text Files (.txt)
    - Rich Text Format (.rtf)
    
    **Note:** Chatbot names must be unique.
    """
    
    log.info(f"Received request to create chatbot: {chatbot_name}")
    
    doc_contents: List[bytes] = []
    doc_names: List[str] = []
    
    # Parse metadata
    meta_data_dict = {}
    if meta_data:
        try:
            meta_data_dict = json.loads(meta_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format in meta_data")
    
    # Process all documents
    for doc in documents:
        if not is_supported_document(doc.filename):
            supported = ", ".join(f".{ext}" for ext in get_supported_extensions())
            raise HTTPException(
                status_code=400,
                detail=f"File {doc.filename} has unsupported format. Supported: {supported}"
            )
        
        try:
            content = await doc.read()
            
            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {doc.filename} is empty"
                )
            
            doc_contents.append(content)
            doc_names.append(doc.filename)
            
            log.info(f"Processed document: {doc.filename} ({len(content)} bytes)")
            
        except Exception as e:
            log.error(f"Error reading document {doc.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing {doc.filename}: {str(e)}"
            )
    
    # Parse generated_by UUID
    generated_by_uuid = None
    if generated_by:
        try:
            generated_by_uuid = uuid.UUID(generated_by)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format for generated_by")
    
    # Create chatbot
    try:
        chatbot = create_chatbot_service(
            db=db,
            chatbot_name=chatbot_name,
            status=status,
            description=description,
            instruction=instruction,
            doc_contents=doc_contents,
            doc_names=doc_names,
            generated_by=generated_by_uuid,
            meta_data=meta_data_dict,
            mode=mode,
            questions=questions,
            background_tasks=background_tasks
        )
        
        log.info(
            f"Chatbot created successfully with {len(doc_names)} documents: "
            f"{chatbot.chatbot_name} (ID: {chatbot.chatbot_id})"
        )
        
       
        return {
            "chatbot_id": str(chatbot.chatbot_id),
            "chatbot_name": chatbot.chatbot_name,
            "status": chatbot.status,
            "description": chatbot.description,
            "document_count": len(doc_names),
            "document_names": doc_names,
            "supported_formats": get_supported_extensions(),
            "message": f"Chatbot created successfully with {len(doc_names)} documents. Processing started in background."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error creating chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating chatbot: {str(e)}")

@router.get("/list", summary="List all chatbots")
async def list_chatbots(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """
    Get a list of all chatbots with pagination.
    
    **Parameters:**
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **status**: Optional filter by status (active/inactive)
    
    **Returns:**
    - List of chatbots with their details
    """
    try:
        query = db.query(Chatbot)
        
        # Apply status filter if provided
        if status:
            query = query.filter(Chatbot.status == status)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        chatbots = query.offset(skip).limit(limit).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "count": len(chatbots),
            "chatbots": [
                {
                    "chatbot_id": str(chatbot.chatbot_id),
                    "chatbot_name": chatbot.chatbot_name,
                    "status": chatbot.status,
                    "description": chatbot.description,
                    "instruction": chatbot.instruction,
                    "document_count": len(chatbot.pdf_names) if chatbot.pdf_names else 0,
                    "document_names": chatbot.pdf_names,
                    "created_at": chatbot.created_at.isoformat() if hasattr(chatbot, 'created_at') else None,
                    "meta_data": chatbot.meta_data
                }
                for chatbot in chatbots
            ]
        }
        
    except Exception as e:
        log.error(f"Error listing chatbots: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing chatbots: {str(e)}")


@router.get("/{chatbot_id}", summary="Get chatbot details")
async def get_chatbot(
    chatbot_id: str,
    db: Session = Depends(get_db)
):
    """
    Get details of a specific chatbot by name.
    
    **Parameters:**
    - **chatbot_name**: Name of the chatbot
    
    **Returns:**
    - Chatbot details including documents
    """
    try:
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_id == chatbot_id
        ).first()
        
        if not chatbot:
            raise HTTPException(
                status_code=404,
                detail=f"Chatbot '{chatbot_id}' not found"
            )
        
        questions_data = None

        # Always fetch questions if they exist (dual-mode support)
        # Frontend can toggle between chatbot mode and quiz mode
        log.info(f"Fetching chatbot: {chatbot.chatbot_name} (mode: {chatbot.mode})")
        questions_list = db.query(Question).filter(
            Question.chatbot_id == chatbot.chatbot_id
        ).all()
        
        if not questions_list:
            questions_data = []
        else:
            # Format all questions with their options for frontend display
            questions_data = []
            for q in questions_list:
                q_data = q.question_data
                if isinstance(q_data, dict):
                    # If question_data contains a "questions" array (from quiz file upload)
                    if "questions" in q_data:
                        questions_array = q_data["questions"]
                        if isinstance(questions_array, list):
                            questions_data.extend(questions_array)
                    else:
                        # Individual question format
                        question_info = {
                            "question_id": str(q.question_id),
                            "status": q.status,
                            "created_at": q.created_at.isoformat() if q.created_at else None
                        }
                        
                        # Extract question details including options
                        question_info.update({
                            "question_text": q_data.get("question"),
                            "type": q_data.get("type"),
                            "options": q_data.get("options"),  # Options array for frontend
                            "difficulty": q_data.get("difficulty"),
                            "points": q_data.get("points"),
                            "metadata": q_data.get("metadata", {})
                        })
                        
                        questions_data.append(question_info)
        
        
        return {
            "chatbot_id": str(chatbot.chatbot_id),
            "chatbot_name": chatbot.chatbot_name,
            "status": chatbot.status,
            "description": chatbot.description,
            "instruction": chatbot.instruction,
            "document_count": len(chatbot.pdf_names) if chatbot.pdf_names else 0,
            "document_names": chatbot.pdf_names,
            "mode": chatbot.mode,
            "questions": questions_data,
            "total_questions": len(questions_data) if questions_data else 0,
            "has_quiz_capability": len(questions_data) > 0 if questions_data else False,
            "generated_by": str(chatbot.generated_by) if chatbot.generated_by else None,
            "created_at": chatbot.created_at.isoformat() if hasattr(chatbot, 'created_at') else None,
            "updated_at": chatbot.updated_at.isoformat() if hasattr(chatbot, 'updated_at') else None,
            "meta_data": chatbot.meta_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chatbot: {str(e)}")


@router.post("/{chatbot_name}/query", summary="Query chatbot with context")
async def query_chatbot(
    chatbot_name: str,
    query: str = Form(..., description="User query"),
    top_k: int = Form(5, ge=1, le=20, description="Number of similar chunks to retrieve"),
    document_type: Optional[str] = Form(None, description="Filter by document type (pdf, docx, txt, etc.)"),
    max_context_length: int = Form(3000, ge=500, le=10000, description="Maximum context length"),
    include_context: bool = Form(True, description="Include context in response")
):
    """
    Query a chatbot and get relevant context from its documents.
    
    **Parameters:**
    - **chatbot_name**: Name of the chatbot
    - **query**: The user's question or query
    - **top_k**: Number of similar chunks to retrieve (1-20)
    - **document_type**: Optional filter by document type
    - **max_context_length**: Maximum characters of context (500-10000)
    - **include_context**: Whether to include the full context in response
    
    **Returns:**
    - Query results with relevant context and sources
    """
    try:
        log.info(f"Querying chatbot '{chatbot_name}' with: {query[:100]}...")
        
        # Search for similar chunks
        chunks = search_similar_chunks(
            chatbot_name=chatbot_name,
            query=query,
            top_k=top_k,
            document_type=document_type
        )
        
        if not chunks:
            return {
                "chatbot_name": chatbot_name,
                "query": query,
                "context": "",
                "chunks": [],
                "message": "No relevant context found for this query."
            }
        
        # Get formatted context if requested
        context = ""
        if include_context:
            context = get_chatbot_context(
                chatbot_name=chatbot_name,
                query=query,
                max_context_length=max_context_length,
                document_type=document_type
            )
        
        return {
            "chatbot_name": chatbot_name,
            "query": query,
            "context": context,
            "chunks_found": len(chunks),
            "chunks": [
                {
                    "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "full_text": chunk['text'] if include_context else None,
                    "source_file": chunk['source_file'],
                    "document_type": chunk['document_type'],
                    "chunk_index": chunk['chunk_index'],
                    "score": round(chunk['score'], 4)
                }
                for chunk in chunks
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error querying chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying chatbot: {str(e)}")


@router.delete("/{chatbot_name}", summary="Delete a chatbot")
async def delete_chatbot(
    chatbot_name: str,
    db: Session = Depends(get_db)
):
    """
    Delete a chatbot and its associated documents/embeddings.
    
    **Warning:** This action cannot be undone. All document embeddings will be removed from Qdrant.
    
    **Parameters:**
    - **chatbot_name**: Name of the chatbot to delete
    
    **Returns:**
    - Confirmation message
    """
    try:
        log.info(f"Deleting chatbot: {chatbot_name}")
        
        success = delete_chatbot_service(db, chatbot_name)
        
        if success:
            return {
                "message": f"Chatbot '{chatbot_name}' deleted successfully",
                "chatbot_name": chatbot_name,
                "deleted": True
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete chatbot '{chatbot_name}'"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error deleting chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chatbot: {str(e)}")


@router.get("/supported-formats", summary="Get list of supported document formats")
async def get_supported_formats():
    """
    Get list of supported document formats for upload.
    
    **Returns:**
    - List of supported file extensions and their descriptions
    """
    return {
        "formats": [
            {"extension": "pdf", "description": "Portable Document Format"},
            {"extension": "docx", "description": "Microsoft Word (2007+)"},
            {"extension": "doc", "description": "Microsoft Word (Legacy)"},
            {"extension": "txt", "description": "Plain Text"},
            {"extension": "rtf", "description": "Rich Text Format"}
        ],
        "extensions": get_supported_extensions()
    }
    
class QueryRequest(BaseModel):
    query: str
    instructions: Optional[str] = None
    history: Optional[list[dict]] = None  # if you plan to support conversation history later
    user_id: Optional[str] = None  # For quiz mode to track attempter

# @router.post("/user/{chatbot_id}/query")
# async def user_query_chatbot(
#     chatbot_id: str,
#     request: QueryRequest = Body(...),
#     db: Session = Depends(get_db)
# ):

#     # Fetch chatbot to validate existence and get default instructions
#     chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
#     if not chatbot:
#         raise HTTPException(status_code=404, detail=f"Chatbot '{chatbot_id}' not found")
#     chatbot_name = chatbot.chatbot_name
#     # Use provided instructions or fallback to DB
#     final_instructions = request.instructions or chatbot.instruction

#     result = rag_query_service(
#         chatbot_name=chatbot_name,
#         query=request.query,
#         chatbot_instructions=final_instructions,
#         conversation_history=request.history or []
#     )

#     return result


@router.post("/user/{chatbot_id}/query")
async def user_query_chatbot(
    chatbot_id: str,
    request: QueryRequest = Body(...),
    db: Session = Depends(get_db)
):
    """
    Universal chatbot query endpoint.
    Handles both general RAG mode and quiz mode based on chatbot configuration.
    
    Args:
        chatbot_id: UUID of the chatbot
        request: Query request with message, optional instructions, and history
        db: Database session
        
    Returns:
        Response based on chatbot mode:
        - General mode: RAG response with sources
        - Quiz mode: Interactive quiz response with state management
    """
    try:
        # Fetch chatbot to validate existence and get configuration
        chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(
                status_code=404, 
                detail=f"Chatbot '{chatbot_id}' not found"
            )
        
        chatbot_name = chatbot.chatbot_name
        chatbot_mode = chatbot.mode or "general"
        
        log.info(f"Processing query for chatbot '{chatbot_name}' in '{chatbot_mode}' mode")
        
        # Route to appropriate service based on mode
        if chatbot_mode == "quiz":
            # Handle quiz mode
            user_id = None
            if request.user_id:
                try:
                    user_id = uuid.UUID(request.user_id)
                except ValueError:
                    log.warning(f"Invalid user_id format: {request.user_id}")
            
            result = quiz_query_service(
                db=db,
                chatbot_id=chatbot_id,
                chatbot_name=chatbot_name,
                query=request.query,
                conversation_history=request.history or [],
                user_id=user_id
            )
            
            return {
                "mode": "quiz",
                "response": result["response"],
                "quiz_state": result.get("quiz_state", {}),
                "metadata": result.get("metadata", {}),
                "error": result.get("error")
            }
        
        else:
            # Handle general RAG mode (default)
            final_instructions = request.instructions or chatbot.instruction
            
            result = rag_query_service(
                chatbot_name=chatbot_name,
                query=request.query,
                chatbot_instructions=final_instructions,
                conversation_history=request.history or []
            )
            
            return {
                "mode": "general",
                "response": result["response"],
                "sources": result.get("sources", []),
                "context_used": result.get("context_used", False),
                "num_chunks_used": result.get("num_chunks_used", 0),
                "error": result.get("error")
            }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Chatbot query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
