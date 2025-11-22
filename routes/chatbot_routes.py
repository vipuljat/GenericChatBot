from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import config
from sqlalchemy.orm import Session
from core.database import get_db
from services.chatbot_service import create_chatbot_with_documents
from services.document_service import render_quiz_questions
from services.rag_service import get_rag_service
from models.chatbot import Chatbot
from pydantic import BaseModel
from typing import List, Optional, Dict, Union

router = APIRouter()

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    context_used: bool
    num_chunks_used: Optional[int] = None

@router.post("/chatbots")
def create_chatbot_route(
    chatbot_name: str = Form(...),
    description: str = Form(...),
    instructions: str = Form(...),
    conversation_starters: Union[str, List[str], None] = Form(default=None),
    is_quiz_mode: bool = Form(False),
    is_active: bool = Form(True),
    recommended_model: Optional[str] = Form(None),
    file: UploadFile = File(...),
    quiz_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Create a new chatbot with documents.
    Documents will be automatically processed and embeddings stored in Qdrant.
    """
    # Handle conversation_starters - normalize to list or None
    if conversation_starters is not None:
        if isinstance(conversation_starters, str):
            conversation_starters = [conversation_starters]
        elif not isinstance(conversation_starters, list):
            conversation_starters = None
    
    chatbot_id = create_chatbot_with_documents(
        db=db,
        chatbot_name=chatbot_name,
        description=description,
        instructions=instructions,
        conversation_starters=conversation_starters,
        is_quiz_mode=is_quiz_mode,
        is_active=is_active,
        recommended_model=recommended_model,
        file=file,
        quiz_file=quiz_file,
    )

    return {
        "message": "Chatbot created successfully",
        "chatbot_id": chatbot_id
    }

@router.post("/chatbots/{chatbot_id}/chat", response_model=ChatResponse)
def chat_with_chatbot(
    chatbot_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Chat with a specific chatbot using RAG (Retrieval-Augmented Generation).
    The chatbot will search its documents and generate responses based on relevant context.
    
    Example:
    - POST /chatbot/chatbots/1/chat
    - Body: {"query": "How many leaves am I entitled to?"}
    """
    # Verify chatbot exists
    chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail=f"Chatbot {chatbot_id} not found")
    
    if not bool(getattr(chatbot, 'is_active', False)):
        raise HTTPException(status_code=400, detail="Chatbot is not active")
    
    # Get RAG service
    rag_service = get_rag_service()
    
    # Generate response using RAG
    result = rag_service.generate_response(
        query=request.query,
        chatbot_id=chatbot_id,
        chatbot_instructions=str(getattr(chatbot, 'instructions', '') or ''),
        conversation_history=request.conversation_history,
        model_name=str(getattr(chatbot, 'recommended_model', '') or '') or config.GEMINI_MODEL
    )
    
    return ChatResponse(**result)

@router.get("/chatbots/{chatbot_id}")
def get_chatbot(chatbot_id: int, db: Session = Depends(get_db)):
    """Get chatbot details by ID"""
    chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail=f"Chatbot {chatbot_id} not found")
    
    return {
        "id": chatbot.id,
        "chatbot_name": chatbot.chatbot_name,
        "description": chatbot.description,
        "instructions": chatbot.instructions,
        "conversation_starters": chatbot.conversation_starters,
        "is_quiz_mode": chatbot.is_quiz_mode,
        "is_active": chatbot.is_active,
        "recommended_model": chatbot.recommended_model
    }

@router.post("/quiz/render")
async def upload_and_render_quiz_document(file: UploadFile = File(...)):
    """
    Upload and render quiz questions from a document.
    """
    try:
        rendered_questions = await render_quiz_questions(file)
        return {
            "file_name": file.filename,
            "total_questions": len(rendered_questions),
            "questions": rendered_questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
