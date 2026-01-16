from typing import List, Optional, Dict, Any
import uuid
import json
from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, BackgroundTasks, HTTPException, Query, Body
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agent.agent import render_quiz_questions
from stateful_services.database import get_db
from stateful_services.db_schema import Chatbot, ChatbotAccess, PeopleAnalyzer, Question, Answer, Employee, ChatbotPermission
from utils.document_service import is_supported_document, get_supported_extensions
from utils.logging import log
import os
import time
import httpx
from fastapi import HTTPException, Query
from services.chatbots_services import sync_kestrel_employees_service

# Import functional services
from services.chatbots_services import (
    create_chatbot,
    get_chatbot_responses_service,
    get_employee_evaluation_service,
    update_chatbot,
    delete_chatbot_by_id
)
from services.rag_service import generate_rag_response
from services.quiz_services import quiz_query_service
from utils.token_decode import get_current_employee_from_token

router = APIRouter()
KESTREL_BASE_URL = os.getenv("KESTREL_BASE_URL")
KESTREL_API_KEY = os.getenv("KESTREL_API_KEY")

# ============================================================================
# REQUEST MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    instructions: Optional[str] = None
    history: Optional[List[dict]] = None
    user_id: Optional[str] = None
    employee_id: Optional[str] = None


class ChatbotUpdate(BaseModel):
    status: Optional[str] = None
    description: Optional[str] = None
    instruction: Optional[str] = None
    mode: Optional[str] = None
    questions: Optional[List[Dict[str, Any]]] = None
    employee_ids: Optional[List[str]]= None
    access_list: Optional[List[Dict]]= None
    


class SubmitQuizRequest(BaseModel):
    chatbot_id: str
    answers: Dict[str, Any] = {}
    employee_id: Optional[str] = None
    user_id: Optional[str] = None




def generate_greeting_response(chatbot_name: str = "Assistant") -> str:
    """Generate a friendly greeting response."""
    return f"Hello! I'm {chatbot_name}. How can I assist you today?"


# ============================================================================
# CHATBOT CRUD ENDPOINTS
# ============================================================================

@router.post("/create", summary="Create chatbot with documents")
async def create_chatbot_endpoint(
    request: Request,
    chatbot_name: str = Form(...),
    status: str = Form(...),
    description: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    generated_by: Optional[str] = Form(None),
    meta_data: Optional[str] = Form(None),
    documents: List[UploadFile] = File(default=[]), 
    mode: Optional[str] = Form("general"),
    questions: Optional[str] = Form(None),
    employee_ids: Optional[str] = Form(None),
    access_list: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create chatbot with document upload."""
    
    # Validate and process documents
    doc_contents = []
    doc_names = []
    
    for doc in documents:
        if not is_supported_document(doc.filename):
            supported = ", ".join(f".{ext}" for ext in get_supported_extensions())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported: {doc.filename}. Supported: {supported}"
            )
        
        content = await doc.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Empty document: {doc.filename}"
            )
        
        doc_contents.append(content)
        doc_names.append(doc.filename)
    
    # Parse inputs
    meta_dict = json.loads(meta_data) if meta_data else {}
    parsed_questions = json.loads(questions) if questions else None
    generated_by_uuid = uuid.UUID(generated_by) if generated_by else None
    parsed_employee_ids = json.loads(employee_ids) if employee_ids else []
    parsed_access_list = json.loads(access_list) if access_list else []

    # Create chatbot (functional call)
    chatbot = create_chatbot(
        request=request,
        db=db,
        chatbot_name=chatbot_name,
        status=status,
        description=description,
        instruction=instruction,
        doc_contents=doc_contents,
        doc_names=doc_names,
        generated_by=generated_by_uuid,
        meta_data=meta_dict,
        employee_ids=parsed_employee_ids,
        access_list=parsed_access_list,
        mode=mode,
        questions=parsed_questions,
        background_tasks=background_tasks
    )
    

   
    return {
        "chatbot_id": str(chatbot.chatbot_id),
        "chatbot_name": chatbot.chatbot_name,
        "status": chatbot.status,
        "document_count": len(doc_names),
        "permissions_created": len(parsed_employee_ids) if parsed_employee_ids else 0,
        "message": f"Chatbot created. Processing {len(doc_names)} documents in background."
    }


@router.patch("/{chatbot_id}", summary="Update chatbot")
async def update_chatbot_endpoint(
    chatbot_id: str,
    status: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    questions: Optional[str] = Form(None),  # JSON string
    employee_ids: Optional[str] = Form(None),  # JSON string
    access_list: Optional[str] = Form(None),  # JSON string
    deleted_documents: Optional[str] = Form(None),  # JSON string - list of document names to delete
    documents: List[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update chatbot settings and optionally upload new documents."""
    
    import json
    
    # Parse JSON strings
    parsed_questions = json.loads(questions) if questions else None
    parsed_employee_ids = json.loads(employee_ids) if employee_ids else None
    parsed_access_list = json.loads(access_list) if access_list else None
    parsed_deleted_documents = json.loads(deleted_documents) if deleted_documents else None
    
    # Handle document uploads
    doc_contents = []
    doc_names = []
    
    if documents:
        for doc in documents:
            content = await doc.read()
            doc_contents.append(content)
            doc_names.append(doc.filename)

    chatbot = update_chatbot(
        db=db,
        chatbot_id=uuid.UUID(chatbot_id),
        status=status,
        description=description,
        instruction=instruction,
        mode=mode,
        questions=parsed_questions,
        employee_ids=parsed_employee_ids,
        access_list=parsed_access_list,
        deleted_documents=parsed_deleted_documents,
        doc_contents=doc_contents if doc_contents else None,
        doc_names=doc_names if doc_names else None,
        background_tasks=background_tasks
    )
    
    return {
        "chatbot_id": str(chatbot.chatbot_id),
        "status": chatbot.status,
        "message": "Updated successfully"
    }


@router.delete("/{chatbot_id}", summary="Delete chatbot")
async def delete_chatbot_endpoint(
    chatbot_id: str,
    db: Session = Depends(get_db)
):
    """Delete chatbot and embeddings."""
    
    success = delete_chatbot_by_id(db, uuid.UUID(chatbot_id))
    
    if success:
        return {
            "message": "Chatbot deleted successfully",
            "chatbot_id": chatbot_id,
            "deleted": True
        }
    
    raise HTTPException(status_code=500, detail="Delete failed")


@router.get("/list", summary="List chatbots")
async def list_chatbots(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    user_info = get_current_employee_from_token(request, db)

    role = user_info.employee_role
    user_id = str(user_info.id)

    query = db.query(Chatbot)

    # 🔐 Non-admins: restrict by ChatbotAccess.allowed_users
    if role != "admin":
        allowed_chatbot_ids_subquery = (
            db.query(ChatbotAccess.chatbot_id)
            .filter(ChatbotAccess.allowed_users.contains([user_id]))
            .subquery()
        )

        query = query.filter(
            Chatbot.chatbot_id.in_(allowed_chatbot_ids_subquery)
        )

    # Optional status filter
    if status:
        query = query.filter(Chatbot.status == status)

    total = query.count()
    chatbots = query.offset(skip).limit(limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "count": len(chatbots),
        "chatbots": [
            {
                "chatbot_id": str(c.chatbot_id),
                "chatbot_name": c.chatbot_name,
                "status": c.status,
                "description": c.description,
                "mode": c.mode,
                "document_count": len(c.pdf_names) if c.pdf_names else 0,
                "document_names": c.pdf_names or [],
            }
            for c in chatbots
        ],
    }



@router.get("/{chatbot_id}", summary="Get chatbot details")
async def get_chatbot_endpoint(
    chatbot_id: str,
    db: Session = Depends(get_db)
):
    """Get chatbot by ID."""
    
    chatbot = db.query(Chatbot).filter(
        Chatbot.chatbot_id == chatbot_id
    ).first()
    
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Initialize common variables
    questions_data = None
    access_list = []
    employee_ids = []
    
    # Get questions if quiz mode
    if chatbot.mode in ("quiz", "people_analyzer"):
        questions = db.query(Question).filter(
            Question.chatbot_id == chatbot.chatbot_id
        ).first()
        questions_data = questions.question_data if questions else []
        
        # Get permissions
        permissions = db.query(ChatbotPermission).filter(
            ChatbotPermission.chatbot_id == chatbot.chatbot_id
        ).all()
        for perm in permissions:
            access_list.append({
                "reviewer_id": str(perm.reviewer_id),
                "allowed_users": perm.can_review_users or []
            })
    
    # Get allowed users for ALL modes
    access = db.query(ChatbotAccess).filter(
        ChatbotAccess.chatbot_id == chatbot_id
    ).first()
    employee_ids = access.allowed_users if access else []
    
    return {
        "chatbot_id": str(chatbot.chatbot_id),
        "chatbot_name": chatbot.chatbot_name,
        "status": chatbot.status,
        "description": chatbot.description,
        "instruction": chatbot.instruction,
        "mode": chatbot.mode,
        "document_count": len(chatbot.pdf_names) if chatbot.pdf_names else 0,
        "document_names": chatbot.pdf_names or [],
        "questions": questions_data,
        "meta_data": chatbot.meta_data,
        "access_list": access_list,
        "employee_ids": employee_ids
    }


# ============================================================================
# UNIFIED QUERY ENDPOINT
# ============================================================================

@router.post("/user/{chatbot_id}/query", summary="Query chatbot (unified)")
async def query_chatbot_endpoint(
    chatbot_id: str,
    request: QueryRequest = Body(...),
    db: Session = Depends(get_db)
):
    """
    Universal query endpoint.
    Routes to RAG, Quiz, or People Analyzer based on chatbot mode.
    ALL COSTS LOGGED AUTOMATICALLY.
    
    Modes:
    - general: RAG-based question answering
    - quiz: Interactive quiz assessments
    - people_analyzer: People analysis with +/- ratings
    """
    
    # Validate chatbot
    chatbot = db.query(Chatbot).filter(
        Chatbot.chatbot_id == chatbot_id
    ).first()
    
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    chatbot_mode = str(chatbot.mode).lower() if chatbot.mode else "general"
    
    log.info(f"🔍 Query for '{chatbot.chatbot_name}' ({chatbot_mode} mode)")
    
    # Route based on mode
    if chatbot_mode in ["quiz", "people_analyzer"]:
        # Quiz or People Analyzer mode (use same service)
        user_id = None
        employee_id = None
        
        if request.user_id:
            try:
                user_id = uuid.UUID(request.user_id)
            except ValueError:
                pass
        
        if request.employee_id:
            try:
                employee_id = uuid.UUID(request.employee_id)
            except ValueError:
                pass
        
        result = quiz_query_service(
            db=db,
            chatbot_id=chatbot_id,
            chatbot_name=chatbot.chatbot_name,
            chatbot_mode=chatbot_mode,
            query=request.query,
            conversation_history=request.history or [],
            user_id=user_id,
            employee_id=employee_id
        )
        
        return {
            "mode": chatbot_mode,
            "response": result["response"],
            "quiz_state": result.get("quiz_state", {}),
            "metadata": result.get("metadata", {})
        }

    else:
        # General RAG mode (costs logged inside)
        result = generate_rag_response(
            query=request.query,
            chatbot_name=chatbot.chatbot_name,
            chatbot_instructions=request.instructions or chatbot.instruction,
            conversation_history=request.history or []
        )
        
        return {
            "mode": "general",
            "response": result["response"],
            "sources": result.get("sources", []),
            "context_used": result.get("context_used", False),
            "num_chunks_used": result.get("num_chunks_used", 0)
        }



# ============================================================================
# QUIZ SUBMISSION
# ============================================================================

@router.post("/submit-quiz", summary="Submit quiz answers")
async def submit_quiz_endpoint(
    payload: SubmitQuizRequest,
    db: Session = Depends(get_db)
):
    """Submit quiz answers (normal quizzes only)."""
    
    chatbot = db.query(Chatbot).filter(
        Chatbot.chatbot_id == payload.chatbot_id
    ).first()
    
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Prevent people_analyzer submissions
    mode = str(chatbot.mode).lower() if chatbot.mode else ''
    if mode in ['people_analyzer', 'people-analyzer']:
        # Check for duplicate submission: same reviewer reviewing the same employee
        if payload.user_id and payload.employee_id:
            existing_review = db.query(PeopleAnalyzer).filter(
                PeopleAnalyzer.chatbot_id == uuid.UUID(payload.chatbot_id),
                PeopleAnalyzer.created_by == payload.user_id,
                PeopleAnalyzer.employee_id == payload.employee_id
            ).first()
            
            if existing_review:
                raise HTTPException(
                    status_code=400,
                    detail=f"You have already submitted a review for this employee. Duplicate submissions are not allowed."
                )
        
        new_answer = PeopleAnalyzer(
            id=uuid.uuid4(),
            chatbot_id=uuid.UUID(payload.chatbot_id),
            answers=payload.answers,
            created_by=payload.user_id,
            employee_id=payload.employee_id
        )
    else:
        # Check for duplicate submission: same employee submitting for the same quiz/chatbot
        if payload.user_id:
            existing_answer = db.query(Answer).filter(
                Answer.chatbot_id == uuid.UUID(payload.chatbot_id),
                Answer.attempter_by_id == payload.user_id
            ).first()
            
            if existing_answer:
                raise HTTPException(
                    status_code=400,
                    detail=f"You have already submitted a response for this quiz. Duplicate submissions are not allowed."
                )
        
        new_answer = Answer(
            id=uuid.uuid4(),
            chatbot_id=uuid.UUID(payload.chatbot_id),
            answer_data=payload.answers,
            attempter_by_id=payload.user_id,
            chat_history=[]
        )
    
    db.add(new_answer)
    db.commit()
    db.refresh(new_answer)
    
    return {
        "message": "Quiz submitted successfully",
        "quiz_id": str(new_answer.id),
        "chatbot_id": payload.chatbot_id,
        "answers_submitted": len(payload.answers)
    }

@router.get("/supported-formats", summary="Supported document formats")
async def get_supported_formats():
    """Get supported document formats."""
    return {
        "formats": [
            {"extension": "pdf", "description": "PDF"},
            {"extension": "docx", "description": "Word (2007+)"},
            {"extension": "doc", "description": "Word (Legacy)"},
            {"extension": "txt", "description": "Plain Text"},
            {"extension": "rtf", "description": "Rich Text"}
        ],
        "extensions": get_supported_extensions()
    }
    

@router.post("/quiz/render", summary="Render quiz questions from document")
async def upload_and_render_quiz_document(
    file: UploadFile = File(..., description="Quiz document (PDF, DOCX, DOC, TXT, RTF)")
):
   
    try:
        log.info(f" Quiz render request for: {file.filename}")
        
        # Render questions (no file storage)
        rendered_questions = await render_quiz_questions(file)
        
        result = {
            "file_name": file.filename,
            "total_questions": len(rendered_questions),
            "questions": rendered_questions,
            "message": f"Successfully extracted {len(rendered_questions)} questions"
        }
        
        log.info(f"✅ Quiz render complete: {len(rendered_questions)} questions from {file.filename}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Quiz render endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/responses/{chatbot_id}")
def get_chatbot_responses(
    request: Request,
    chatbot_id: uuid.UUID,
    department: Optional[str] = None,  # New filter
    db: Session = Depends(get_db),
    
):
    """Get responses for a chatbot based on mode with optional department filter."""
    # user_info = get_current_employee_from_token(request, db)
    # print(user_info,"----------------------------------")
    
    return get_chatbot_responses_service(chatbot_id, department, db)


@router.get("/evaluation/{chatbot_id}")
def get_employee_evaluation(
    request: Request,
    chatbot_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Get all employee evaluation data for people analyzer by chatbot_id."""
    # Optional: Verify user has permission to view this data
    # user_info = get_current_employee_from_token(request, db)
    
    return get_employee_evaluation_service(chatbot_id, db)



   #Token Cache
_kestrel_token_cache = {
    "token": None,
    "expires_at": 0
}

#  Reuse one HTTP client
_kestrel_client = httpx.AsyncClient(
timeout=httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=20.0)
)

SUPPORTED_SORT_BY = {"name", "email", "employeeCode", "teamName"}
SUPPORTED_SORT_ORDER = {"ASC", "DESC"}


async def get_kestrel_token() -> str:
    """
    Step 1: Generate Authorization token from Kestrel
    POST /generate-key
    Uses caching to avoid repeated token generation.
    """
    if not KESTREL_BASE_URL or not KESTREL_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="KESTREL_BASE_URL or KESTREL_API_KEY missing in env"
        )

    #  If token exists and not expired -> reuse
    now = time.time()
    if _kestrel_token_cache["token"] and now < _kestrel_token_cache["expires_at"]:
        return _kestrel_token_cache["token"]

    url = f"{KESTREL_BASE_URL.rstrip('/')}/generate-key"
    payload = {"apiKey": KESTREL_API_KEY}

    try:
        resp = await _kestrel_client.post(url, json=payload)

        if resp.status_code != 200:
            log.error(f"Kestrel generate-key failed: {resp.status_code} | {resp.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Kestrel generate-key failed ({resp.status_code})"
            )

        data = resp.json()
        encrypted_key = data.get("encryptedKey")

        if not encrypted_key:
            raise HTTPException(status_code=502, detail="Kestrel returned empty encryptedKey")

        #  cache token for 25 minutes (adjust as per kestrel TTL)
        _kestrel_token_cache["token"] = encrypted_key
        _kestrel_token_cache["expires_at"] = now + (25 * 60)

        return encrypted_key

    except httpx.RequestError as e:
        log.error(f"Kestrel connection error (generate-key): {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Unable to connect to Kestrel")

    except Exception as e:
        log.error(f"Unexpected error in get_kestrel_token: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/kestrel/employee-list", summary="Fetch Employee List from Kestrel")
async def kestrel_employee_list(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=1000),
    search: str = Query("", description="Search by name, email, or employee code"),
    sortBy: str = Query("name", description="Supported: name, email, employeeCode, teamName"),
    sortOrder: str = Query("ASC", description="Supported: ASC or DESC"),
):
    """
    Proxy endpoint:
    Frontend calls this.
    Backend fetches token from Kestrel (cached) and returns employee list.
    """

    #  Validate inputs before calling kestrel
    if sortBy not in SUPPORTED_SORT_BY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sortBy. Supported: {list(SUPPORTED_SORT_BY)}"
        )

    sortOrder = sortOrder.upper()
    if sortOrder not in SUPPORTED_SORT_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sortOrder. Supported: {list(SUPPORTED_SORT_ORDER)}"
        )

    token = await get_kestrel_token()
    url = f"{KESTREL_BASE_URL.rstrip('/')}/employee-list"

    params = {
        "page": page,
        "limit": limit,
        "search": search,
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }

    headers = {"Authorization": token}

    try:
        resp = await _kestrel_client.get(url, params=params, headers=headers)

        #  if unauthorized - refresh token and retry once
        if resp.status_code == 401:
            _kestrel_token_cache["token"] = None
            _kestrel_token_cache["expires_at"] = 0

            token = await get_kestrel_token()
            headers = {"Authorization": token}
            resp = await _kestrel_client.get(url, params=params, headers=headers)

        if resp.status_code != 200:
            log.error(f"Kestrel employee-list failed: {resp.status_code} | {resp.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Kestrel employee-list failed ({resp.status_code})"
            )

        return resp.json()

    except httpx.RequestError as e:
        log.error(f"Kestrel connection error (employee-list): {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Unable to connect to Kestrel")

    except Exception as e:
        log.error(f"Unexpected error in kestrel_employee_list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/kestrel/sync-employees", summary="Sync Kestrel Employees with DB")
async def sync_kestrel_employees(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Frontend refresh trigger hits this API.
    It syncs Kestrel employees (page+limit) with your local Employee table.
    """
    return await sync_kestrel_employees_service(db=db, page=page, limit=limit)

