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

# Import functional services
from services.chatbots_services import (
    create_chatbot,
    get_chatbot_responses_service,
    update_chatbot,
    delete_chatbot_by_id
)
from services.rag_service import generate_rag_response
from services.quiz_services import quiz_query_service
from utils.token_decode import get_current_employee_from_token

router = APIRouter()


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
    
    # Create ChatbotPermission entries for selected employees
    if parsed_employee_ids and len(parsed_employee_ids) > 0:
        log.info(f"Creating permissions for {len(parsed_employee_ids)} employees")
        for emp_id_str in parsed_employee_ids:
            try:
                # Find the employee by employee_id (string) to get the UUID (id)
                employee = db.query(Employee).filter(Employee.employee_id == str(emp_id_str)).first()
                if employee:
                    # Check if permission already exists
                    existing = db.query(ChatbotPermission).filter(
                        ChatbotPermission.chatbot_id == chatbot.chatbot_id,
                        ChatbotPermission.employee_id == employee.id
                    ).first()
                    
                    if not existing:
                        permission = ChatbotPermission(
                            chatbot_id=chatbot.chatbot_id,
                            employee_id=employee.id,
                            created_by=generated_by_uuid
                        )
                        db.add(permission)
                        log.info(f"✓ Permission created for employee {emp_id_str}")
                    else:
                        log.info(f"Permission already exists for employee {emp_id_str}")
                else:
                    log.warning(f"Employee not found: {emp_id_str}")
            except Exception as e:
                log.error(f"Error creating permission for employee {emp_id_str}: {e}")
        
        db.commit()
    
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
    payload: ChatbotUpdate,
    db: Session = Depends(get_db)
):
    """Update chatbot settings."""

    chatbot = update_chatbot(
        db=db,
        chatbot_id=uuid.UUID(chatbot_id),
        status=payload.status,
        description=payload.description,
        instruction=payload.instruction,
        mode=payload.mode,
        questions=payload.questions,
        employee_ids=payload.employee_ids,
        access_list=payload.access_list,
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
        
        # Get allowed users
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
    """
    Submit quiz answers with proper validation.
    Handles both quiz and people_analyzer modes.
    Never returns 500 for user input errors.
    """

    try:
        # Validate chatbot exists
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_id == payload.chatbot_id
        ).first()
        
        if not chatbot:
            print(f"❌ Chatbot not found: {payload.chatbot_id}")
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        print(f"✅ Chatbot found: {chatbot.chatbot_name} (mode: {chatbot.mode})")
        
        # Validate user_id
        if not payload.user_id:
            print(f"❌ Missing user_id")
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Validate chatbot_id format
        try:
            chatbot_uuid = uuid.UUID(payload.chatbot_id)
            print(f"✅ chatbot_uuid: {chatbot_uuid}")
        except (ValueError, AttributeError) as e:
            log.error(f"Invalid chatbot_id: {payload.chatbot_id}, error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid chatbot_id format: {payload.chatbot_id}")
        
        # Validate user_id format (more lenient - try to convert)
        try:
            user_uuid = uuid.UUID(str(payload.user_id))
        except (ValueError, AttributeError) as e:
            log.error(f"Invalid user_id: {payload.user_id}, error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid user_id format: {payload.user_id}")
        
        # Query the user from DB to get their actual ID (UUID primary key)
        user = db.query(Employee).filter(
            Employee.employee_id == str(user_uuid)
        ).first()
        
        if not user:
            print(f"❌ User not found: {user_uuid}")
            raise HTTPException(status_code=404, detail=f"User with ID {user_uuid} not found")
        
        print(f"✅ User found: {user.employee_name} (DB ID: {user.id})")
        user_db_id = user.id  # Use the actual DB primary key
        
        # Get chatbot mode
        mode = str(chatbot.mode).lower() if chatbot.mode else 'quiz'
        log.info(f"📝 Quiz submission for '{chatbot.chatbot_name}' (mode: {mode})")
        log.info(f"Payload: chatbot_id={payload.chatbot_id}, user_id={payload.user_id}, employee_id={payload.employee_id}")
        
        # Validate answers (allow empty answers for skipped questions)
        if not isinstance(payload.answers, dict):
            log.error(f"Answers not a dict: {type(payload.answers)}")
            raise HTTPException(status_code=400, detail=f"Answers must be a dictionary, got {type(payload.answers)}")
        
        # Clean and validate answers
        cleaned_answers = {}
        for question_id, answer_data in payload.answers.items():
            # Handle both formats: {"answer": value} or direct value
            if isinstance(answer_data, dict) and "answer" in answer_data:
                cleaned_answers[question_id] = answer_data
            else:
                cleaned_answers[question_id] = {"answer": answer_data}
        
        # People Analyzer mode - requires employee_id
        if mode in ['people_analyzer', 'people-analyzer']:
            if not payload.employee_id:
                log.error("Missing employee_id for people_analyzer mode")
                raise HTTPException(
                    status_code=400,
                    detail="employee_id is required for people analyzer mode"
                )
            
            try:
                employee_uuid = uuid.UUID(str(payload.employee_id))
            except (ValueError, AttributeError) as e:
                log.error(f"Invalid employee_id: {payload.employee_id}, error: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid employee_id format: {payload.employee_id}")
            
            # Validate employee exists (query by string employee_id field, not UUID)
            employee = db.query(Employee).filter(
                Employee.id == str(employee_uuid)
            ).first()
            
            if not employee:
                log.error(f"Employee not found: {employee_uuid}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Employee with ID {employee_uuid} not found. Please ensure the employee exists."
                )
            
            log.info(f"✅ Employee found: {employee.employee_name}")
            log.info(f"📊 People analyzer submission for employee: {employee.employee_name} (DB ID: {employee.id})")
            
            # Use employee.id (UUID primary key) for the foreign key
            new_answer = PeopleAnalyzer(
                id=uuid.uuid4(),
                chatbot_id=chatbot_uuid,
                answers=cleaned_answers,
                created_by=user_db_id,  # Use the DB primary key
                employee_id=employee.id  # Use the UUID primary key from DB
            )
        else:
            # Regular quiz mode
            log.info(f"✅ Quiz submission with {len(cleaned_answers)} answers")
            
            new_answer = Answer(
                id=uuid.uuid4(),
                chatbot_id=chatbot_uuid,
                answer_data=cleaned_answers,
                attempter_by_id=user_db_id,  # Use the DB primary key
                chat_history=[]
            )
        
        # Save to database
        db.add(new_answer)
        db.commit()
        db.refresh(new_answer)
        
        log.info(f"✅ Submission saved successfully: {new_answer.id}")
        
        return {
            "message": "Review submitted successfully!" if mode == 'people_analyzer' else "Quiz submitted successfully!",
            "submission_id": str(new_answer.id),
            "chatbot_id": payload.chatbot_id,
            "answers_submitted": len(cleaned_answers),
            "mode": mode
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Quiz submission error: {e}", exc_info=True)
        # Return detailed error for debugging
        raise HTTPException(
            status_code=400,
            detail=f"Submission failed: {str(e)}"
        )


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