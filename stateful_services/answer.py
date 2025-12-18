"""
Answer API with Parameter-Based Ownership
=================================================================

This module implements a strict answer management system with:
1. Parameter-based ownership (source_type)
2. Complete question data returns
3. Question update handling
4. Full data traceability
"""

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from stateful_services.db_schema import Answer, Question, Chatbot, Employee
from models.user import User
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter()

# ===================== HELPERS =====================

def get_current_employee(request: Request, db: Session) -> Optional[Employee]:
    """
    Extract logged-in employee from authorization token.
    Returns Employee object if authenticated, None otherwise.
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.replace("Bearer ", "")
        
        # Import auth service to verify token
        from auth.auth_service import MicrosoftAuthService
        auth_service = MicrosoftAuthService()
        payload = auth_service.verify_jwt_token(token)
        
        if not payload:
            return None
        
        # Get user email from token
        email = payload.get("email")
        if not email:
            return None
        
        # Try to find employee by email
        employee = db.query(Employee).filter(Employee.employee_email == email).first()
        
        # If not found, try finding user and create/link employee
        if not employee:
            user = db.query(User).filter(User.email == email).first()
            if user:
                # Create employee record from user
                employee = Employee(
                    employee_id=str(user.microsoft_id or user.id),
                    employee_name=user.name or email.split('@')[0],
                    employee_email=user.email,
                    employee_role="user"
                )
                db.add(employee)
                db.commit()
                db.refresh(employee)
        
        return employee
    except Exception as e:
        print(f"Error getting current employee: {e}")
        return None

# ===================== ROUTES =====================

@router.post("/submit", tags=["Answers"])
def submit_answer(
    request: Request,
    chatbot_id: uuid.UUID = Body(...),
    question_id: uuid.UUID = Body(...),
    answer_content: str = Body(...),
    source_type: str = Body("quiz"),  # chatbot, FAQ, support_ticket, knowledge_base, quiz
    attempter_id: Optional[uuid.UUID] = Body(None),
    attempter_name: Optional[str] = Body(None),
    answer_metadata: Optional[Dict[str, Any]] = Body(None),
    db: Session = Depends(get_db)
):
    """Submit an answer with full ownership tracking - auto-captures logged-in employee."""
    # Auto-capture logged-in employee
    current_employee = get_current_employee(request, db)
    employee_details = None
    if current_employee:
        attempter_id = current_employee.id if isinstance(current_employee.id, uuid.UUID) else uuid.UUID(str(current_employee.id))
        attempter_name = str(current_employee.employee_name)
        employee_details = {
            "employee_id": str(current_employee.id),
            "employee_name": current_employee.employee_name,
            "employee_email": current_employee.employee_email,
            "employee_role": current_employee.employee_role
        }
    
    return submit_answer_service(
        chatbot_id=chatbot_id,
        question_id=question_id,
        answer_content=answer_content,
        source_type=source_type,
        attempter_id=attempter_id,
        attempter_name=attempter_name,
        answer_metadata=answer_metadata,
        employee_details=employee_details,
        db=db
    )


@router.post("/user/quiz/chatbot/{chatbot_id}/answer")
def submit_quiz_answer(request: Request, chatbot_id: uuid.UUID, body: dict, db: Session = Depends(get_db)):
    """Submit quiz answer with auto employee tracking - validates by chatbot_id."""
    question_id = body.get("question_id")
    if not question_id:
        raise HTTPException(status_code=400, detail="question_id is required in request body")
    
    try:
        question_id = uuid.UUID(str(question_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid question_id format")
    
    # Auto-capture logged-in employee
    current_employee = get_current_employee(request, db)
    attempter_id = body.get("attempter_id")
    attempter_name = body.get("attempter_name")
    employee_details = None
    
    if current_employee:
        attempter_id = current_employee.id if isinstance(current_employee.id, uuid.UUID) else uuid.UUID(str(current_employee.id))
        attempter_name = str(current_employee.employee_name)
        employee_details = {
            "employee_id": str(current_employee.id),
            "employee_name": current_employee.employee_name,
            "employee_email": current_employee.employee_email,
            "employee_role": current_employee.employee_role
        }
    
    # Ensure attempter_name is a string, not a Column object
    attempter_name = str(attempter_name) if attempter_name is not None else None
    
    return submit_answer_service(
        chatbot_id=chatbot_id,
        question_id=question_id,
        answer_content=str(body.get("answer", "")),
        source_type="quiz",
        attempter_id=attempter_id,
        attempter_name=attempter_name,
        answer_metadata=body.get("answer") if isinstance(body.get("answer"), dict) else None,
        employee_details=employee_details,
        db=db
    )


@router.post("/user/quiz/chatbot/{chatbot_id}/skip")
def skip_question(request: Request, chatbot_id: uuid.UUID, body: dict, db: Session = Depends(get_db)):
    """Skip a question - recorded as an answer with skip status, validates by chatbot_id."""
    question_id = body.get("question_id")
    if not question_id:
        raise HTTPException(status_code=400, detail="question_id is required in request body")
    
    try:
        question_id = uuid.UUID(str(question_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid question_id format")
    
    # Auto-capture logged-in employee
    current_employee = get_current_employee(request, db)
    employee_details = None
    if current_employee:
        body["attempter_id"] = current_employee.id if isinstance(current_employee.id, uuid.UUID) else uuid.UUID(str(current_employee.id))
        body["attempter_name"] = str(current_employee.employee_name)
        employee_details = {
            "employee_id": str(current_employee.id),
            "employee_name": current_employee.employee_name,
            "employee_email": current_employee.employee_email,
            "employee_role": current_employee.employee_role
        }
    
    return skip_question_service(chatbot_id, question_id, body, employee_details, db)


@router.get("/chatbot/{chatbot_id}", tags=["Answers"])
def get_chatbot_answers(
    chatbot_id: uuid.UUID,
    source_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all answers for a chatbot with full details."""
    return get_answers_by_chatbot(chatbot_id, source_type, db)


@router.get("/user/quiz/chatbot/{chatbot_id}/answers")
def get_answers(chatbot_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get answers for a chatbot with employee details."""
    return get_user_answers_by_chatbot(chatbot_id, db)


@router.get("/admin/chatbot/{chatbot_id}/attempts")
def admin_attempts(chatbot_id: uuid.UUID, db: Session = Depends(get_db)):
    """Admin view of all attempts for a chatbot with full employee details."""
    return get_admin_attempts_by_chatbot(chatbot_id, db)


# ===================== SERVICES =====================

def get_question_full_data(question_id: uuid.UUID, db: Session) -> Dict[str, Any]:
    """
    Retrieve COMPLETE question data - never just the ID.
    Returns reformulated question if it was updated.
    """
    question = db.query(Question).filter(Question.question_id == question_id).first()
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Get chatbot info for context
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == question.chatbot_id).first()
    
    return {
        "question_id": str(question.question_id),
        "chatbot_id": str(question.chatbot_id),
        "chatbot_name": chatbot.chatbot_name if chatbot else None,
        "chatbot_mode": chatbot.mode if chatbot else None,
        "status": question.status,
        "question_text": question.question_data.get("question") if isinstance(question.question_data, dict) else None,
        "question_data": question.question_data,
        "created_at": question.created_at.isoformat() if hasattr(question.created_at, 'isoformat') else None,
        "metadata": {
            "type": question.question_data.get("type") if isinstance(question.question_data, dict) else None,
            "options": question.question_data.get("options") if isinstance(question.question_data, dict) else None,
            "difficulty": question.question_data.get("difficulty") if isinstance(question.question_data, dict) else None
        }
    }


def submit_answer_service(
    chatbot_id: uuid.UUID,
    question_id: uuid.UUID,
    answer_content: str,
    source_type: str,
    attempter_id: Optional[uuid.UUID],
    attempter_name: Optional[str],
    answer_metadata: Optional[Dict[str, Any]],
    employee_details: Optional[Dict[str, Any]],
    db: Session
) -> Dict[str, Any]:
    """
    Submit answer with FULL ownership tracking - validates by chatbot_id.
    
    Rule 1: Validate by chatbot_id, not question_id
    Rule 2: Employee details prominently in response
    Rule 3: Parameter-based ownership via source_type
    Rule 4: Storage with all metadata
    """
    try:
        # Validate chatbot exists
        chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Validate source_type
        valid_sources = ["chatbot", "FAQ", "support_ticket", "knowledge_base", "quiz"]
        if source_type not in valid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type. Must be one of: {', '.join(valid_sources)}"
            )
        
        # Validate question belongs to this chatbot
        question = db.query(Question).filter(
            Question.question_id == question_id,
            Question.chatbot_id == chatbot_id
        ).first()
        if not question:
            raise HTTPException(
                status_code=404,
                detail=f"Question not found in chatbot {chatbot_id}"
            )
        
        # Get COMPLETE question data
        question_data = get_question_full_data(question_id, db)
        
        # Default answer content if empty
        if not answer_content:
            answer_content = ""
        
        # Get chatbot_id from question data
        answer_chatbot_id: Optional[uuid.UUID] = None
        if question_data.get("chatbot_id"):
            try:
                answer_chatbot_id = uuid.UUID(question_data["chatbot_id"])
            except (ValueError, TypeError):
                answer_chatbot_id = None
        
        # Create answer with full ownership and traceability
        answer = Answer(
            question_id=question_id,
            chatbot_id=answer_chatbot_id,
            attempter_id=attempter_id,
            attempter_name=attempter_name,
            session_id=None,  # No session tracking needed
            source_type=source_type,
            answer_content=answer_content,
            answer_data=answer_metadata or {},
            question_snapshot=question_data  # Store question state at time of answer
        )
        db.add(answer)
        db.commit()
        db.refresh(answer)
        
        # Get full employee details from database if not provided
        if not employee_details and attempter_id:
            employee = db.query(Employee).filter(Employee.id == attempter_id).first()
            if employee:
                employee_details = {
                    "employee_id": str(employee.id),
                    "employee_name": employee.employee_name,
                    "employee_email": employee.employee_email,
                    "employee_role": employee.employee_role
                }
        
        # Return COMPLETE structured response with employee details prominently
        return {
            "status": "success",
            "message": "Answer stored successfully",
            "employee": employee_details or {
                "employee_id": str(attempter_id) if attempter_id else None,
                "employee_name": attempter_name,
                "employee_email": None,
                "employee_role": None,
                "auto_tracked": False
            },
            "answer": {
                "answer_id": str(answer.id),
                "content": answer_content,
                "metadata": answer_metadata,
                "submitted_at": answer.created_at.isoformat()
            },
            "chatbot": {
                "chatbot_id": str(answer_chatbot_id) if answer_chatbot_id else str(chatbot_id),
                "chatbot_name": chatbot.chatbot_name,
                "mode": chatbot.mode
            },
            "question": {
                "question_id": question_data["question_id"],
                "question_text": question_data["question_text"],
                "question_data": question_data["question_data"],
                "metadata": question_data["metadata"]
            },
            "validation": {
                "validated_by": "chatbot_id",
                "source_type": source_type,
                "traceable": True
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_detail = f"Error submitting answer: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit answer: {str(e)}"
        )


def skip_question_service(chatbot_id: uuid.UUID, question_id: uuid.UUID, body: dict, employee_details: Optional[Dict[str, Any]], db: Session):
    """Skip question - records as answer with skip status, validates by chatbot_id."""
    return submit_answer_service(
        chatbot_id=chatbot_id,
        question_id=question_id,
        answer_content="SKIPPED",
        source_type="quiz",
        attempter_id=body.get("attempter_id"),
        attempter_name=body.get("attempter_name"),
        answer_metadata={"status": "skipped", "reason": body.get("reason")},
        employee_details=employee_details,
        db=db
    )


def get_answers_by_chatbot(
    chatbot_id: uuid.UUID,
    source_type: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """Get all answers for a chatbot WITH employee details."""
    # Validate chatbot exists
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Query answers with optional source filtering
    query = db.query(Answer).filter(Answer.chatbot_id == chatbot_id)
    if source_type:
        query = query.filter(Answer.source_type == source_type)
    
    answers = query.all()
    
    return {
        "chatbot": {
            "chatbot_id": str(chatbot_id),
            "chatbot_name": chatbot.chatbot_name,
            "mode": chatbot.mode
        },
        "total_answers": len(answers),
        "answers": [
            {
                "answer_id": str(a.id),
                "answer_content": a.answer_content,
                "answer_metadata": a.answer_data,
                "source_type": a.source_type,
                "employee": {
                    "employee_id": str(a.attempter_id) if hasattr(a.attempter_id, '__str__') and a.attempter_id is not None else None,
                    "employee_name": a.attempter_name
                },
                "timestamps": {
                    "created_at": a.created_at.isoformat() if hasattr(a.created_at, 'isoformat') else None,
                    "updated_at": a.updated_at.isoformat() if hasattr(a.updated_at, 'isoformat') else None
                },
                "question_snapshot": a.question_snapshot  # Question state at time of answer
            }
            for a in answers
        ]
    }


def get_user_answers_by_chatbot(chatbot_id: uuid.UUID, db: Session):
    """Get answers for a chatbot with employee details."""
    # Validate chatbot exists
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    # Return answers ordered by newest first and include answer_id and chatbot_id
    answers = (
        db.query(Answer)
        .filter(Answer.chatbot_id == chatbot_id)
        .order_by(Answer.created_at.desc())
        .all()
    )

    return {
        "chatbot_id": str(chatbot_id),
        "chatbot_name": chatbot.chatbot_name,
        "answers": [
            {
                "answer_id": str(a.id),
                "chatbot_id": str(a.chatbot_id) if a.chatbot_id else str(chatbot_id),
                "answer": a.answer_content,
                "answer_data": a.answer_data,
                "employee": {
                    "employee_id": str(a.attempter_id) if a.attempter_id else None,
                    "employee_name": a.attempter_name
                },
                "source_type": a.source_type,
                "created_at": a.created_at.isoformat() if hasattr(a.created_at, 'isoformat') else None
            }
            for a in answers
        ]
    }


@router.get("/chatbot/{chatbot_id}/latest", tags=["Answers"])
def get_latest_answers(chatbot_id: uuid.UUID, limit: int = 10, db: Session = Depends(get_db)):
    """Get the most recent answers for a chatbot.

    - `limit` controls how many recent answers to return (default 10).
    - Results are ordered newest first.
    """
    # Validate chatbot exists
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    # Query latest answers ordered by creation time desc
    answers = (
        db.query(Answer)
        .filter(Answer.chatbot_id == chatbot_id)
        .order_by(Answer.created_at.desc())
        .limit(int(limit))
        .all()
    )

    return {
        "chatbot_id": str(chatbot_id),
        "chatbot_name": chatbot.chatbot_name,
        "latest_count": len(answers),
        "answers": [
            {
                "answer_id": str(a.id),
                "answer_content": a.answer_content,
                "answer_metadata": a.answer_data,
                "source_type": a.source_type,
                "employee": {
                    "employee_id": str(a.attempter_id) if a.attempter_id else None,
                    "employee_name": a.attempter_name
                },
                "created_at": a.created_at.isoformat() if hasattr(a.created_at, 'isoformat') else None,
                "question_snapshot": a.question_snapshot
            }
            for a in answers
        ]
    }


def get_admin_attempts_by_chatbot(chatbot_id: uuid.UUID, db: Session):
    """Admin view with complete traceability - validates by chatbot_id."""
    # Validate chatbot exists
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    answers = db.query(Answer).filter(Answer.chatbot_id == chatbot_id).all()
    
    # Get employee details for each answer
    attempts_with_employees = []
    for a in answers:
        employee = db.query(Employee).filter(Employee.id == a.attempter_id).first() if a.attempter_id else None
        attempts_with_employees.append({
            "answer_id": str(a.id),
            "answer_content": a.answer_content,
            "answer_data": a.answer_data,
            "source_type": a.source_type,
            "employee": {
                "employee_id": str(a.attempter_id) if a.attempter_id else None,
                "employee_name": employee.employee_name if employee else a.attempter_name,
                "employee_email": employee.employee_email if employee else None,
                "employee_role": employee.employee_role if employee else None
            },
            "timestamps": {
                "created_at": a.created_at.isoformat() if hasattr(a.created_at, 'isoformat') else None,
                "updated_at": a.updated_at.isoformat() if hasattr(a.updated_at, 'isoformat') else None
            },
            "question_snapshot": a.question_snapshot
        })

    return {
        "chatbot": {
            "chatbot_id": str(chatbot_id),
            "chatbot_name": chatbot.chatbot_name,
            "mode": chatbot.mode
        },
        "total_attempts": len(answers),
        "attempts": attempts_with_employees
    }
