# questions.py
# All question operations return COMPLETE question data
# Never returns just question ID - always includes full question details

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from stateful_services.db_schema import Question, Chatbot
from sqlalchemy.dialects.postgresql import UUID
from typing import Dict, Any
import uuid

router = APIRouter()

# ======================================================
#                       ROUTES
# ======================================================

@router.get("/admin/questions/{chatbot_id}")
def route_get_all_questions(chatbot_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get all questions for a chatbot with complete data."""
    return get_all_questions(chatbot_id, db)


@router.get("/admin/question/{question_id}")
def route_get_question(question_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get a single question with complete data."""
    return get_question_complete_data(question_id, db)


@router.post("/admin/questions/{chatbot_id}")
def route_create_question(chatbot_id: uuid.UUID, body: dict, db: Session = Depends(get_db)):
    """Create a question and return complete data."""
    return create_question(chatbot_id, body, db)


@router.put("/admin/question/{question_id}")
def route_update_question(question_id: uuid.UUID, body: dict, db: Session = Depends(get_db)):
    """Update a question and return complete updated data."""
    return update_question(question_id, body, db)


@router.delete("/admin/question/{question_id}")
def route_delete_question(question_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a question and return complete data of deleted question."""
    return delete_question(question_id, db)


def format_complete_question_data(question: Question, chatbot: Chatbot = None) -> Dict[str, Any]:
    """
    Format complete question data - NEVER return just ID.
    Returns all question details with chatbot context.
    """
    return {
        "question_id": str(question.question_id),
        "chatbot_id": str(question.chatbot_id),
        "chatbot_name": chatbot.chatbot_name if chatbot else None,
        "chatbot_mode": chatbot.mode if chatbot else None,
        "chatbot_description": chatbot.description if chatbot else None,
        "status": question.status,
        "question_text": question.question_data.get("question") if isinstance(question.question_data, dict) else None,
        "question_data": question.question_data,
        "metadata": {
            "type": question.question_data.get("type") if isinstance(question.question_data, dict) else None,
            "options": question.question_data.get("options") if isinstance(question.question_data, dict) else None,
            "difficulty": question.question_data.get("difficulty") if isinstance(question.question_data, dict) else None,
            "points": question.question_data.get("points") if isinstance(question.question_data, dict) else None,
            "correct_answer": question.question_data.get("correct_answer") if isinstance(question.question_data, dict) else None,
        },
        "created_at": question.created_at.isoformat() if question.created_at else None
    }


def get_question_complete_data(question_id: uuid.UUID, db: Session) -> Dict[str, Any]:
    """
    Get COMPLETE question data for a single question.
    Returns full question details with chatbot information.
    """
    question = db.query(Question).filter(Question.question_id == question_id).first()
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == question.chatbot_id).first()
    
    return {
        "question": format_complete_question_data(question, chatbot)
    }


def get_all_questions(chatbot_id: uuid.UUID, db: Session):
    """Get all questions with COMPLETE data for each question."""
    questions = db.query(Question).filter(
        Question.chatbot_id == chatbot_id
    ).all()
    
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()

    return {
        "chatbot_id": str(chatbot_id),
        "chatbot_name": chatbot.chatbot_name if chatbot else None,
        "chatbot_mode": chatbot.mode if chatbot else None,
        "total_questions": len(questions),
        "questions": [
            format_complete_question_data(q, chatbot)
            for q in questions
        ]
    }


def create_question(chatbot_id: uuid.UUID, body: dict, db: Session):
    """Create question and return COMPLETE question data."""
    if not isinstance(body, dict) or not body:
        raise HTTPException(status_code=400, detail="Invalid question payload")

    question = Question(
        chatbot_id=chatbot_id,
        question_data=body
    )

    db.add(question)
    db.commit()
    db.refresh(question)
    
    # Get chatbot info
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()

    return {
        "status": "success",
        "message": "Question created successfully",
        "question": format_complete_question_data(question, chatbot)
    }


def update_question(question_id: uuid.UUID, body: dict, db: Session):
    """Update question and return COMPLETE updated question data."""
    question = db.query(Question).filter(
        Question.question_id == question_id
    ).first()

    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Store original data
    original_data = question.question_data.copy() if isinstance(question.question_data, dict) else {}

    # Update status if provided
    if "status" in body:
        question.status = body["status"]
        body = {k: v for k, v in body.items() if k != "status"}

    # Update question data
    if body:
        if isinstance(question.question_data, dict):
            question.question_data.update(body)
        else:
            question.question_data = body

    db.commit()
    db.refresh(question)
    
    # Get chatbot info
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == question.chatbot_id).first()

    return {
        "status": "success",
        "message": "Question updated successfully",
        "original_question_data": original_data,
        "updated_question": format_complete_question_data(question, chatbot),
        "changes_applied": body
    }


def delete_question(question_id: uuid.UUID, db: Session):
    """Delete question and return COMPLETE data of deleted question."""
    question = db.query(Question).filter(
        Question.question_id == question_id
    ).first()

    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Get chatbot info before deletion
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == question.chatbot_id).first()
    
    # Store complete question data before deletion
    deleted_question_data = format_complete_question_data(question, chatbot)

    db.delete(question)
    db.commit()

    return {
        "status": "success",
        "message": "Question deleted successfully",
        "deleted_question": deleted_question_data
    }
